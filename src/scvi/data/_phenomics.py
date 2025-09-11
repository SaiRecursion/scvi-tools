"""Data loading and AnnData setup utilities for phenomics embeddings.

This module focuses on preparation of phenomics data
for downstream non-linear batch correction with PHVI (a VAE-style model). It:

- Loads a parquet table where feature columns share a prefix (e.g., ``feature_``)
  and metadata columns include batch/covariate information (e.g., ``plate_number``,
  ``experiment``, etc.).
- Builds an :class:`anndata.AnnData` object with:
    - ``X`` = continuous phenomics features (``float32``),
    - ``obs`` = metadata,
    - ``var`` = feature names.
- Optionally creates a **vectorized** ``perturbation_summary`` column to summarize
  genes and compounds, avoiding per-row ``apply`` (critical for large tables).
- **No control-only training behavior** is embedded here by design. Training on
  *all* wells preserves biological variation and supports higher BMDB.
- Provides a convenience wrapper to register the AnnData with ``scvi.model.PHVI``.
  It forwards a **hierarchical batch** option (e.g., ``experiment × plate_number``),
  which should match how your KNN batch-predictability metric is defined.

Typical usage:
--------------
>>> adata = read_phenomics("phenom2_embeddings.parquet", feature_prefix="feature_")
>>> PHVI.setup_anndata(adata)
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from anndata import AnnData

logger = logging.getLogger(__name__)


def _safe_text(obs, key: str, default: str) -> pd.Series:
    """Helper function to safely extract text from obs."""
    s = obs.get(key)
    if s is None:
        return pd.Series([default] * len(obs), index=obs.index, dtype="string")
    s = s.astype("string")
    return s.fillna(default)

# --------------------------------------------------------------------------- #
# Core loader
# --------------------------------------------------------------------------- #
def read_phenomics_from_df(
    df: pd.DataFrame,
    *,
    feature_prefix: str = "feature_",
    batch_key: str = "experiment",
    create_perturbation_summary: bool = True,
    standardize: bool = True,
    raw_layer_name: str = "X_raw",
    hierarchical_batch_keys: list[str] | None = None,
) -> AnnData:
    """Build AnnData directly from an in-memory DataFrame (skip parquet IO).
    
    Parameters
    ----------
    df
        Input DataFrame with feature columns and metadata columns.
    feature_prefix
        Prefix for feature columns (e.g., "feature_").
    batch_key
        Column name for batch information.
    create_perturbation_summary
        Whether to create a perturbation_summary column from metadata.
    standardize
        Whether to standardize features using StandardScaler.
    raw_layer_name
        Name for storing raw (unstandardized) features in layers.
    hierarchical_batch_keys
        If provided, must be a list of exactly two keys in df columns to be
        concatenated into a hierarchical batch key. This mirrors (experiment x plate)
        which is what KNN evaluators typically use. Example: ['experiment', 'plate_number'].
        If provided, this overrides the batch_key parameter.
        
    Returns
    -------
    AnnData object with continuous phenomics features.
    """
    feature_cols = [c for c in df.columns if str(c).startswith(feature_prefix)]
    if not feature_cols:
        raise ValueError(f"No feature columns found with prefix '{feature_prefix}'.")
    meta_cols = [c for c in df.columns if c not in feature_cols]

    feature_df = df[feature_cols].copy()
    for c in feature_df.columns:
        if pd.api.types.is_categorical_dtype(feature_df[c].dtype):
            feature_df[c] = pd.to_numeric(feature_df[c].astype("string"), errors="coerce")
    X = feature_df.to_numpy(copy=False).astype("float32", copy=False)
    obs = df[meta_cols].copy()

    # Build a concatenated hierarchical batch column if requested
    if hierarchical_batch_keys is not None:
        if len(hierarchical_batch_keys) != 2:
            raise ValueError("`hierarchical_batch_keys` must be a list of exactly two keys.")
        key1, key2 = hierarchical_batch_keys
        if key1 not in obs.columns or key2 not in obs.columns:
            raise ValueError(f"One or both keys '{key1}', '{key2}' not found in obs columns.")

        # Create a deterministic combined key; this column name is logged and then used as batch.
        new_batch_col_name = f"{key1}_{key2}"
        logger.info(f"[read_phenomics_from_df] Creating hierarchical batch key '{new_batch_col_name}' from '{key1}' and '{key2}'.")
        obs[new_batch_col_name] = (
            obs[key1].astype(str) + "_" + obs[key2].astype(str)
        )
        if obs[new_batch_col_name].dtype.name != "category":
            obs[new_batch_col_name] = obs[new_batch_col_name].astype("category")
        logger.info(f"[read_phenomics_from_df] Using '{new_batch_col_name}' as the final batch key.")
        batch_key = new_batch_col_name

    # vectorized perturbation summary (same logic as read_phenomics)
    if create_perturbation_summary and "map_perturbation_type" in obs.columns:
        pt = obs["map_perturbation_type"].astype(str).str.lower()
        gene = _safe_text(obs, "map_gene", "GENE_UNKNOWN")
        
        # Check for missing compound fields
        if 'map_rec_id' in obs.columns and 'map_concentration' in obs.columns:
            rec = _safe_text(obs, "map_rec_id", "UNKNOWN_COMPOUND")
            conc = _safe_text(obs, "map_concentration", "UNKNOWN_CONC")
            comp = rec.str.cat(conc, sep="_")
        else:
            comp = pd.Series(["COMPOUND_UNKNOWN"] * len(obs), index=obs.index, dtype="string")
            logger.warning("Missing compound fields (map_rec_id/map_concentration) for perturbation summary")
        
        # Check for unexpected perturbation types
        expected_types = {"gene", "compound", "empty"}
        unique_types = set(pt.unique())
        unexpected = unique_types - expected_types
        if unexpected:
            logger.warning(f"Found unexpected perturbation types: {unexpected}")
        
        vals = np.select(
            [pt.eq("gene"), pt.eq("compound"), pt.eq("empty")],
            [gene,          comp,              "CONTROL_EMPTY"],
            default="unknown",
        )
        obs["perturbation_summary"] = pd.Series(vals, index=obs.index, dtype="category")

    if batch_key not in obs.columns:
        raise ValueError(f"Batch key '{batch_key}' not found in columns.")

    var = pd.DataFrame(index=pd.Index(feature_cols, name="feature_name"))
    adata = AnnData(X=X, obs=obs, var=var)
    adata.obs_names = adata.obs_names.astype(str); adata.obs_names_make_unique()
    adata.var_names_make_unique()
    adata.uns.update({
        "data_type": "phenomics",
        "feature_type": "continuous",
        "n_features": int(len(feature_cols)),
        "feature_prefix": feature_prefix,
        "simple_batch_key": batch_key,
    })
    if standardize:
        adata.layers[raw_layer_name] = adata.X.copy()
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        adata.X = scaler.fit_transform(adata.X).astype("float32")
        adata.uns["scaler_mean_"] = scaler.mean_.astype("float32")
        adata.uns["scaler_scale_"] = scaler.scale_.astype("float32")
    logger.info(f"[read_phenomics_from_df] Created AnnData with shape: {adata.shape}")
    return adata


def validate_phenomics_distribution(adata):
    """Check if features follow Gaussian assumptions.
    
    Parameters
    ----------
    adata
        AnnData object with phenomics features in X.
        
    Notes
    -----
    Tests a sample of features using the Shapiro-Wilk test for normality.
    Warns if more than half of tested features appear non-Gaussian.
    """
    from scipy import stats
    
    # Sample features to test
    n_test = min(10, adata.n_vars)
    sample_indices = np.random.choice(adata.n_vars, n_test, replace=False)
    
    non_gaussian_count = 0
    for idx in sample_indices:
        feature_data = adata.X[:, idx]
        # Limit to 1000 samples for Shapiro-Wilk (it has sample size limits)
        n_samples = min(1000, len(feature_data))
        _, p_value = stats.shapiro(feature_data[:n_samples])
        if p_value < 0.05:
            non_gaussian_count += 1
    
    if non_gaussian_count > n_test / 2:
        logger.warning(
            f"{non_gaussian_count}/{n_test} tested features appear non-Gaussian. "
            "Consider using a different likelihood or transforming the data."
        )
