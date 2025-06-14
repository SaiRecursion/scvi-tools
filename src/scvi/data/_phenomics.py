"""Data loading functionality for phenomics data."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from anndata import AnnData

logger = logging.getLogger(__name__)


def read_phenomics(
    filepath: str | Path,
    feature_prefix: str = "feature_",
    batch_key: str = "plate_number",
    create_perturbation_summary: bool = True,
    control_only_training: bool = False,
    control_well_type: str = "center_introns_v1",
    subset_rows: int | None = None,
    subset_seed: int | None = None,
) -> AnnData:
    """Read phenomics data from parquet file.

    Parameters
    ----------
    filepath
        Path to the parquet file containing phenomics data.
    feature_prefix
        Prefix used to identify feature columns.
    batch_key
        Column name to use as batch information (default: "plate_number").
    create_perturbation_summary
        Whether to create a perturbation_summary column from map_* columns.
    control_only_training
        If True, mark only control wells for training while keeping all wells for inference.
    control_well_type
        Type of control wells to use for training when control_only_training=True.
    subset_rows
        If not None, subset the data to this many rows (for testing).
    subset_seed
        Random seed for subsetting.

    Returns
    -------
    AnnData object with phenomics features in X and metadata in obs.
    """
    logger.info(f"Reading phenomics data from {filepath}")

    # Read parquet file
    df = pd.read_parquet(filepath)

    # Subset if requested (for testing)
    if subset_rows is not None:
        if subset_seed is not None:
            df = df.sample(n=min(subset_rows, len(df)), random_state=subset_seed)
        else:
            df = df.head(subset_rows)
        logger.info(f"Subsetted data to {len(df)} rows")

    # Extract feature columns
    feature_cols = [col for col in df.columns if col.startswith(feature_prefix)]
    if not feature_cols:
        raise ValueError(f"No feature columns found with prefix '{feature_prefix}'")

    logger.info(f"Found {len(feature_cols)} feature columns")

    # Create X matrix (features)
    X = df[feature_cols].values.astype(np.float32)

    # Create obs dataframe (metadata)
    metadata_cols = [col for col in df.columns if not col.startswith(feature_prefix)]
    obs = df[metadata_cols].copy()

    # Create perturbation summary if requested
    if create_perturbation_summary and all(col in obs.columns for col in ["map_perturbation_type", "map_gene", "map_concentration"]):
        def create_summary(row):
            pert_type = str(row["map_perturbation_type"]).lower()
            if pert_type == "gene":
                return row['map_gene']
            elif pert_type == "compound":
                return f"{row['map_rec_id']}_{row['map_concentration']}"
            elif pert_type == "empty":
                return "CONTROL_EMPTY"
            else:
                return "unknown"

        obs["perturbation_summary"] = obs.apply(create_summary, axis=1)
        logger.info("Created perturbation_summary column")

    # Mark control wells for training if control_only_training is enabled
    if control_only_training:
        if "map_well_type" in obs.columns:
            obs["is_training_well"] = obs["map_well_type"] == control_well_type
            n_training_wells = obs["is_training_well"].sum()
            logger.info(f"Marked {n_training_wells} control wells ({control_well_type}) for training")
            if n_training_wells == 0:
                logger.warning(f"No wells found with type '{control_well_type}' for training")
        else:
            logger.warning("map_well_type column not found. Cannot mark control wells for training.")
            obs["is_training_well"] = True
    else:
        obs["is_training_well"] = True  # All wells used for training

    # Create var dataframe (feature names)
    var = pd.DataFrame(index=feature_cols)
    var.index.name = "feature_name"

    # Ensure batch column exists
    if batch_key not in obs.columns:
        raise ValueError(f"Batch key '{batch_key}' not found in data columns")

    # Create AnnData object
    adata = AnnData(X=X, obs=obs, var=var)

    # Store some useful information
    adata.uns["data_type"] = "phenomics"
    adata.uns["feature_type"] = "continuous"
    adata.uns["n_features"] = len(feature_cols)
    adata.uns["control_only_training"] = control_only_training
    adata.uns["control_well_type"] = control_well_type

    logger.info(f"Created AnnData object with shape {adata.shape}")

    return adata


def setup_phenomics_anndata(
    adata: AnnData,
    batch_key: str = "plate_number",
    categorical_covariate_keys: list[str] | None = None,
    continuous_covariate_keys: list[str] | None = None,
    condition_key: str | None = None,
) -> None:
    """Set up AnnData object for phenomics analysis with scVI.

    Parameters
    ----------
    adata
        AnnData object containing phenomics data.
    batch_key
        Key in adata.obs for batch information.
        Can use "hierarchical_batch" for nested batch effects.
    categorical_covariate_keys
        Keys in adata.obs for categorical covariates.
    continuous_covariate_keys
        Keys in adata.obs for continuous covariates.
    condition_key
        Key in adata.obs for condition information (for CVAE).
    """
    from scvi.model import PHVI

    # Ensure data is continuous
    if adata.uns.get("data_type") != "phenomics":
        logger.warning("Data type is not marked as 'phenomics'. Proceeding anyway.")

    # Setup the AnnData for use with PHVI
    PHVI.setup_anndata(
        adata,
        batch_key=batch_key,
        categorical_covariate_keys=categorical_covariate_keys,
        continuous_covariate_keys=continuous_covariate_keys,
        condition_key=condition_key,
    )
