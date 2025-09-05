"""Phenomics Variational Inference (PHVI) model.

This file defines a *thin, clean* wrapper around a Phenomics VAE module that
follows best practices for disentangling **non-linear batch effects** in
continuous phenomics embeddings (e.g., Phenom2):

- **Goal**: lower KNN (batch predictability) while increasing BMDB (biological
  recapitulation). TVN (linear) already handled most linear batch; this model
  targets **residual non-linear** effects without washing out true biology.

Principles implemented here
---------------------------
1) **Train on all wells**: We remove any control-only training behavior.
   The model needs biological variation to *learn what to preserve*; limiting
   to controls depresses BMDB.

2) **Batch goes to the DECODER, not the encoder**: The decoder is allowed to
   condition on batch to reconstruct batch-related nuisance. The encoder does
   *not* receive batch so the latent `z` does not need to carry it. This reduces
   batch leakage into `z` (↓KNN) without forcing over-smoothing (↑BMDB).

3) **No BatchNorm; optional LayerNorm in encoder**: BatchNorm interacts with
   mini-batch composition and can entangle batch signal. We default to no BN
   and use LayerNorm in the encoder for stability on heterogeneous features.

4) **Gaussian likelihood**: Phenomic features are continuous and the model
   uses a diagonal Gaussian likelihood for reconstruction.

5) **Keep it simple, iterate quickly**: We drop unstable extras (adversarial
   training without GRL/two-step optimization, corr-loss, bio-contrastive).
   You can add a proper GRL+disc later if needed.

Typical usage
-------------
>>> # 1) Prepare AnnData with _phenomics.read_phenomics(...)
>>> # 2) Register with PHVI:
>>> PHVI.setup_anndata(adata, batch_key="experiment_plate_number")  # batch_key required
>>> # 3) Train PHVI
>>> model = PHVI(adata)                 # uses sane defaults
>>> model.train(max_epochs=200, batch_size=256)
>>> # 4) Get latent for benchmarking
>>> adata.obsm["X_phvi"] = model.get_latent_representation()

Notes
-----
- This class intentionally keeps the public API similar to scvi-tools models.
- The actual encoder/decoder wiring (batch to decoder only) must be respected
  inside the module (e.g., `PhenoVAE.generative` should condition on batch).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from scvi import REGISTRY_KEYS
from scvi.data import AnnDataManager
from scvi.data.fields import (
    CategoricalJointObsField,
    CategoricalObsField,
    LayerField,
    NumericalJointObsField,
)
from scvi.model.base import EmbeddingMixin, UnsupervisedTrainingMixin
from scvi.module._phenovae import PhenoVAE
from scvi.utils import setup_anndata_dsp

from .base import BaseModelClass, VAEMixin

if TYPE_CHECKING:
    from typing import Literal
    from anndata import AnnData

logger = logging.getLogger(__name__)


class PHVI(
    EmbeddingMixin,
    VAEMixin,
    UnsupervisedTrainingMixin,
    BaseModelClass,
):
    """
    Phenomics Variational Inference (PHVI) model.

    A lightweight scvi-tools model wrapper that pairs an AnnData manager with a
    phenomics-appropriate VAE module (``PhenoVAE``). The module is expected to:
      - Use a Gaussian likelihood over continuous features
      - Diagonal gaussian output variance
      - Condition the **decoder** on batch (one-hot or embedding)
      - Keep the **encoder** free of batch inputs

    Parameters
    ----------
    adata
        AnnData registered via :meth:`~scvi.model.PHVI.setup_anndata`.
    n_hidden
        Width of the MLP hidden layers.
    n_latent
        Dimensionality of the latent space `z`. Start with 128; increase if BMDB
        plateaus *and* KNN remains low.
    n_layers
        Depth of the encoder/decoder MLPs.
    dropout_rate
        Dropout rate used in the MLPs (both encoder and decoder).
    encode_covariates
        Whether to concatenate covariates into the **encoder** input.
        For batch, this should be **False** to avoid leaking batch into `z`.
    deeply_inject_covariates
        Whether to inject covariates at hidden layers (FiLM-style) in encoder/decoder.
        Keep **False** initially for simplicity; decoder already conditions on batch.
    batch_representation
        "one-hot" or "embedding" representation for batch. One-hot is simple and robust.
    use_batch_norm
        Where to apply BatchNorm. Default "none" (see rationale above).
    use_layer_norm
        Where to apply LayerNorm. Default "encoder" (stabilizes encoder).
    recon_beta
        Weight on reconstruction loss. Keep β=1.0 and use KL annealing instead of
        detuning β (avoids over-compression harming BMDB).
    adv_batch_weight
        Weight for adversarial batch prediction loss on latent z. When > 0,
        adds a gradient reversal layer that penalizes batch information in z.
    adv_lambda
        Gradient reversal scaling factor for adversarial training. Controls the strength
        of gradient reversal during backpropagation. Default 1.0.
    use_nll
        Whether to use full Gaussian negative log-likelihood loss. If False (default), uses
        a weighted MSE loss that still incorporates learned variance but is more 
        numerically stable for initial training. Both losses use the learned variance.
    **model_kwargs
        Passed through to the underlying :class:`~scvi.module.PhenoVAE`.

    Examples
    --------
    >>> PHVI.setup_anndata(adata, batch_key="experiment_plate_number")  # batch_key required
    >>> model = PHVI(adata, n_latent=128, n_hidden=256, n_layers=2)
    >>> model.train(max_epochs=200, batch_size=256)
    >>> adata.obsm["X_phvi"] = model.get_latent_representation()
    """

    # Attach the module class; this should implement the decoder-conditioned-on-batch design.
    _module_cls = PhenoVAE

    def __init__(
        self,
        adata: AnnData,
        *,
        # Recommended starting hyperparameters for phenomics:
        n_hidden: int = 256,
        n_latent: int = 128,   # start at 128; consider 160–256 only if BMDB plateaus
        n_layers: int = 2,
        dropout_rate: float = 0.10,
        # Critical flags for disentanglement:
        encode_covariates: bool = False,  # batch **must not** go to encoder
        deeply_inject_covariates: bool = False,
        batch_representation: Literal["embedding", "one-hot"] = "embedding",
        use_batch_norm: Literal["encoder", "decoder", "none", "both"] = "none",
        use_layer_norm: Literal["encoder", "decoder", "none", "both"] = "encoder",
        # Likelihood/regularization:
        recon_beta: float = 1.0,
        # Batch effect mitigation:
        adv_batch_weight: float = 0.0,
        adv_lambda: float = 1.0,
        # Loss function:
        use_nll: bool = False,
        # Any additional kwargs passed straight into the module
        **model_kwargs,
    ):
        # Initialize BaseModelClass/Embedding/VAEMixins plumbing
        super().__init__(adata)

        # Build the argument dictionary handed to the underlying module. These
        # flags must be honored by the module implementation (PhenoVAE).
        self._module_kwargs = {
            "n_hidden": n_hidden,
            "n_latent": n_latent,
            "n_layers": n_layers,
            "dropout_rate": dropout_rate,
            "encode_covariates": encode_covariates,            # encoder stays batch-free
            "deeply_inject_covariates": deeply_inject_covariates,
            "batch_representation": batch_representation,      # decoder will consume this
            "use_batch_norm": use_batch_norm,                  # recommended: "none"
            "use_layer_norm": use_layer_norm,                  # recommended: "encoder"
            "recon_beta": recon_beta,                          # keep β=1.0; use KL anneal in training plan
            "adv_batch_weight": adv_batch_weight,
            "adv_lambda": adv_lambda,
            "use_nll": use_nll,
            **model_kwargs,
        }

        # Human-readable model summary (useful in logs and notebooks)
        self._model_summary_string = (
            "PHVI configuration:\n"
            f"  n_hidden={n_hidden}, n_latent={n_latent}, n_layers={n_layers}, dropout={dropout_rate}\n"
            f"  encode_covariates={encode_covariates}, deeply_inject_covariates={deeply_inject_covariates}\n"
            f"  batch_representation={batch_representation}, use_batch_norm={use_batch_norm}, use_layer_norm={use_layer_norm}\n"
            f"  recon_beta={recon_beta}, adv_batch_weight={adv_batch_weight}, adv_lambda={adv_lambda}\n"
            f"  use_nll={use_nll} (loss={'NLL' if use_nll else 'MSE'})\n"
        )

        # Discover covariates from the AnnData manager
        if REGISTRY_KEYS.CAT_COVS_KEY in self.adata_manager.data_registry:
            cat_cov_registry = self.adata_manager.get_state_registry(REGISTRY_KEYS.CAT_COVS_KEY)
            n_cats_per_cov = cat_cov_registry.get("n_cats_per_key", None)
        else:
            n_cats_per_cov = None

        n_batch = self.summary_stats.n_batch

        # Initialize the module with counts of inputs/variables; the module
        # must implement the "batch to decoder" pattern internally.
        self.module = self._module_cls(
            n_input=self.summary_stats.n_vars,
            n_batch=n_batch,
            n_continuous_cov=self.summary_stats.get("n_extra_continuous_covs", 0),
            n_cats_per_cov=n_cats_per_cov,
            **self._module_kwargs,
        )

        # Stash init params for serialization/reproducibility
        self.init_params_ = self._get_init_params(locals())

    # --------------------------------------------------------------------- #
    # OPTIONAL utilities
    # --------------------------------------------------------------------- #
    def get_reconstruction(
        self,
        adata: AnnData | None = None,
        indices=None,
        n_samples: int = 1,
        batch_size: int | None = None,
    ) -> np.ndarray:
        """Return reconstructed features using original batch information.

        This returns the decoder's reconstruction of the input features, conditioned
        on the original batch of each sample. The reconstruction will still contain
        batch effects.

        Parameters
        ----------
        adata
            AnnData object with batch info. If None, uses the adata used to train model.
        indices
            Indices of cells to use. If None, uses all cells.
        n_samples
            Number of posterior samples to use for reconstruction.
        batch_size
            Minibatch size for data loading.

        Returns
        -------
        Reconstructed feature matrix with same shape as input features.
        """
        from scvi.module._constants import MODULE_KEYS

        ad = self._validate_anndata(adata)
        data_loader = self._make_data_loader(adata=ad, indices=indices, batch_size=batch_size)
        reconstructed = []
        for tensors in data_loader:
            inf_out = self.module.inference(**self.module._get_inference_input(tensors), n_samples=n_samples)
            gen_out = self.module.generative(**self.module._get_generative_input(tensors, inf_out))
            px = gen_out[MODULE_KEYS.PX_KEY]  # a torch.distributions.Normal
            reconstructed.append(px.loc.detach().cpu().numpy())
        return np.concatenate(reconstructed, axis=0)

    def get_normalized_expression(
        self,
        adata: AnnData | None = None,
        indices=None,
        n_samples: int = 1,
        batch_size: int | None = None,
        reference_batch: int | None = None,
    ) -> np.ndarray:
        """Return batch-corrected features by decoding with a reference batch.

        This method encodes the data to get batch-free latent representations,
        then decodes using a reference batch instead of the original batches.
        This produces features that are corrected for batch effects.

        Parameters
        ----------
        adata
            AnnData object with batch info. If None, uses the adata used to train model.
        indices
            Indices of cells to use. If None, uses all cells.
        n_samples
            Number of posterior samples to use for reconstruction.
        batch_size
            Minibatch size for data loading.
        reference_batch
            Batch index to use for decoding. If None, uses the most common batch.

        Returns
        -------
        Batch-corrected feature matrix with same shape as input features.
        """
        from scvi.module._constants import MODULE_KEYS
        import torch

        ad = self._validate_anndata(adata)
        
        # Find reference batch if not provided
        if reference_batch is None:
            # Get batch information from the adata manager
            batch_key = self.adata_manager.get_state_registry(REGISTRY_KEYS.BATCH_KEY).get("original_key", None)
            if batch_key is None:
                logger.warning("No batch key found in registry, using batch 0 as reference")
                reference_batch = 0
            else:
                # Get the actual batch values from the subset of data we're using
                if indices is not None:
                    batch_values = ad.obs[batch_key].iloc[indices]
                else:
                    batch_values = ad.obs[batch_key]
                
                # Find most common batch
                most_common_batch_label = batch_values.value_counts().idxmax()
                # Convert to numeric index using the categorical mapping
                batch_mapping = self.adata_manager.get_state_registry(REGISTRY_KEYS.BATCH_KEY).get("categorical_mapping", None)
                if batch_mapping is not None:
                    # Handle numpy array or list
                    if hasattr(batch_mapping, 'tolist'):
                        batch_mapping = batch_mapping.tolist()
                    reference_batch = batch_mapping.index(most_common_batch_label)
                else:
                    logger.warning("No categorical mapping found, using batch 0 as reference")
                    reference_batch = 0
                
                logger.info(f"Using most common batch '{most_common_batch_label}' (index {reference_batch}) as reference")
        
        # Process data
        data_loader = self._make_data_loader(adata=ad, indices=indices, batch_size=batch_size)
        batch_corrected = []
        
        for tensors in data_loader:
            # Get latent representation (batch-free)
            inf_out = self.module.inference(**self.module._get_inference_input(tensors), n_samples=n_samples)
            
            # Create reference batch tensor
            batch_tensor = tensors[REGISTRY_KEYS.BATCH_KEY]
            reference_batch_tensor = torch.full_like(batch_tensor, reference_batch)
            
            # Prepare generative inputs with reference batch
            gen_inputs = {
                MODULE_KEYS.Z_KEY: inf_out[MODULE_KEYS.Z_KEY],
                MODULE_KEYS.BATCH_INDEX_KEY: reference_batch_tensor,
                MODULE_KEYS.CONT_COVS_KEY: tensors.get(REGISTRY_KEYS.CONT_COVS_KEY, None),
                MODULE_KEYS.CAT_COVS_KEY: tensors.get(REGISTRY_KEYS.CAT_COVS_KEY, None),
            }
            
            # Decode with reference batch
            gen_out = self.module.generative(**gen_inputs)
            px = gen_out[MODULE_KEYS.PX_KEY]  # a torch.distributions.Normal
            batch_corrected.append(px.loc.detach().cpu().numpy())
        
        return np.concatenate(batch_corrected, axis=0)

    def get_control_anchored_latent_representation(
        self,
        adata: AnnData | None = None,
        indices=None,
        control_key: str = "is_control",
        control_value: bool = True,
        batch_size: int | None = None,
    ) -> np.ndarray:
        """Return latent `z` after subtracting the control centroid (optional viz trick).

        This is a light post-hoc centering that can help visualization or shave
        a bit of residual batch when controls are well-distributed across batches.
        **Do not** anchor during training—train on *all* wells.
        """
        ad = self._validate_anndata(adata)
        z = self.get_latent_representation(adata=ad, indices=indices, batch_size=batch_size)

        if control_key not in ad.obs:
            logger.warning(f"[get_control_anchored_latent_representation] '{control_key}' not in adata.obs; returning raw latent.")
            return z

        obs_subset = ad.obs.iloc[indices] if indices is not None else ad.obs
        mask = (obs_subset[control_key] == control_value).to_numpy()
        if not np.any(mask):
            logger.warning(f"[get_control_anchored_latent_representation] No rows with {control_key}={control_value}; returning raw latent.")
            return z

        control_centroid = z[mask].mean(axis=0, keepdims=True)
        return z - control_centroid

    # --------------------------------------------------------------------- #
    # AnnData registration
    # --------------------------------------------------------------------- #
    @classmethod
    @setup_anndata_dsp.dedent
    def setup_anndata(
        cls,
        adata: AnnData,
        layer: str | None = None,
        batch_key: str | None = None,
        labels_key: str | None = None,
        categorical_covariate_keys: list[str] | None = None,
        continuous_covariate_keys: list[str] | None = None,
        **kwargs,
    ):
        """%(summary)s.

        Parameters
        ----------
        %(param_adata)s
        %(param_layer)s
        %(param_batch_key)s
        %(param_labels_key)s
        %(param_cat_cov_keys)s
        %(param_cont_cov_keys)s

        Notes
        -----
        - We register X as **non-count** data (continuous phenomics embeddings).
        - **Interface gotcha (important)**: `batch_key` is required. The module 
          unconditionally expects batch indices. Users must either provide an explicit 
          `batch_key` or use hierarchical batch creation in `read_phenomics_from_df()`.
        """
        setup_method_args = cls._get_setup_method_args(**locals())

        # Register fields: X is continuous (non-count), batch is categorical.
        anndata_fields = [
            LayerField(REGISTRY_KEYS.X_KEY, layer, is_count_data=False),
            CategoricalObsField(REGISTRY_KEYS.BATCH_KEY, batch_key),
            CategoricalObsField(REGISTRY_KEYS.LABELS_KEY, labels_key),
            CategoricalJointObsField(REGISTRY_KEYS.CAT_COVS_KEY, categorical_covariate_keys),
            NumericalJointObsField(REGISTRY_KEYS.CONT_COVS_KEY, continuous_covariate_keys),
        ]

        adata_manager = AnnDataManager(fields=anndata_fields, setup_method_args=setup_method_args)
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)
