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
            **model_kwargs,
        }

        # Human-readable model summary (useful in logs and notebooks)
        self._model_summary_string = (
            "PHVI configuration:\n"
            f"  n_hidden={n_hidden}, n_latent={n_latent}, n_layers={n_layers}, dropout={dropout_rate}\n"
            f"  encode_covariates={encode_covariates}, deeply_inject_covariates={deeply_inject_covariates}\n"
            f"  batch_representation={batch_representation}, use_batch_norm={use_batch_norm}, use_layer_norm={use_layer_norm}\n"
            f"  recon_beta={recon_beta}, adv_batch_weight={adv_batch_weight}\n"
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
    def get_normalized_expression(
        self,
        adata: AnnData | None = None,
        indices=None,
        n_samples: int = 1,
        batch_size: int | None = None,
    ) -> np.ndarray:
        """Return denoised/decoded features (mean of Gaussian).

        Notes
        -----
        - This returns the decoder's mean (`px.loc`) which is a *denoised*
          reconstruction in feature space. Use the *latent `z`* for KNN/BMDB maps.
        """
        from scvi.module._constants import MODULE_KEYS

        ad = self._validate_anndata(adata)
        data_loader = self._make_data_loader(adata=ad, indices=indices, batch_size=batch_size)
        denoised = []
        for tensors in data_loader:
            inf_out = self.module.inference(**self.module._get_inference_input(tensors), n_samples=n_samples)
            gen_out = self.module.generative(**self.module._get_generative_input(tensors, inf_out))
            px = gen_out[MODULE_KEYS.PX_KEY]  # a torch.distributions.Normal
            denoised.append(px.loc.detach().cpu().numpy())
        return np.concatenate(denoised, axis=0)

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
