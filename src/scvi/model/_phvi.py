"""Phenomics Variational Inference (PHVI) model."""

from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING

from scvi import REGISTRY_KEYS, settings
from scvi.data import AnnDataManager
from scvi.data.fields import (
    CategoricalJointObsField,
    CategoricalObsField,
    LayerField,
    NumericalJointObsField,
)
from scvi.model.base import EmbeddingMixin, UnsupervisedTrainingMixin
from scvi.module import PhenoVAE
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
    """Phenomics Variational Inference for continuous feature data.

    Parameters
    ----------
    adata
        AnnData object that has been registered via :meth:`~scvi.model.PHVI.setup_anndata`.
    n_hidden
        Number of nodes per hidden layer.
    n_latent
        Dimensionality of the latent space.
    n_layers
        Number of hidden layers used for encoder and decoder NNs.
    dropout_rate
        Dropout rate for neural networks.
    encode_covariates
        If ``True``, covariates are concatenated to features prior to encoding.
    deeply_inject_covariates
        If ``True`` and ``n_layers > 1``, covariates are concatenated to hidden layer outputs.
    batch_representation
        Method for encoding batch information. One of ``'one-hot'`` or ``'embedding'``.
    condition_representation
        Method for encoding condition information. One of ``'one-hot'`` or ``'embedding'``.
    use_batch_norm
        Specifies where to use batch normalization.
    use_layer_norm
        Specifies where to use layer normalization.
    output_var_param
        How to parameterize the output variance. One of:
        
        * ``'learned'``: learn a single variance parameter per feature
        * ``'fixed'``: use a fixed variance of 1.0
        * ``'feature'``: learn separate variance for each feature
        * ``'conditional'``: learn variance conditioned on perturbation (for CVAE)
    **kwargs
        Additional keyword arguments for :class:`~scvi.module.PhenoVAE`.

    Examples
    --------
    >>> from scvi.data import read_phenomics
    >>> adata = read_phenomics("path/to/phenomics.parquet")
    >>> scvi.model.PHVI.setup_anndata(adata, batch_key="hierarchical_batch", condition_key="condition")
    >>> vae = scvi.model.PHVI(adata)
    >>> vae.train()
    >>> adata.obsm["X_PHVI"] = vae.get_latent_representation()
    """

    _module_cls = PhenoVAE

    def __init__(
        self,
        adata: AnnData | None = None,
        registry: dict | None = None,
        n_hidden: int = 128,
        n_latent: int = 10,
        n_layers: int = 1,
        dropout_rate: float = 0.1,
        encode_covariates: bool = False,
        deeply_inject_covariates: bool = True,
        batch_representation: Literal["one-hot", "embedding"] = "one-hot",
        condition_representation: Literal["one-hot", "embedding"] = "embedding",
        use_batch_norm: Literal["encoder", "decoder", "none", "both"] = "both",
        use_layer_norm: Literal["encoder", "decoder", "none", "both"] = "none",
        output_var_param: Literal["learned", "fixed", "feature", "conditional"] = "learned",
        **kwargs,
    ):
        super().__init__(adata, registry)

        self._module_kwargs = {
            "n_hidden": n_hidden,
            "n_latent": n_latent,
            "n_layers": n_layers,
            "dropout_rate": dropout_rate,
            "encode_covariates": encode_covariates,
            "deeply_inject_covariates": deeply_inject_covariates,
            "batch_representation": batch_representation,
            "condition_representation": condition_representation,
            "use_batch_norm": use_batch_norm,
            "use_layer_norm": use_layer_norm,
            "output_var_param": output_var_param,
            **kwargs,
        }
        self._model_summary_string = (
            "PHVI model with the following parameters: \n"
            f"n_hidden: {n_hidden}, n_latent: {n_latent}, n_layers: {n_layers}, "
            f"dropout_rate: {dropout_rate}, output_var_param: {output_var_param}, "
            f"batch_representation: {batch_representation}."
        )

        if self._module_init_on_train:
            self.module = None
            warnings.warn(
                "Model was initialized without `adata`. The module will be initialized when "
                "calling `train`. This behavior is experimental and may change in the future.",
                UserWarning,
                stacklevel=settings.warnings_stacklevel,
            )
        else:
            if adata is not None:
                n_cats_per_cov = (
                    self.adata_manager.get_state_registry(
                        REGISTRY_KEYS.CAT_COVS_KEY
                    ).n_cats_per_key
                    if REGISTRY_KEYS.CAT_COVS_KEY in self.adata_manager.data_registry
                    else None
                )
                # Get number of conditions if available
                n_conditions = (
                    self.adata_manager.get_state_registry(
                        "condition"
                    ).n_cats
                    if "condition" in self.adata_manager.data_registry
                    else 0
                )
            else:
                # custom datamodule
                if (
                    len(
                        self.registry["field_registries"][f"{REGISTRY_KEYS.CAT_COVS_KEY}"][
                            "state_registry"
                        ]
                    )
                    > 0
                ):
                    n_cats_per_cov = tuple(
                        self.registry["field_registries"][f"{REGISTRY_KEYS.CAT_COVS_KEY}"][
                            "state_registry"
                        ]["n_cats_per_key"]
                    )
                else:
                    n_cats_per_cov = None
                n_conditions = 0

            n_batch = self.summary_stats.n_batch

            self.module = self._module_cls(
                n_input=self.summary_stats.n_vars,
                n_batch=n_batch,
                n_continuous_cov=self.summary_stats.get("n_extra_continuous_covs", 0),
                n_cats_per_cov=n_cats_per_cov,
                n_conditions=n_conditions,
                n_hidden=n_hidden,
                n_latent=n_latent,
                n_layers=n_layers,
                dropout_rate=dropout_rate,
                encode_covariates=encode_covariates,
                deeply_inject_covariates=deeply_inject_covariates,
                batch_representation=batch_representation,
                condition_representation=condition_representation,
                use_batch_norm=use_batch_norm,
                use_layer_norm=use_layer_norm,
                output_var_param=output_var_param,
                **kwargs,
            )

        self.init_params_ = self._get_init_params(locals())

    def get_normalized_expression(
        self,
        adata: AnnData | None = None,
        indices=None,
        n_samples: int = 1,
        batch_size: int | None = None,
    ) -> np.ndarray:
        """Get the denoised phenomics features.

        Parameters
        ----------
        adata
            AnnData object to use. If None, uses the registered AnnData.
        indices
            Indices of cells to use.
        n_samples
            Number of samples to use for posterior averaging.
        batch_size
            Batch size to use for computation.

        Returns
        -------
        Denoised phenomics features of shape (n_obs, n_features).
        """
        import numpy as np

        adata = self._validate_anndata(adata)

        data_loader = self._make_data_loader(
            adata=adata, indices=indices, batch_size=batch_size
        )

        # Get posterior samples and average
        denoised = []
        for tensors in data_loader:
            inference_outputs = self.module.inference(**self.module._get_inference_input(tensors))
            generative_outputs = self.module.generative(
                **self.module._get_generative_input(tensors, inference_outputs)
            )
            px = generative_outputs["px"]
            # Use mean of the distribution for denoised expression
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
        """Get control-anchored latent representation.

        This method computes the latent representation and then centers it
        by subtracting the mean of the control samples, making the origin
        represent the average control phenotype.

        Parameters
        ----------
        adata
            AnnData object to use. If None, uses the registered AnnData.
        indices
            Indices of cells to use.
        control_key
            Key in adata.obs that identifies control samples.
        control_value
            Value that identifies control samples.
        batch_size
            Batch size to use for computation.

        Returns
        -------
        Control-anchored latent representation of shape (n_obs, n_latent).
        """
        import numpy as np

        adata = self._validate_anndata(adata)

        # Get standard latent representation
        latent = self.get_latent_representation(
            adata=adata, indices=indices, batch_size=batch_size
        )

        # Find control samples
        if control_key not in adata.obs:
            logger.warning(f"Control key '{control_key}' not found in adata.obs. "
                          "Returning standard latent representation.")
            return latent

        # Get control mask
        if indices is not None:
            obs_subset = adata.obs.iloc[indices]
        else:
            obs_subset = adata.obs
        
        control_mask = obs_subset[control_key] == control_value
        
        if not control_mask.any():
            logger.warning(f"No control samples found with {control_key}={control_value}. "
                          "Returning standard latent representation.")
            return latent

        # Compute control centroid
        control_centroid = np.mean(latent[control_mask], axis=0, keepdims=True)

        # Center latent space by control centroid
        anchored_latent = latent - control_centroid

        return anchored_latent

    @classmethod
    @setup_anndata_dsp.dedent
    def setup_anndata(
        cls,
        adata: AnnData,
        layer: str | None = None,
        batch_key: str | None = None,
        labels_key: str | None = None,
        condition_key: str | None = None,
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
        condition_key
            Key in `adata.obs` for condition information used in conditional VAE.
        %(param_cat_cov_keys)s
        %(param_cont_cov_keys)s
        """
        setup_method_args = cls._get_setup_method_args(**locals())

        # Note: is_count_data=False for continuous phenomics features
        anndata_fields = [
            LayerField(REGISTRY_KEYS.X_KEY, layer, is_count_data=False),
            CategoricalObsField(REGISTRY_KEYS.BATCH_KEY, batch_key),
            CategoricalObsField(REGISTRY_KEYS.LABELS_KEY, labels_key),
            CategoricalJointObsField(REGISTRY_KEYS.CAT_COVS_KEY, categorical_covariate_keys),
            NumericalJointObsField(REGISTRY_KEYS.CONT_COVS_KEY, continuous_covariate_keys),
        ]
        
        # Add condition field if specified
        if condition_key is not None:
            anndata_fields.append(CategoricalObsField("condition", condition_key))

        adata_manager = AnnDataManager(fields=anndata_fields, setup_method_args=setup_method_args)
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)
