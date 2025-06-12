"""Phenomics Variational Inference (PHVI) model."""

from __future__ import annotations

import logging
import warnings
import numpy as np
from typing import TYPE_CHECKING

from scvi import REGISTRY_KEYS, settings
from scvi.data import AnnDataManager
from scvi.data.fields import (
    CategoricalJointObsField,
    CategoricalObsField,
    LayerField,
    NumericalJointObsField,
)
from scvi.dataloaders import DataSplitter
from scvi.model.base import EmbeddingMixin, UnsupervisedTrainingMixin
from scvi.module import PhenoVAE
from scvi.utils import setup_anndata_dsp

from .base import BaseModelClass, VAEMixin

if TYPE_CHECKING:
    from typing import Literal

    from anndata import AnnData

logger = logging.getLogger(__name__)

# ==============================================================================
# CORRECTED AND FINAL PHVIDataSplitter CLASS
# ==============================================================================
class PHVIDataSplitter(DataSplitter):
    """
    Custom DataSplitter for PHVI that respects control_only_training.
    This version correctly handles empty validation sets.
    """

    def __init__(self, adata_manager, **kwargs):
        # Pop our custom argument before passing to the parent __init__.
        # This is a cleaner pattern than re-implementing the whole __init__.
        self.control_only_training = kwargs.pop("control_only_training", False)
        super().__init__(adata_manager, **kwargs)

    def setup(self, stage=None):
        """
        Split indices in train/test/val sets.
        If control_only_training, the train set is ONLY the control wells.
        The val and test sets are split from the remaining non-control wells.
        """
        # First, let the parent class perform its default index splitting.
        super().setup(stage)

        # Now, if control_only_training is enabled, we override the splits.
        if self.control_only_training:
            adata = self.adata_manager.adata
            if "is_training_well" not in adata.obs.columns:
                logger.warning(
                    "'is_training_well' column not found. Disabling control_only_training logic."
                )
                return

            all_indices = np.arange(adata.n_obs)
            control_indices = all_indices[adata.obs["is_training_well"].to_numpy(dtype=bool)]
            
            if len(control_indices) == 0:
                raise ValueError("control_only_training is True but no control wells were found.")

            # The training set is now exclusively the control wells.
            self.train_idx = control_indices
            logger.info(f"PHVI: Overriding training data with {len(self.train_idx)} control wells.")

            # IMPORTANT: Since all wells in the training anndata are controls, the non-control
            # set is empty. Therefore, the validation and test sets must also be empty.
            self.val_idx = np.array([], dtype=int)
            self.test_idx = np.array([], dtype=int)
            logger.info("PHVI: Setting validation and test sets to empty for control-only training.")

    def val_dataloader(self):
        """
        Override to prevent crashing when the validation set is empty.
        """
        # THE CRITICAL FIX: If val_idx is empty, return a valid, empty dataloader.
        if len(self.val_idx) == 0:
            return self.data_loader_cls(
                self.adata_manager,
                indices=self.val_idx, # Pass the empty array
                batch_size=self.data_loader_kwargs.get("batch_size", 128)
            )
        # Otherwise, let the parent handle it.
        return super().val_dataloader()


class PHVI(
    EmbeddingMixin,
    VAEMixin,
    UnsupervisedTrainingMixin,
    BaseModelClass,
):
    """
    Phenomics Variational Inference (PHVI) model.

    This model is designed for phenomics data analysis, providing dimensionality
    reduction and batch correction capabilities for high-dimensional phenotypic measurements.

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
        Whether to concatenate covariates to expression in encoder.
    deeply_inject_covariates
        Whether to concatenate covariates into output of hidden layers in encoder/decoder.
    batch_representation
        One of the following:

        * ``"embedding"`` - Embedding representation for batch.
        * ``"one-hot"`` - One-hot representation for batch.
    condition_representation
        One of the following:

        * ``"embedding"`` - Embedding representation for condition.
        * ``"one-hot"`` - One-hot representation for condition.
    use_batch_norm
        Whether to use batch normalization in layers.
    use_layer_norm
        Whether to use layer normalization in layers.
    output_var_param
        Output variance parameter for the decoder.
    control_only_training
        If True, train only on control wells while using all wells for inference.
    **model_kwargs
        Keyword args for :class:`~scvi.module.PhenoVAE`

    Examples
    --------
    >>> adata = scvi.data.synthetic_iid()
    >>> scvi.model.PHVI.setup_anndata(adata, batch_key="batch")
    >>> vae = scvi.model.PHVI(adata)
    >>> vae.train()
    >>> adata.obsm["X_phvi"] = vae.get_latent_representation()
    """

    _module_cls = PhenoVAE
    _data_splitter_cls = PHVIDataSplitter

    def __init__(
        self,
        adata: AnnData,
        n_hidden: int = 128,
        n_latent: int = 10,
        n_layers: int = 1,
        dropout_rate: float = 0.1,
        encode_covariates: bool = True,
        deeply_inject_covariates: bool = True,
        batch_representation: Literal["embedding", "one-hot"] = "one-hot",
        condition_representation: Literal["embedding", "one-hot"] = "one-hot",
        use_batch_norm: bool = True,
        use_layer_norm: bool = False,
        output_var_param: Literal["learned", "fixed", "feature", "conditional"] = "learned",
        control_only_training: bool = False,
        **model_kwargs,
    ):
        super().__init__(adata)

        # Store control_only_training for use in data splitting
        self.control_only_training = control_only_training

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
            **model_kwargs,
        }
        self._model_summary_string = (
            "PHVI model with the following parameters: \n"
            f"n_hidden: {n_hidden}, n_latent: {n_latent}, n_layers: {n_layers}, "
            f"dropout_rate: {dropout_rate}, output_var_param: {output_var_param}, "
            f"batch_representation: {batch_representation}, "
            f"control_only_training: {control_only_training}."
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
                control_only_training=control_only_training,
                **model_kwargs,
            )

        self.init_params_ = self._get_init_params(locals())

    def train(
        self,
        max_epochs=None,
        accelerator="auto",
        devices="auto",
        train_size=None,
        validation_size=None,
        shuffle_set_split=True,
        load_sparse_tensor=False,
        batch_size=128,
        early_stopping=False,
        datasplitter_kwargs=None,
        plan_kwargs=None,
        datamodule=None,
        **trainer_kwargs,
    ):
        """Train the model.
        
        Parameters are the same as UnsupervisedTrainingMixin.train(), but we override
        to pass control_only_training to the data splitter.
        """
        if datamodule is None:
            datasplitter_kwargs = datasplitter_kwargs or {}
            # Pass control_only_training to the custom data splitter
            datasplitter_kwargs["control_only_training"] = self.control_only_training
        
        # Call parent train method
        return super().train(
            max_epochs=max_epochs,
            accelerator=accelerator,
            devices=devices,
            train_size=train_size,
            validation_size=validation_size,
            shuffle_set_split=shuffle_set_split,
            load_sparse_tensor=load_sparse_tensor,
            batch_size=batch_size,
            early_stopping=early_stopping,
            datasplitter_kwargs=datasplitter_kwargs,
            plan_kwargs=plan_kwargs,
            datamodule=datamodule,
            **trainer_kwargs,
        )

    def _make_data_loader(
        self,
        adata=None,
        indices=None,
        batch_size=None,
        shuffle=False,
        for_training=False,
        **data_loader_kwargs,
    ):
        """Override to filter training data to controls only when enabled."""
        if self.control_only_training and for_training and indices is None and adata is not None:
            # Filter to training wells (controls) during training ONLY
            if "is_training_well" in adata.obs.columns:
                control_indices = np.where(adata.obs["is_training_well"])[0]
                logger.info(f"Using {len(control_indices)} control wells for training")
                return super()._make_data_loader(
                    adata, control_indices, batch_size, shuffle, **data_loader_kwargs
                )
            else:
                logger.warning("is_training_well column not found. Using all wells for training.")
        
        return super()._make_data_loader(
            adata, indices, batch_size, shuffle, **data_loader_kwargs
        )

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
