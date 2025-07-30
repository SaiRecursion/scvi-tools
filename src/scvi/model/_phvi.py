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
    This version correctly handles empty validation sets when control_only_training is True,
    and defaults to standard behavior when it's False.
    """

    def __init__(self, adata_manager, **kwargs):
        # Pop our custom argument before passing to the parent __init__.
        self.control_only_training = kwargs.pop("control_only_training", False)
        super().__init__(adata_manager, **kwargs)

    def setup(self, stage=None):
        """
        Split indices in train/test/val sets.
        If control_only_training is True, the train set is ONLY the control wells.
        The val and test sets are empty.
        If control_only_training is False, it uses the parent's default behavior.
        """
        # Let the parent class perform its default index splitting first.
        # This will correctly handle train_size/validation_size for the False case.
        super().setup(stage)

        # Now, IF AND ONLY IF control_only_training is enabled, we override the splits.
        if self.control_only_training:
            adata = self.adata_manager.adata
            if "is_training_well" not in adata.obs.columns:
                logger.warning(
                    "'is_training_well' column not found. Disabling control_only_training logic."
                )
                # Fallback to standard behavior if the column is missing
                self.control_only_training = False 
                return

            all_indices = np.arange(adata.n_obs)
            # Use the 'is_training_well' column created in the notebook
            control_indices = all_indices[adata.obs["is_training_well"].to_numpy(dtype=bool)]
            
            if len(control_indices) == 0:
                raise ValueError("control_only_training is True but no control wells were found.")

            # The training set is now exclusively the control wells.
            self.train_idx = control_indices
            logger.info(f"PHVI: Overriding training data with {len(self.train_idx)} control wells.")

            # The validation and test sets MUST be empty for this mode to avoid errors.
            self.val_idx = np.array([], dtype=int)
            self.test_idx = np.array([], dtype=int)
            logger.info("PHVI: Setting validation and test sets to empty for control-only training.")

    def val_dataloader(self):
        """
        Override to prevent crashing when the validation set is empty, which is the case
        when control_only_training is True.
        """
        # If the validation index array is empty, return a valid, empty dataloader.
        if len(self.val_idx) == 0:
            return self.data_loader_cls(
                self.adata_manager,
                indices=self.val_idx, # Pass the empty array
                batch_size=self.data_loader_kwargs.get("batch_size", 128)
            )
        # Otherwise (when control_only_training=False), let the parent handle it.
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
        Whether to use batch normalization in layers. One of "encoder", "decoder", "none", "both".
    use_layer_norm
        Whether to use layer normalization in layers. One of "encoder", "decoder", "none", "both".
    output_var_param
        Output variance parameter for the decoder.
    control_only_training
        If True, train only on control wells while using all wells for inference.
        This requires an `is_training_well` column in `adata.obs`.
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
        n_latent: int = 128,
        n_layers: int = 1,
        dropout_rate: float = 0.1,
        encode_covariates: bool = False,
        deeply_inject_covariates: bool = False,
        batch_representation: Literal["embedding", "one-hot"] = "one-hot",
        condition_representation: Literal["embedding", "one-hot"] = "one-hot",
        use_batch_norm: Literal["encoder", "decoder", "none", "both"] = "both",
        use_layer_norm: Literal["encoder", "decoder", "none", "both"] = "none",
        output_var_param: Literal["learned", "fixed", "feature", "conditional"] = "conditional",
        var_reg_lambda: float = 0.001,
        recon_beta: float = 1.0,
        control_only_training: bool = True,
        **model_kwargs,
    ):
        super().__init__(adata)

        # Store control_only_training for use in data splitting and module init
        self.control_only_training = control_only_training

        # Pass all relevant parameters to the underlying module
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
            "use_layer_norm": use_layer_norm, # Make sure this is passed
            "output_var_param": output_var_param,
            "var_reg_lambda": var_reg_lambda,
            "recon_beta": recon_beta,
            "control_only_training": control_only_training,
            **model_kwargs,
        }
        self._model_summary_string = (
            "PHVI model with the following parameters: \n"
            f"n_hidden: {n_hidden}, n_latent: {n_latent}, n_layers: {n_layers}, "
            f"dropout_rate: {dropout_rate}, output_var_param: {output_var_param}, "
            f"batch_representation: {batch_representation}, "
            f"use_layer_norm: {use_layer_norm}, " # Add to summary for clarity
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
                if REGISTRY_KEYS.CAT_COVS_KEY in self.adata_manager.data_registry:
                    cat_cov_registry = self.adata_manager.get_state_registry(REGISTRY_KEYS.CAT_COVS_KEY)
                    n_cats_per_cov = cat_cov_registry.get("n_cats_per_key", None)
                else:
                    n_cats_per_cov = None
                if "condition" in self.adata_manager.data_registry:
                    condition_registry = self.adata_manager.get_state_registry("condition")
                    # Debug: print available keys
                    print("Condition registry keys:", list(condition_registry.keys()))
                    n_conditions = len(condition_registry.get("categorical_mapping", []))
                    print(f"Number of conditions found: {n_conditions}")
                else:
                    n_conditions = 0
            else:
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
                **self._module_kwargs,
            )

        self.init_params_ = self._get_init_params(locals())

    def train(
        self,
        max_epochs=None,
        accelerator="auto",
        devices="auto",
        train_size=0.9,
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
        # Debug: Print what we're receiving
        print(f"DEBUG: plan_kwargs = {plan_kwargs}")
        print(f"DEBUG: trainer_kwargs keys = {list(trainer_kwargs.keys()) if trainer_kwargs else 'None'}")
        
        # Filter out any optimizer-related params that might have leaked into trainer_kwargs
        if trainer_kwargs:
            optimizer_params = ['lr', 'weight_decay', 'kl_weight', 'kl_anneal_epochs']
            filtered_trainer_kwargs = {k: v for k, v in trainer_kwargs.items() 
                                     if k not in optimizer_params}
            print(f"DEBUG: filtered_trainer_kwargs keys = {list(filtered_trainer_kwargs.keys())}")
        else:
            filtered_trainer_kwargs = {}
        
        if datamodule is None:
            datasplitter_kwargs = datasplitter_kwargs or {}
            # Pass control_only_training to our custom data splitter
            datasplitter_kwargs["control_only_training"] = self.control_only_training
        
        # Call parent train method with the updated kwargs
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
            **filtered_trainer_kwargs,
        )

    def get_normalized_expression(
        self,
        adata: AnnData | None = None,
        indices=None,
        n_samples: int = 1,
        batch_size: int | None = None,
    ) -> np.ndarray:
        """Get the denoised phenomics features."""
        import numpy as np
        adata = self._validate_anndata(adata)
        dl_kwargs = {}
        data_loader = self._make_data_loader(
            adata=adata, indices=indices, batch_size=batch_size, **dl_kwargs
        )
        denoised = []
        for tensors in data_loader:
            inference_outputs = self.module.inference(**self.module._get_inference_input(tensors))
            generative_outputs = self.module.generative(
                **self.module._get_generative_input(tensors, inference_outputs)
            )
            px = generative_outputs["px"]
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
        """Get control-anchored latent representation."""
        import numpy as np
        adata = self._validate_anndata(adata)
        latent = self.get_latent_representation(
            adata=adata, indices=indices, batch_size=batch_size
        )
        if control_key not in adata.obs:
            logger.warning(f"Control key '{control_key}' not found in adata.obs. "
                          "Returning standard latent representation.")
            return latent
        if indices is not None:
            obs_subset = adata.obs.iloc[indices]
        else:
            obs_subset = adata.obs
        control_mask = obs_subset[control_key] == control_value
        if not control_mask.any():
            logger.warning(f"No control samples found with {control_key}={control_value}. "
                          "Returning standard latent representation.")
            return latent
        control_centroid = np.mean(latent[control_mask], axis=0, keepdims=True)
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
        hierarchical_batch_keys: list[str] | None = None,
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
        hierarchical_batch_keys
            A list of two keys in `adata.obs` to be combined to create a hierarchical
            batch key. If provided, this will override the `batch_key` argument.
            Example: `['experiment', 'plate_number']`.
        """
        if hierarchical_batch_keys is not None:
            if len(hierarchical_batch_keys) != 2:
                raise ValueError("`hierarchical_batch_keys` must be a list of exactly two keys.")
            key1, key2 = hierarchical_batch_keys
            if key1 not in adata.obs.columns or key2 not in adata.obs.columns:
                raise ValueError(f"One or both keys '{key1}', '{key2}' not found in adata.obs.")
            
            new_batch_col_name = f"{key1}_{key2}"
            logger.info(
                f"Creating hierarchical batch key '{new_batch_col_name}' from '{key1}' and '{key2}'."
            )
            adata.obs[new_batch_col_name] = (
                adata.obs[key1].astype(str) + "_" + adata.obs[key2].astype(str)
            )
            batch_key = new_batch_col_name
            logger.info(f"Using '{batch_key}' as the final batch key.")
            
        setup_method_args = cls._get_setup_method_args(**locals())
    
        anndata_fields = [
            LayerField(REGISTRY_KEYS.X_KEY, layer, is_count_data=False),
            CategoricalObsField(REGISTRY_KEYS.BATCH_KEY, batch_key),
            CategoricalObsField(REGISTRY_KEYS.LABELS_KEY, labels_key),
            CategoricalJointObsField(REGISTRY_KEYS.CAT_COVS_KEY, categorical_covariate_keys),
            NumericalJointObsField(REGISTRY_KEYS.CONT_COVS_KEY, continuous_covariate_keys),
        ]
        
        if condition_key is not None:
            anndata_fields.append(CategoricalObsField("condition", condition_key))
    
        adata_manager = AnnDataManager(fields=anndata_fields, setup_method_args=setup_method_args)
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)
