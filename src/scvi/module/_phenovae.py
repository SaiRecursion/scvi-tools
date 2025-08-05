"""Phenomics VAE module with Gaussian likelihood for continuous features."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
from torch.distributions import Normal
from torch import nn

from scvi import REGISTRY_KEYS
from scvi.module._constants import MODULE_KEYS
from scvi.module.base import BaseModuleClass, EmbeddingModuleMixin, LossOutput, auto_move_data

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Literal

    from torch.distributions import Distribution

logger = logging.getLogger(__name__)


class ResidualFCLayers(nn.Module):
    """Custom FCLayers with residual connections."""
    def __init__(
        self,
        n_in: int,
        n_out: int,
        n_cat_list: list[int] | None = None,
        n_layers: int = 1,
        n_hidden: int = 128,
        dropout_rate: float = 0.1,
        inject_covariates: bool = True,
        use_batch_norm: bool = False,
        use_layer_norm: bool = False,
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            in_dim = n_in if i == 0 else n_hidden
            out_dim = n_hidden if i < n_layers - 1 else n_out
            layer = nn.Linear(in_dim, out_dim)
            self.layers.append(layer)
            if use_batch_norm:
                self.layers.append(nn.BatchNorm1d(out_dim))
            if use_layer_norm:
                self.layers.append(nn.LayerNorm(out_dim))
            if dropout_rate > 0 and i < n_layers - 1:
                self.layers.append(nn.Dropout(dropout_rate))
            self.layers.append(nn.ReLU())
            if in_dim == out_dim and i < n_layers - 1:
                self.layers.append(nn.Identity())
        self.inject_covariates = inject_covariates
        self.n_cat_list = n_cat_list or []

    def forward(self, x: torch.Tensor, *cat_list: torch.Tensor):
        h = x
        for i, layer in enumerate(self.layers):
            if isinstance(layer, nn.Linear):
                h_prev = h
                h = layer(h)
                if h_prev.shape == h.shape:
                    h = h + h_prev
            else:
                h = layer(h)
            if self.inject_covariates and cat_list and i < len(self.n_cat_list):
                cat = cat_list[i]
                if cat is not None:
                    h = h + torch.nn.functional.one_hot(cat.long(), self.n_cat_list[i]).float()
        return h


class PhenoVAE(EmbeddingModuleMixin, BaseModuleClass):
    """Variational auto-encoder for phenomics data with Gaussian likelihood.

    Parameters
    ----------
    n_input
        Number of input features (phenomic features).
    n_batch
        Number of batches. If ``0``, no batch correction is performed.
    n_hidden
        Number of nodes per hidden layer.
    n_latent
        Dimensionality of the latent space.
    n_layers
        Number of hidden layers.
    n_continuous_cov
        Number of continuous covariates.
    n_cats_per_cov
        A list of integers containing the number of categories for each categorical covariate.
    n_conditions
        Number of unique conditions for CVAE. If ``0``, standard VAE is used.
    dropout_rate
        Dropout rate.
    encode_covariates
        If ``True``, covariates are concatenated to features prior to encoding.
    deeply_inject_covariates
        If ``True`` and ``n_layers > 1``, covariates are concatenated to hidden layer outputs.
    batch_representation
        Method for encoding batch information. One of ``"one-hot"`` or ``"embedding"``.
    condition_representation
        Method for encoding condition information. One of ``"one-hot"`` or ``"embedding"``.
    use_batch_norm
        Specifies where to use :class:`~torch.nn.BatchNorm1d` in the model.
    use_layer_norm
        Specifies where to use :class:`~torch.nn.LayerNorm` in the model.
    var_activation
        Callable used to ensure positivity of the variance of the variational distribution.
    output_var_param
        How to parameterize the output variance. One of:
        * ``"learned"``: learn a single variance parameter per feature
        * ``"fixed"``: use a fixed variance of 1.0
        * ``"feature"``: learn separate variance for each feature
    control_only_training
        If ``True``, the model is only used for control and not for training.
    use_corr_loss
        If ``True``, adds adversarial correlation loss for batch disentanglement.
    """

    def __init__(
        self,
        n_input: int,
        n_batch: int = 0,
        n_hidden: int = 128,
        n_latent: int = 128,
        n_layers: int = 1,
        n_continuous_cov: int = 0,
        n_cats_per_cov: list[int] | None = None,
        n_conditions: int = 0,
        dropout_rate: float = 0.1,
        encode_covariates: bool = False,
        deeply_inject_covariates: bool = False,
        batch_representation: Literal["one-hot", "embedding"] = "one-hot",
        condition_representation: Literal["one-hot", "embedding"] = "embedding",
        use_batch_norm: Literal["encoder", "decoder", "none", "both"] = "both",
        use_layer_norm: Literal["encoder", "decoder", "none", "both"] = "none",
        var_activation: Callable[[torch.Tensor], torch.Tensor] = None,
        output_var_param: Literal["learned", "fixed", "feature"] = "feature",
        var_reg_lambda: float = 0.01,
        recon_beta: float = 1.0,
        batch_embedding_kwargs: dict | None = None,
        condition_embedding_kwargs: dict | None = None,
        control_only_training: bool = False,
        use_corr_loss: bool = False,
    ):
        super().__init__()

        self.n_latent = n_latent
        self.n_batch = n_batch
        self.n_input = n_input
        self.n_conditions = n_conditions
        self.encode_covariates = encode_covariates
        self.output_var_param = output_var_param
        self.var_reg_lambda = var_reg_lambda
        self.recon_beta = recon_beta
        self.is_cvae = n_conditions > 0
        self.control_only_training = control_only_training
        self.use_corr_loss = use_corr_loss

        # Initialize output variance parameters
        if self.output_var_param == "learned":
            self.px_log_var = torch.nn.Parameter(torch.full((1,), -2.3))
        elif self.output_var_param == "feature":
            self.px_log_var = torch.nn.Parameter(torch.zeros((n_input,)))
        elif self.output_var_param == "fixed":
            self.px_log_var = None
        else:
            raise ValueError("output_var_param must be 'learned', 'fixed', or 'feature'")

        self.batch_representation = batch_representation
        if self.batch_representation == "embedding":
            self.init_embedding(REGISTRY_KEYS.BATCH_KEY, n_batch, **(batch_embedding_kwargs or {}))
            batch_dim = self.get_embedding(REGISTRY_KEYS.BATCH_KEY).embedding_dim
        else:
            batch_dim = n_batch

        self.condition_representation = condition_representation
        condition_dim = 0
        if self.is_cvae:
            if self.condition_representation == "embedding":
                self.init_embedding("condition", n_conditions, **(condition_embedding_kwargs or {}))
                condition_dim = self.get_embedding("condition").embedding_dim
            else:
                condition_dim = n_conditions

        use_batch_norm_encoder = use_batch_norm in ["encoder", "both"]
        use_batch_norm_decoder = use_batch_norm in ["decoder", "both"]
        use_layer_norm_encoder = use_layer_norm in ["encoder", "both"]
        use_layer_norm_decoder = use_layer_norm in ["decoder", "both"]

        n_input_encoder = n_input + n_continuous_cov * encode_covariates
        n_input_encoder += batch_dim * encode_covariates
        cat_list = list([] if n_cats_per_cov is None else n_cats_per_cov)
        if self.batch_representation != "embedding":
            cat_list = [n_batch] + cat_list
        if self.is_cvae and self.condition_representation == "one-hot":
            n_input_encoder += condition_dim
            cat_list = [n_conditions] + cat_list
        if self.is_cvae and self.condition_representation == "embedding" and self.encode_covariates:
            n_input_encoder += condition_dim

        encoder_cat_list = cat_list if encode_covariates else None

        self.z_encoder = ResidualFCLayers(
            n_in=n_input_encoder,
            n_out=n_latent * 2,
            n_cat_list=encoder_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            inject_covariates=deeply_inject_covariates,
            use_batch_norm=use_batch_norm_encoder,
            use_layer_norm=use_layer_norm_encoder,
        )

        n_input_decoder = n_latent + n_continuous_cov + batch_dim
        if self.is_cvae:
            n_input_decoder += condition_dim

        self.decoder = ResidualFCLayers(
            n_in=n_input_decoder,
            n_out=n_input,
            n_cat_list=cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            inject_covariates=deeply_inject_covariates,
            use_batch_norm=use_batch_norm_decoder,
            use_layer_norm=use_layer_norm_decoder,
        )

    def _get_inference_input(
        self,
        tensors: dict[str, torch.Tensor | None],
    ) -> dict[str, torch.Tensor | None]:
        """Get input tensors for the inference process."""
        input_dict = {
            MODULE_KEYS.X_KEY: tensors[REGISTRY_KEYS.X_KEY],
            MODULE_KEYS.BATCH_INDEX_KEY: tensors[REGISTRY_KEYS.BATCH_KEY],
            MODULE_KEYS.CONT_COVS_KEY: tensors.get(REGISTRY_KEYS.CONT_COVS_KEY, None),
            MODULE_KEYS.CAT_COVS_KEY: tensors.get(REGISTRY_KEYS.CAT_COVS_KEY, None),
        }
        if self.is_cvae:
            input_dict["condition_index"] = tensors.get("condition", None)
        return input_dict

    def _get_generative_input(
        self,
        tensors: dict[str, torch.Tensor],
        inference_outputs: dict[str, torch.Tensor | Distribution | None],
    ) -> dict[str, torch.Tensor | None]:
        """Get input tensors for the generative process."""
        input_dict = {
            MODULE_KEYS.Z_KEY: inference_outputs[MODULE_KEYS.Z_KEY],
            MODULE_KEYS.BATCH_INDEX_KEY: tensors[REGISTRY_KEYS.BATCH_KEY],
            MODULE_KEYS.CONT_COVS_KEY: tensors.get(REGISTRY_KEYS.CONT_COVS_KEY, None),
            MODULE_KEYS.CAT_COVS_KEY: tensors.get(REGISTRY_KEYS.CAT_COVS_KEY, None),
        }
        if self.is_cvae:
            input_dict["condition_index"] = tensors.get("condition", None)
        return input_dict

    @auto_move_data
    def inference(
        self,
        x: torch.Tensor,
        batch_index: torch.Tensor,
        cont_covs: torch.Tensor | None = None,
        cat_covs: torch.Tensor | None = None,
        condition_index: torch.Tensor | None = None,
        n_samples: int = 1,
    ) -> dict[str, torch.Tensor | Distribution | None]:
        """Run the inference process with batch index validation."""
        batch_index = batch_index.long()
        if batch_index.max() >= self.n_batch or batch_index.min() < 0:
            raise ValueError(f"batch_index out of bounds: max={batch_index.max()}, min={batch_index.min()}, n_batch={self.n_batch}")

        encoder_inputs = [x]
        categorical_inputs = []
        if cont_covs is not None and self.encode_covariates:
            encoder_inputs.append(cont_covs)
        if self.batch_representation == "embedding" and self.encode_covariates:
            batch_rep = self.compute_embedding(REGISTRY_KEYS.BATCH_KEY, batch_index)
            encoder_inputs.append(batch_rep)
        else:
            categorical_inputs.append(batch_index)
        if self.is_cvae and condition_index is not None:
            condition_index = condition_index.long()
            if self.condition_representation == "embedding":
                condition_rep = self.compute_embedding("condition", condition_index)
                encoder_inputs.append(condition_rep)
            else:
                categorical_inputs.append(condition_index)
        if cat_covs is not None and self.encode_covariates:
            categorical_inputs.extend(torch.split(cat_covs, 1, dim=1))

        encoder_input = torch.cat(encoder_inputs, dim=-1)
        qz_params = self.z_encoder(encoder_input, *categorical_inputs)
        qz_m, qz_v = qz_params.chunk(2, dim=-1)
        qz = Normal(qz_m, torch.nn.functional.softplus(qz_v) + 1e-6)
        z = qz.rsample() if n_samples == 1 else qz.rsample((n_samples,))

        return {MODULE_KEYS.Z_KEY: z, MODULE_KEYS.QZ_KEY: qz}

    @auto_move_data
    def generative(
        self,
        z: torch.Tensor,
        batch_index: torch.Tensor,
        cont_covs: torch.Tensor | None = None,
        cat_covs: torch.Tensor | None = None,
        condition_index: torch.Tensor | None = None,
    ) -> dict[str, Distribution | None]:
        """Run the generative process with batch index validation."""
        batch_index = batch_index.long()
        if batch_index.max() >= self.n_batch or batch_index.min() < 0:
            raise ValueError(f"batch_index out of bounds: max={batch_index.max()}, min={batch_index.min()}, n_batch={self.n_batch}")

        decoder_inputs = [z]
        categorical_inputs = []
        if cont_covs is not None:
            if z.dim() != cont_covs.dim():
                decoder_inputs.append(cont_covs.unsqueeze(0).expand(z.size(0), -1, -1))
            else:
                decoder_inputs.append(cont_covs)
        if self.batch_representation == "embedding":
            batch_rep = self.compute_embedding(REGISTRY_KEYS.BATCH_KEY, batch_index)
            decoder_inputs.append(batch_rep)
        else:
            categorical_inputs.append(batch_index)
        if self.is_cvae and condition_index is not None:
            condition_index = condition_index.long()
            if self.condition_representation == "embedding":
                condition_rep = self.compute_embedding("condition", condition_index)
                decoder_inputs.append(condition_rep)
            else:
                categorical_inputs.append(condition_index)
        if cat_covs is not None:
            categorical_inputs.extend(torch.split(cat_covs, 1, dim=1))

        decoder_input = torch.cat(decoder_inputs, dim=-1)
        px_mean = self.decoder(decoder_input, *categorical_inputs)

        if self.output_var_param == "fixed":
            px_var = torch.ones_like(px_mean)
        elif self.output_var_param == "learned":
            log_var_clamped = torch.clamp(self.px_log_var, min=-10, max=2)
            px_var = torch.exp(log_var_clamped) * torch.ones_like(px_mean)
        else:  # "feature"
            log_var_clamped = torch.clamp(self.px_log_var, min=-10, max=2)
            px_var = torch.exp(log_var_clamped).unsqueeze(0).expand_as(px_mean)

        px_var = torch.clamp(px_var, min=1e-6, max=100.0)
        px_std = torch.sqrt(px_var)
        px = Normal(px_mean, px_std)

        return {MODULE_KEYS.PX_KEY: px}

    def loss(
        self,
        tensors: dict[str, torch.Tensor],
        inference_outputs: dict[str, torch.Tensor | Distribution | None],
        generative_outputs: dict[str, Distribution | None],
        kl_weight: float = 1.0,
        **kwargs,
    ) -> LossOutput:
        """Compute the loss for phenomics VAE with KL annealing and correlation loss."""
        if kwargs and not hasattr(self, '_logged_kwargs'):
            print(f"DEBUG: Loss function received additional kwargs: {list(kwargs.keys())}")
            self._logged_kwargs = True

        x = tensors[REGISTRY_KEYS.X_KEY]
        px = generative_outputs[MODULE_KEYS.PX_KEY]
        qz = inference_outputs[MODULE_KEYS.QZ_KEY]

        log_likelihood = px.log_prob(x).sum(-1)
        reconst_loss = -log_likelihood
        var_reg = self.var_reg_lambda * torch.mean(px.scale)
        reconst_loss = self.recon_beta * reconst_loss + var_reg

        kl_divergence_z = torch.distributions.kl_divergence(
            qz, Normal(torch.zeros_like(qz.loc), torch.ones_like(qz.scale))
        ).sum(dim=1)

        kl_anneal_epochs = kwargs.get('kl_anneal_epochs', 0)
        current_epoch = getattr(self, '_current_epoch', 0)
        effective_kl_weight = min(1.0, current_epoch / kl_anneal_epochs) if kl_anneal_epochs > 0 else kl_weight

        # Ensure tensors are proper shape for LossOutput
        mean_reconst_loss = torch.mean(reconst_loss)
        mean_kl_loss = torch.mean(kl_divergence_z)
        
        corr_loss = torch.tensor(0.0, device=x.device)
        if hasattr(self, 'use_corr_loss') and self.use_corr_loss:
            batch_id = tensors.get(REGISTRY_KEYS.BATCH_KEY, torch.zeros_like(x[:, 0])).float()
            z = inference_outputs[MODULE_KEYS.Z_KEY]
            
            # Handle both 2D and 3D z tensors
            if z.dim() == 3:  # Shape: [n_samples, batch_size, n_latent]
                n_samples = z.size(0)
                # Expand batch_id to match n_samples dimension, then add feature dimension
                batch_id = batch_id.unsqueeze(0).expand(n_samples, -1).unsqueeze(-1)  # Shape: [n_samples, batch_size, 1]
                z = z.view(-1, z.size(-1))  # Flatten to [n_samples * batch_size, n_latent]
                batch_id = batch_id.view(-1, 1)  # Flatten to [n_samples * batch_size, 1]
            else:  # Shape: [batch_size, n_latent]
                # batch_id is already [batch_size], just need to add feature dimension
                if batch_id.dim() == 1:
                    batch_id = batch_id.unsqueeze(-1)  # Shape: [batch_size, 1]
                # If batch_id is already [batch_size, 1], keep it as is
            
            corr_matrix = torch.corrcoef(torch.cat([z, batch_id], dim=-1).T)
            corr_loss = torch.abs(corr_matrix[:-1, -1]).mean() * 0.005

        total_loss = mean_reconst_loss + effective_kl_weight * mean_kl_loss + corr_loss

        # Use the same format as the standard VAE implementation
        return LossOutput(
            loss=total_loss,
            reconstruction_loss=reconst_loss,  # Pass the full tensor, not the mean
            kl_local={MODULE_KEYS.KL_Z_KEY: kl_divergence_z},  # Pass the full tensor, not the mean
            extra_metrics={"corr_loss": corr_loss}
        )

    @torch.inference_mode()
    def sample(
        self,
        tensors: dict[str, torch.Tensor],
        n_samples: int = 1,
    ) -> torch.Tensor:
        """Sample from the generative model."""
        inference_outputs = self.inference(**self._get_inference_input(tensors), n_samples=n_samples)
        generative_outputs = self.generative(**self._get_generative_input(tensors, inference_outputs))
        px = generative_outputs[MODULE_KEYS.PX_KEY]
        return px.sample()
