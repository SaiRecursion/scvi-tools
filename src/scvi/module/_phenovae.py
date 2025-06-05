"""Phenomics VAE module with Gaussian likelihood for continuous features."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
from torch.distributions import Normal

from scvi import REGISTRY_KEYS
from scvi.module._constants import MODULE_KEYS
from scvi.module.base import BaseModuleClass, EmbeddingModuleMixin, LossOutput, auto_move_data

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Literal

    from torch.distributions import Distribution

logger = logging.getLogger(__name__)


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
    dropout_rate
        Dropout rate.
    encode_covariates
        If ``True``, covariates are concatenated to features prior to encoding.
    deeply_inject_covariates
        If ``True`` and ``n_layers > 1``, covariates are concatenated to hidden layer outputs.
    batch_representation
        Method for encoding batch information. One of ``"one-hot"`` or ``"embedding"``.
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
    """

    def __init__(
        self,
        n_input: int,
        n_batch: int = 0,
        n_hidden: int = 128,
        n_latent: int = 10,
        n_layers: int = 1,
        n_continuous_cov: int = 0,
        n_cats_per_cov: list[int] | None = None,
        dropout_rate: float = 0.1,
        encode_covariates: bool = False,
        deeply_inject_covariates: bool = True,
        batch_representation: Literal["one-hot", "embedding"] = "one-hot",
        use_batch_norm: Literal["encoder", "decoder", "none", "both"] = "both",
        use_layer_norm: Literal["encoder", "decoder", "none", "both"] = "none",
        var_activation: Callable[[torch.Tensor], torch.Tensor] = None,
        output_var_param: Literal["learned", "fixed", "feature"] = "learned",
        batch_embedding_kwargs: dict | None = None,
    ):
        from scvi.nn import Encoder

        super().__init__()

        self.n_latent = n_latent
        self.n_batch = n_batch
        self.n_input = n_input
        self.encode_covariates = encode_covariates
        self.output_var_param = output_var_param

        # Initialize output variance parameters
        if self.output_var_param == "learned":
            # Single learnable log variance
            self.px_log_var = torch.nn.Parameter(torch.zeros(1))
        elif self.output_var_param == "feature":
            # Per-feature learnable log variance
            self.px_log_var = torch.nn.Parameter(torch.zeros(n_input))
        # For "fixed", we don't need parameters

        self.batch_representation = batch_representation
        if self.batch_representation == "embedding":
            self.init_embedding(REGISTRY_KEYS.BATCH_KEY, n_batch, **(batch_embedding_kwargs or {}))
            batch_dim = self.get_embedding(REGISTRY_KEYS.BATCH_KEY).embedding_dim
        elif self.batch_representation != "one-hot":
            raise ValueError("`batch_representation` must be one of 'one-hot', 'embedding'.")

        use_batch_norm_encoder = use_batch_norm == "encoder" or use_batch_norm == "both"
        use_batch_norm_decoder = use_batch_norm == "decoder" or use_batch_norm == "both"
        use_layer_norm_encoder = use_layer_norm == "encoder" or use_layer_norm == "both"
        use_layer_norm_decoder = use_layer_norm == "decoder" or use_layer_norm == "both"

        n_input_encoder = n_input + n_continuous_cov * encode_covariates
        if self.batch_representation == "embedding":
            n_input_encoder += batch_dim * encode_covariates
            cat_list = list([] if n_cats_per_cov is None else n_cats_per_cov)
        else:
            cat_list = [n_batch] + list([] if n_cats_per_cov is None else n_cats_per_cov)

        encoder_cat_list = cat_list if encode_covariates else None

        # Encoder for latent representation
        self.z_encoder = Encoder(
            n_input_encoder,
            n_latent,
            n_cat_list=encoder_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            distribution="normal",
            inject_covariates=deeply_inject_covariates,
            use_batch_norm=use_batch_norm_encoder,
            use_layer_norm=use_layer_norm_encoder,
            var_activation=var_activation,
            return_dist=True,
        )

        n_input_decoder = n_latent + n_continuous_cov
        if self.batch_representation == "embedding":
            n_input_decoder += batch_dim

        # Custom decoder for phenomics that outputs mean and log variance
        self.decoder = PhenomicsDecoder(
            n_input_decoder,
            n_input,
            n_cat_list=cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            inject_covariates=deeply_inject_covariates,
            use_batch_norm=use_batch_norm_decoder,
            use_layer_norm=use_layer_norm_decoder,
        )

    def _get_inference_input(
        self,
        tensors: dict[str, torch.Tensor | None],
    ) -> dict[str, torch.Tensor | None]:
        """Get input tensors for the inference process."""
        return {
            MODULE_KEYS.X_KEY: tensors[REGISTRY_KEYS.X_KEY],
            MODULE_KEYS.BATCH_INDEX_KEY: tensors[REGISTRY_KEYS.BATCH_KEY],
            MODULE_KEYS.CONT_COVS_KEY: tensors.get(REGISTRY_KEYS.CONT_COVS_KEY, None),
            MODULE_KEYS.CAT_COVS_KEY: tensors.get(REGISTRY_KEYS.CAT_COVS_KEY, None),
        }

    def _get_generative_input(
        self,
        tensors: dict[str, torch.Tensor],
        inference_outputs: dict[str, torch.Tensor | Distribution | None],
    ) -> dict[str, torch.Tensor | None]:
        """Get input tensors for the generative process."""
        return {
            MODULE_KEYS.Z_KEY: inference_outputs[MODULE_KEYS.Z_KEY],
            MODULE_KEYS.BATCH_INDEX_KEY: tensors[REGISTRY_KEYS.BATCH_KEY],
            MODULE_KEYS.CONT_COVS_KEY: tensors.get(REGISTRY_KEYS.CONT_COVS_KEY, None),
            MODULE_KEYS.CAT_COVS_KEY: tensors.get(REGISTRY_KEYS.CAT_COVS_KEY, None),
        }

    @auto_move_data
    def inference(
        self,
        x: torch.Tensor,
        batch_index: torch.Tensor,
        cont_covs: torch.Tensor | None = None,
        cat_covs: torch.Tensor | None = None,
        n_samples: int = 1,
    ) -> dict[str, torch.Tensor | Distribution | None]:
        """Run the inference process."""
        x_ = x  # No log transformation needed for continuous features

        if cont_covs is not None and self.encode_covariates:
            encoder_input = torch.cat((x_, cont_covs), dim=-1)
        else:
            encoder_input = x_

        if cat_covs is not None and self.encode_covariates:
            categorical_input = torch.split(cat_covs, 1, dim=1)
        else:
            categorical_input = ()

        if self.batch_representation == "embedding" and self.encode_covariates:
            batch_rep = self.compute_embedding(REGISTRY_KEYS.BATCH_KEY, batch_index)
            encoder_input = torch.cat([encoder_input, batch_rep], dim=-1)
            qz, z = self.z_encoder(encoder_input, *categorical_input)
        else:
            qz, z = self.z_encoder(encoder_input, batch_index, *categorical_input)

        if n_samples > 1:
            untran_z = qz.sample((n_samples,))
            z = self.z_encoder.z_transformation(untran_z)

        return {
            MODULE_KEYS.Z_KEY: z,
            MODULE_KEYS.QZ_KEY: qz,
        }

    @auto_move_data
    def generative(
        self,
        z: torch.Tensor,
        batch_index: torch.Tensor,
        cont_covs: torch.Tensor | None = None,
        cat_covs: torch.Tensor | None = None,
    ) -> dict[str, Distribution | None]:
        """Run the generative process."""
        # Prepare decoder input
        if cont_covs is None:
            decoder_input = z
        elif z.dim() != cont_covs.dim():
            decoder_input = torch.cat(
                [z, cont_covs.unsqueeze(0).expand(z.size(0), -1, -1)], dim=-1
            )
        else:
            decoder_input = torch.cat([z, cont_covs], dim=-1)

        if cat_covs is not None:
            categorical_input = torch.split(cat_covs, 1, dim=1)
        else:
            categorical_input = ()

        if self.batch_representation == "embedding":
            batch_rep = self.compute_embedding(REGISTRY_KEYS.BATCH_KEY, batch_index)
            decoder_input = torch.cat([decoder_input, batch_rep], dim=-1)
            px_mean = self.decoder(decoder_input, *categorical_input)
        else:
            px_mean = self.decoder(decoder_input, batch_index, *categorical_input)

        # Get variance
        if self.output_var_param == "fixed":
            px_var = torch.ones_like(px_mean)
        elif self.output_var_param == "learned":
            px_var = torch.exp(self.px_log_var) * torch.ones_like(px_mean)
        else:  # "feature"
            px_var = torch.exp(self.px_log_var).unsqueeze(0).expand_as(px_mean)

        # Create Gaussian distribution
        px = Normal(px_mean, torch.sqrt(px_var))

        return {MODULE_KEYS.PX_KEY: px}

    def loss(
        self,
        tensors: dict[str, torch.Tensor],
        inference_outputs: dict[str, torch.Tensor | Distribution | None],
        generative_outputs: dict[str, Distribution | None],
        kl_weight: float = 1.0,
    ) -> LossOutput:
        """Compute the loss."""
        x = tensors[REGISTRY_KEYS.X_KEY]
        px = generative_outputs[MODULE_KEYS.PX_KEY]
        qz = inference_outputs[MODULE_KEYS.QZ_KEY]

        # Reconstruction loss - negative log likelihood
        reconst_loss = -px.log_prob(x).sum(-1)

        # KL divergence
        kl_divergence_z = torch.distributions.kl_divergence(
            qz, Normal(torch.zeros_like(qz.loc), torch.ones_like(qz.scale))
        ).sum(dim=1)

        # Weighted ELBO
        weighted_kl = kl_weight * kl_divergence_z
        loss = torch.mean(reconst_loss + weighted_kl)

        return LossOutput(
            loss=loss,
            reconstruction_loss=reconst_loss,
            kl_local={MODULE_KEYS.KLD_KEY: kl_divergence_z},
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


class PhenomicsDecoder(torch.nn.Module):
    """Decoder for phenomics data that outputs continuous values."""

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_cat_list: list[int] | None = None,
        n_layers: int = 1,
        n_hidden: int = 128,
        inject_covariates: bool = True,
        use_batch_norm: bool = False,
        use_layer_norm: bool = False,
    ):
        from scvi.nn import FCLayers

        super().__init__()
        self.decoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=0,
            inject_covariates=inject_covariates,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
        )

        # Linear layer for mean output
        self.mean_decoder = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x: torch.Tensor, *cat_list: torch.Tensor):
        """Forward pass."""
        # Get hidden representation
        h = self.decoder(x, *cat_list)
        # Output mean
        px_mean = self.mean_decoder(h)
        return px_mean
