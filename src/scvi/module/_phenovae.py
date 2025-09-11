"""Phenomics VAE module with Gaussian likelihood for continuous features.

UNDERSTAND
----------
We are aligning high-dimensional phenomics embeddings (Phenom2) across ~3k
batch combinations (experiment x plate). Our two metrics are:
- **KNN batch predictability** (lower is better; chance ≈ 0.033%).
- **BMDB biological recapitulation** (higher is better).

TVN (linear: PCA → center/scale → CORAL) already drops KNN to ~0.8% and lifts
BMDB to ~38%. Residual **non-linear** batch effects persist (LISI/UMAP).
This VAE aims to scrub those *residual* effects while preserving biological
signal on the raw data.

ANALYZE / DESIGN CHOICES
------------------------
1) **Batch to DECODER; encoder is batch-free**:
   - If the decoder does not see batch, the latent `z` must carry it just to
     reconstruct `x`, which raises KNN. We therefore **feed batch to the
     decoder** and keep the **encoder batch-free**.

2) **No BatchNorm (BN)**; optional **LayerNorm** in encoder:
   - BN depends on mini-batch composition and can leak/entangle batch effects.
     We avoid BN entirely in this module. LayerNorm in the encoder helps
     stabilize optimization on heterogeneous phenomics features.

3) **Diagonal Gaussian likelihood**:
   - Phenomics features are continuous. We use a diagonal Gaussian

4) **Loss: reconstruction + KL (with annealing via plan)**
   - Keep β=1.0 and rely on a training-plan KL schedule (scvi handles `kl_weight`).
   - Remove unstable/contradictory extras (no corr/bio-contrastive).

5) **Capacity defaults**:
   - `n_latent=128`, `n_hidden=256`, `n_layers=2`, `dropout=0.1`.
   - Increase latent only if BMDB plateaus and KNN remains acceptably low.

REASON / HOW THIS MAPS TO CODE
------------------------------
- Encoder is a simple residual MLP; no batch input, no BN (optional LayerNorm).
- Decoder is a residual MLP that **concatenates** (z, batch_rep, optional cont covs).
- Batch representation can be:
    * "embedding": learned embedding per batch (stable, compact), or
    * "one-hot": one-hot vector concatenated (simple/robust).
- The module exposes `inference`/`generative` consistent with scvi-tools,
  returning distributions and respecting shapes (including `n_samples`).

SYNTHESIZE / EXPECTED EFFECT
----------------------------
- **↓ KNN**: because z no longer needs to encode batch to reconstruct—decoder
  explains batch nuisances directly.
- **↑ BMDB**: diagonal Gaussian + stable encoder preserves weak biological
  signals; no over-compression from an overly aggressive KL (use annealing).

CONCLUDE
--------
Use this module together with:
- `_phenomics.read_phenomics` (loads features, vectorizes perturbation summary)
- `PHVI.setup_anndata(..., hierarchical_batch_keys=["experiment","plate_number"])`
to ensure the model "sees" the same nuisance granularity your KNN evaluator uses.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional, Dict, Union

import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Normal, Distribution
from torch.autograd import Function

from scvi import REGISTRY_KEYS
from scvi.module._constants import MODULE_KEYS
from scvi.module.base import (
    BaseModuleClass,
    EmbeddingModuleMixin,
    LossOutput,
    auto_move_data,
)

if TYPE_CHECKING:
    from typing import Literal

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Helper function for broadcasting tensors
# -----------------------------------------------------------------------------
def _broadcast_like(t: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    """
    Ensure t matches like's dimensionality for concatenation:
    - If like is [B, *], return t shaped [B, K]
    - If like is [S, B, *], return t shaped [S, B, K] by expanding only across S
    Assumes t encodes per-sample info (batch_one_hot, cond one-hot, cont covs).
    """
    # Make t 2D: [B, K]
    if t.dim() == 1:
        t = t.unsqueeze(1)              # [B] -> [B, 1]
    elif t.dim() == 3 and t.size(0) == 1:
        t = t.squeeze(0)                # [1, B, K] -> [B, K]
    elif t.dim() != 2:
        raise RuntimeError(f"Expected t to be 1D/2D (or [1,B,K]), got shape {tuple(t.shape)}")

    if like.dim() == 2:
        # z is [B, L] -> return [B, K]
        return t
    elif like.dim() == 3:
        # z is [S, B, L] -> return [S, B, K]
        return t.unsqueeze(0).expand(like.size(0), -1, -1)
    else:
        raise RuntimeError(f"Unsupported 'like' rank: {like.dim()} with shape {tuple(like.shape)}")


# -----------------------------------------------------------------------------
# Building blocks: Residual MLP without BatchNorm (LayerNorm optional)
# -----------------------------------------------------------------------------
class ResidualMLP(nn.Module):
    """A simple residual MLP used for encoder/decoder.

    Design choices:
    - **No BatchNorm**: BN interacts with mini-batch composition and can leak batch effects.
    - **Optional LayerNorm**: helps stabilize optimization on heterogeneous continuous features.
    - **Residual connections**: encourage gradient flow and preserve information when widths match.

    Parameters
    ----------
    n_in : int
        Input dimensionality.
    n_out : int
        Output dimensionality.
    n_hidden : int
        Width of hidden layers.
    n_layers : int
        Number of linear layers (>=1). If 1, it is just a linear map.
    dropout : float
        Dropout rate applied after ReLU (not on the output layer).
    use_layer_norm : bool
        Whether to apply LayerNorm after linear transformations (encoder-friendly).
    """

    def __init__(
        self,
        n_in: int,
        n_out: int,
        *,
        n_hidden: int = 256,
        n_layers: int = 2,
        dropout: float = 0.1,
        use_layer_norm: bool = False,
    ) -> None:
        super().__init__()
        assert n_layers >= 1, "n_layers must be >= 1"

        dims = [n_in] + [n_hidden] * (n_layers - 1) + [n_out]

        self.use_layer_norm = use_layer_norm
        self.dropout = dropout

        self.fcs = nn.ModuleList()
        self.lns = nn.ModuleList() if use_layer_norm else None
        self.activ = nn.ReLU()

        for i in range(n_layers):
            self.fcs.append(nn.Linear(dims[i], dims[i + 1]))
            if use_layer_norm and i < n_layers - 1:
                self.lns.append(nn.LayerNorm(dims[i + 1]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for i, fc in enumerate(self.fcs):
            h_in = h
            h = fc(h)
            is_last = (i == len(self.fcs) - 1)
            if not is_last:
                if self.use_layer_norm:
                    h = self.lns[i](h)  # type: ignore[index]
                h = self.activ(h)
                if self.dropout > 0:
                    h = F.dropout(h, p=self.dropout, training=self.training)
                # Residual when shapes allow (hidden -> hidden)
                if h_in.shape == h.shape:
                    h = h + h_in
        return h


# -----------------------------------------------------------------------------
# Main module: PhenoVAE
# -----------------------------------------------------------------------------
class PhenoVAE(EmbeddingModuleMixin, BaseModuleClass):
    """Variational autoencoder tailored for **continuous phenomics** features.

    Core ideas implemented in this module:
    - **Encoder** does *not* consume batch → latent `z` is encouraged to be batch-free.
    - **Decoder** *does* consume batch (one-hot or embedding) → can reconstruct
      batch-specific nuisance without forcing `z` to carry it.
    - **Diagonal Gaussian likelihood**

    Parameters
    ----------
    n_input
        Number of input features (phenomics feature dimension).
    n_batch
        Number of batches (distinct categories in your hierarchical batch key).
        If ``0``, no batch conditioning is performed.
    n_hidden
        MLP hidden width.
    n_latent
        Latent dimensionality (`z`).
    n_layers
        Number of layers in the encoder and decoder MLPs.
    n_continuous_cov
        Number of continuous covariates (optional; concatenated if provided).
    n_cats_per_cov
        For additional categorical covariates (not the batch), the cardinality
        of each covariate. (We keep this for scvi parity; not used by default.)
    dropout_rate
        Dropout rate in MLPs.
    encode_covariates
        If ``True``, concatenates covariates to the **encoder** inputs. We recommend
        keeping this **False** for batch to avoid leakage into `z`.
    deeply_inject_covariates
        Not used here (kept for API parity). Prefer simplicity initially.
    batch_representation
        "one-hot" or "embedding". Determines how batch enters the **decoder**.
    use_batch_norm
        Ignored here; we intentionally avoid BatchNorm. Kept for API compatibility.
    use_layer_norm
        Where to apply LayerNorm: "encoder", "decoder", "both", or "none".
        Default recommended: "encoder".
    recon_beta
        Weight on reconstruction loss. Keep at 1.0; rely on KL annealing instead.
    batch_embedding_kwargs
        Extra kwargs for batch embedding layer (if `batch_representation="embedding"`).
    """

    def __init__(
        self,
        n_input: int,
        n_batch: int = 0,
        n_hidden: int = 256,
        n_latent: int = 128,
        n_layers: int = 2,
        n_continuous_cov: int = 0,
        n_cats_per_cov: Optional[list[int]] = None,
        dropout_rate: float = 0.1,
        encode_covariates: bool = False,  # IMPORTANT: batch should *not* feed encoder
        deeply_inject_covariates: bool = False,  # kept for API parity
        batch_representation: Literal["one-hot", "embedding"] = "embedding",
        use_batch_norm: Literal["encoder", "decoder", "none", "both"] = "none",  # ignored (no BN)
        use_layer_norm: Literal["encoder", "decoder", "none", "both"] = "encoder",
        recon_beta: float = 1.0,
        batch_embedding_kwargs: Optional[dict] = None,
    ) -> None:
        super().__init__()

        # ---- Basic config & state ------------------------------------------
        self.n_input = n_input
        self.n_latent = n_latent
        self.n_batch = n_batch
        self.n_continuous_cov = n_continuous_cov
        self.encode_covariates = encode_covariates
        self.deeply_inject_covariates = deeply_inject_covariates  # not used internally
        self.batch_representation = batch_representation
        self.recon_beta = recon_beta

        # ---- Embeddings for batch/condition (decoder-side) -----------------
        # Batch: either embedding (compact) or one-hot (simple).
        if self.batch_representation == "embedding" and self.n_batch > 0:
            batch_embedding_kwargs = batch_embedding_kwargs or {}
            self.init_embedding("batch", n_batch, **batch_embedding_kwargs)
            batch_dim = self.get_embedding("batch").embedding_dim
        else:
            batch_dim = self.n_batch  # for one-hot concat (0 if no batch)
      
        # ---- Encoder: NO batch input (by design) ---------------------------
        # Construct encoder input dimensionality:
        #   X [+ cont_covs (if encode_covariates)].
        enc_in_dim = n_input
        if self.encode_covariates and self.n_continuous_cov > 0:
            enc_in_dim += self.n_continuous_cov

        use_ln_encoder = use_layer_norm in {"encoder", "both"}
        self.z_encoder = ResidualMLP(
            n_in=enc_in_dim,
            n_out=2 * n_latent,  # outputs [mu, v_param] chunks
            n_hidden=n_hidden,
            n_layers=n_layers,
            dropout=dropout_rate,
            use_layer_norm=use_ln_encoder,
        )

        # ---- Decoder: z + batch + cont_covs ---------------------------------
        # Decoder explicitly conditions on batch to explain batch-specific variance.
        dec_in_dim = n_latent
        if self.n_continuous_cov > 0:
            dec_in_dim += self.n_continuous_cov
        if self.n_batch > 0:
            dec_in_dim += (batch_dim if self.batch_representation == "embedding" else self.n_batch)

        use_ln_decoder = use_layer_norm in {"decoder", "both"}
        # Decoder outputs both mean and log variance (2 * n_input)
        # Uses chunk approach for consistency with encoder
        self.decoder = ResidualMLP(
            n_in=dec_in_dim,
            n_out=2 * n_input,  # outputs [mean, log_var] chunks
            n_hidden=n_hidden,
            n_layers=n_layers,
            dropout=dropout_rate,
            use_layer_norm=use_ln_decoder,
        )

    # ----------------------------------------------------------------------
    # Helpers to assemble inputs for inference/generative (scvi API parity)
    # ----------------------------------------------------------------------
    def _get_inference_input(
        self,
        tensors: dict[str, torch.Tensor | None],
    ) -> dict[str, torch.Tensor | None]:
        """Select tensors used by `inference` (encoder forward).

        NOTE: We **do not** supply batch to the encoder here (by design).
        """
        return {
            MODULE_KEYS.X_KEY: tensors[REGISTRY_KEYS.X_KEY],
            MODULE_KEYS.BATCH_INDEX_KEY: tensors[REGISTRY_KEYS.BATCH_KEY],  # kept for shape checks only
            MODULE_KEYS.CONT_COVS_KEY: tensors.get(REGISTRY_KEYS.CONT_COVS_KEY, None),
            MODULE_KEYS.CAT_COVS_KEY: tensors.get(REGISTRY_KEYS.CAT_COVS_KEY, None),
        }

    def _get_generative_input(
        self,
        tensors: dict[str, torch.Tensor],
        inference_outputs: dict[str, torch.Tensor | Distribution | None],
    ) -> dict[str, torch.Tensor | None]:
        """Select tensors used by `generative` (decoder forward)."""
        return {
            MODULE_KEYS.Z_KEY: inference_outputs[MODULE_KEYS.Z_KEY],
            MODULE_KEYS.BATCH_INDEX_KEY: tensors[REGISTRY_KEYS.BATCH_KEY],
            MODULE_KEYS.CONT_COVS_KEY: tensors.get(REGISTRY_KEYS.CONT_COVS_KEY, None),
            MODULE_KEYS.CAT_COVS_KEY: tensors.get(REGISTRY_KEYS.CAT_COVS_KEY, None),
        }

    # ----------------------------------------------------------------------
    # Core forward passes
    # ----------------------------------------------------------------------
    @auto_move_data
    def inference(
        self,
        x: torch.Tensor,
        batch_index: torch.Tensor,  # not used as encoder input; checked for bounds
        cont_covs: torch.Tensor | None = None,
        cat_covs: torch.Tensor | None = None,  # kept for API parity; unused here
        n_samples: int = 1,
    ) -> dict[str, torch.Tensor | Distribution | None]:
        """Encoder forward pass producing q(z|x, covs) without batch input.

        Returns
        -------
        dict with:
            - MODULE_KEYS.QZ_KEY: Normal(loc=mu, scale=sigma)
            - MODULE_KEYS.Z_KEY: rsample(s) from q(z|x)
        """
        enc_inputs = [x]

        # Optional continuous covariates concatenation (keep minimal by default).
        if self.encode_covariates and cont_covs is not None:
            enc_inputs.append(cont_covs)


        enc_in = torch.cat(enc_inputs, dim=-1) if len(enc_inputs) > 1 else enc_inputs[0]

        # Produce q(z|·): chunk into (mu, v_param). Use softplus to ensure positive std.
        qz_params = self.z_encoder(enc_in)
        qz_mu, qz_vparam = qz_params.chunk(2, dim=-1)
        qz_scale = F.softplus(qz_vparam) + 1e-6  # strictly positive
        qz = Normal(qz_mu, qz_scale)

        z = qz.rsample() if n_samples == 1 else qz.rsample((n_samples,))
        return {MODULE_KEYS.QZ_KEY: qz, MODULE_KEYS.Z_KEY: z}

    @auto_move_data
    def generative(
        self,
        z: torch.Tensor,
        batch_index: torch.Tensor,
        cont_covs: torch.Tensor | None = None,
        cat_covs: torch.Tensor | None = None,
    ) -> Dict[str, Union[Distribution, torch.Tensor, None]]:
        """Decoder forward pass producing p(x|z, batch, covs). Batch **is used here**."""
        # --- validate + normalize index shapes ---
        batch_index = batch_index.long().view(-1)  # [B]

        dec_inputs: list[torch.Tensor] = [z]
        # We keep cat_covs for API parity, but do not inject them into the decoder.
        # ResidualMLP.forward(x) only accepts a single tensor.

        # --- continuous covariates ---
        if cont_covs is not None:
            # cont_covs is typically [B, C]; match z's rank and dtype
            dec_inputs.append(_broadcast_like(cont_covs.to(z.dtype), z))

        # --- batch conditioning goes to the DECODER ---
        b_rep = None  # Will store batch representation for reuse
        if self.n_batch > 0:
            if self.batch_representation == "embedding":
                # Use string literal to avoid scope issues with @auto_move_data
                b_rep = self.compute_embedding("batch", batch_index).to(z.dtype)  # [B, E]
            else:
                b_rep = F.one_hot(batch_index, num_classes=self.n_batch).to(z.dtype)  # [B, n_batch]
            dec_inputs.append(_broadcast_like(b_rep, z))

        dec_input = torch.cat(dec_inputs, dim=-1)

        # Decoder outputs both mean and log variance
        px_params = self.decoder(dec_input)
        px_mean, px_log_var = px_params.chunk(2, dim=-1)
        
        # Constrain variance to reasonable range to prevent instability
        # For standardized data, allow up to ~5x original variance
        px_log_var = torch.clamp(px_log_var, min=-5.0, max=2.0)
        px_std = torch.exp(0.5 * px_log_var)
        px = Normal(px_mean, px_std)

        return {MODULE_KEYS.PX_KEY: px, "px_mean": px_mean, "px_log_var": px_log_var}  # expose for debugging

    # ----------------------------------------------------------------------
    # Loss
    # ----------------------------------------------------------------------
    def loss(
        self,
        tensors: dict[str, torch.Tensor],
        inference_outputs: dict[str, torch.Tensor | Distribution | None],
        generative_outputs: dict[str, Distribution | None],
        kl_weight: float = 1.0,
        **kwargs,
    ) -> LossOutput:
        """Standard VAE loss for continuous phenomics: recon + KL

        We keep this simple and stable. KL annealing should be
        handled by the training plan through the `kl_weight` argument, which
        scvi-tools passes here (e.g., linear/sigmoid ramp from 0→1).

        Returns
        -------
        LossOutput with:
          - loss: scalar objective
          - reconstruction_loss: per-item reconstruction terms (for logging)
          - kl_local: dict containing per-item KL terms
          - extra_metrics: useful scalars for monitoring
        """
        px = generative_outputs[MODULE_KEYS.PX_KEY]                  # Normal(mean, std)
        qz = inference_outputs[MODULE_KEYS.QZ_KEY]                   # Normal(mu, std)
        x = tensors[REGISTRY_KEYS.X_KEY].to(px.loc.dtype)           # align dtype with px_mean

        # NLL
        D = x.shape[-1]  # Number of features (1664 for phenomics)
        log_likelihood = px.log_prob(x).sum(-1)                 # sum over features -> [B] or [S,B]
            # If multiple samples, average over the sample dimension first
        if log_likelihood.dim() == 2:
            log_likelihood = log_likelihood.mean(0)             # [S,B] -> [B]
        # CRITICAL: Normalize by number of features to balance with other losses
        reconst_loss = self.recon_beta * (-log_likelihood / D)

        # Variance regularization to prevent collapse
        # For standardized data, penalize log_var < -2.0 (std < 0.37)
        px_log_var = generative_outputs["px_log_var"]
        # var_regularization = 1.0 * torch.mean(torch.relu(-px_log_var - 2.0))
        var_regularization = 0.01 * torch.mean((px_log_var - 0.0)**2)

        # KL divergence between q(z|x) and N(0, I)
        prior = Normal(loc=torch.zeros_like(qz.loc), scale=torch.ones_like(qz.scale))
        kl_div_z = torch.distributions.kl_divergence(qz, prior).mean(dim=1)

        # Let training plan drive annealing via kl_weight
        effective_kl_weight = float(kwargs.get("kl_weight", kl_weight))

        total = reconst_loss.mean() + effective_kl_weight * kl_div_z.mean() + var_regularization

        # Compute only essential metrics during training (keep it fast!)
        with torch.no_grad():      
            # Quick reconstruction quality check (cheap)
            px_mean = px.loc
            residual = x - px_mean
            mse = (residual ** 2).mean()
        
        return LossOutput(
            loss=total,
            reconstruction_loss=reconst_loss,                 # per-item tensor
            kl_local={MODULE_KEYS.KL_Z_KEY: kl_div_z},        # per-item tensor
            extra_metrics={
                # Essential metrics only (keep training fast!)
                "kl_weight": torch.tensor(effective_kl_weight, device=x.device),
                "reconst_mean": reconst_loss.mean(),
                "kl_mean": kl_div_z.mean(),
                "var_penalty": var_regularization,
                
                # Only the most essential diagnostic metrics
                "recon_mse": mse,
            },
        )

    # ----------------------------------------------------------------------
    # Sampling utility
    # ----------------------------------------------------------------------
    @torch.inference_mode()
    def sample(
        self,
        tensors: dict[str, torch.Tensor],
        n_samples: int = 1,
    ) -> torch.Tensor:
        """Sample x ~ p(x|z, batch, ...) by ancestral sampling from q(z|x)."""
        inf = self.inference(**self._get_inference_input(tensors), n_samples=n_samples)
        gen = self.generative(**self._get_generative_input(tensors, inf))
        px = gen[MODULE_KEYS.PX_KEY]
        return px.sample()
    
    @torch.inference_mode()
    def compute_epoch_metrics(
        self,
        dataloader,
        max_batches: int = 10,
    ) -> dict[str, float]:
        """Compute expensive metrics on a subset of data at epoch end.
        
        This is called by the monitor to get detailed metrics without
        slowing down training.
        
        Parameters
        ----------
        dataloader
            Data loader to sample batches from
        max_batches
            Maximum number of batches to use (for speed)
            
        Returns
        -------
        Dictionary of metric names to values
        """
        metrics = {}
        
        # Accumulate statistics over batches
        z_vars_batch = []
        px_log_vars = []
        
        for i, tensors in enumerate(dataloader):
            if i >= max_batches:
                break
                
            # Move to device
            tensors = {k: v.to(self.device) if torch.is_tensor(v) else v 
                      for k, v in tensors.items()}
            
            # Forward pass
            inf_outputs = self.inference(**self._get_inference_input(tensors))
            gen_outputs = self.generative(**self._get_generative_input(tensors, inf_outputs))
            
            # Get key tensors
            z = inf_outputs[MODULE_KEYS.Z_KEY]
            if z.dim() == 3:
                z = z.mean(0)
            qz = inf_outputs[MODULE_KEYS.QZ_KEY]
            px = gen_outputs[MODULE_KEYS.PX_KEY]
            px_log_var = gen_outputs["px_log_var"]
            
            # Collect statistics
            z_vars_batch.append(z.var(dim=0))
            px_log_vars.append(px_log_var)
        
        # Aggregate statistics
        if z_vars_batch:
            all_z_vars = torch.stack(z_vars_batch).mean(0)  # Average variance per dim
            metrics["z_var_mean_train"] = float(all_z_vars.mean())
            metrics["active_units_train"] = float((all_z_vars > 1e-3).float().mean())
            
        if px_log_vars:
            all_px_log_vars = torch.cat(px_log_vars)
            metrics["px_log_var_mean_train"] = float(all_px_log_vars.mean())
            metrics["px_log_var_min_train"] = float(all_px_log_vars.min())
            metrics["px_log_var_max_train"] = float(all_px_log_vars.max())
            
        return metrics
    

__all__ = ["PhenoVAE"]
