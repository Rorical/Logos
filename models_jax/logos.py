"""Logos transformer in Flax — sub-quadratic decoder-only architecture.

Entry → Body (looped) → Exit with Block Attention Residuals.
KDA + Snapshot Retrieval + Local SWA + MoE.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn

from models_jax.base import RMSNorm, SwiGLU, count_parameters, model_summary, default_kernel_init
from models_jax.kda import SuperKimiDeltaAttention
from models_jax.retrieval import SnapshotRetrieval
from models_jax.swa import LocalAttention
from models_jax.residual import BlockAttentionResidual
from models_jax.moe import MoELayer
from models_jax.lm_loss import chunked_linear_cross_entropy, standard_lm_cross_entropy


@dataclass
class LogosConfig:
    vocab_size: int = 32000
    d_model: int = 512
    max_seq_len: int = 2048

    num_heads: int = 8
    head_dim: int = 64
    dropout: float = 0.0
    norm_eps: float = 1e-6

    d_ff: int = 1364

    use_moe: bool = True
    num_shared_experts: int = 2
    num_sparse_experts: int = 64
    top_k: int = 6
    expert_d_ff: int = 256
    bias_update_rate: float = 0.01
    capacity_factor: float = 2.0
    moe_diversity_factor: float = 0.0

    chunk_size: int = 64
    conv_size: int = 4
    A_init_min: float = 1.0
    A_init_max: float = 16.0

    snapshot_interval: int = 256
    snapshot_latent_dim: int = 128
    mem_top_k: int = 16
    mem_head_dim: int = 64
    mem_latent_dim: int = 128

    rope_base: float = 10000.0
    qk_norm: bool = True
    partial_rope_dim: Optional[int] = None
    attention_sink: bool = True
    rope_scaling_type: str = "none"
    rope_scaling_factor: float = 1.0
    rope_original_max_position: Optional[int] = None
    yarn_beta_fast: float = 32.0
    yarn_beta_slow: float = 1.0

    swa_window: int = 256
    swa_every: int = 4
    swa_offset: int = 3

    num_entry_layers: int = 2
    num_body_layers: int = 4
    num_exit_layers: int = 2
    num_loops: int = 4

    lm_head_chunk_size: int = 0

    def __post_init__(self):
        if self.d_model % self.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even")
        if self.num_body_layers <= 0 or self.num_loops <= 0:
            raise ValueError("num_body_layers and num_loops must be > 0")
        if self.num_entry_layers < 0 or self.num_exit_layers < 0:
            raise ValueError("num_entry_layers and num_exit_layers must be >= 0")
        if self.rope_original_max_position is None:
            self.rope_original_max_position = self.max_seq_len

    @property
    def num_layers(self):
        return self.num_entry_layers + self.num_body_layers + self.num_exit_layers


class LogosTransformerBlock(nn.Module):
    """A single Logos parameter-block: SWA-attn + FFN or KDA+retrieval+FFN."""

    config: LogosConfig
    layer_idx: int
    num_loops: int = 1

    def setup(self):
        cfg = self.config
        self.is_swa = (self.layer_idx % cfg.swa_every) == cfg.swa_offset

        if self.is_swa:
            self.attn_norm = RMSNorm(eps=cfg.norm_eps)
            self.attn = LocalAttention(
                d_model=cfg.d_model,
                num_heads=cfg.num_heads,
                swa_window=cfg.swa_window,
                dropout=cfg.dropout,
                qk_norm=cfg.qk_norm,
                partial_rope_dim=cfg.partial_rope_dim,
                rope_base=cfg.rope_base,
                max_seq_len=cfg.max_seq_len,
                norm_eps=cfg.norm_eps,
                attention_sink=cfg.attention_sink,
            )
            self.attn_res = BlockAttentionResidual(
                d_model=cfg.d_model, eps=cfg.norm_eps,
            )
        else:
            self.kda_norm = RMSNorm(eps=cfg.norm_eps)
            self.kda = SuperKimiDeltaAttention(
                d_model=cfg.d_model,
                num_heads=cfg.num_heads,
                head_dim=cfg.head_dim,
                conv_size=cfg.conv_size,
                chunk_size=cfg.chunk_size,
                snapshot_interval=cfg.snapshot_interval,
                snapshot_latent_dim=cfg.snapshot_latent_dim,
                A_init_min=cfg.A_init_min,
                A_init_max=cfg.A_init_max,
                norm_eps=cfg.norm_eps,
            )
            self.kda_res = BlockAttentionResidual(
                d_model=cfg.d_model, eps=cfg.norm_eps,
            )

            self.mem_norm = RMSNorm(eps=cfg.norm_eps)
            self.mem = SnapshotRetrieval(
                d_model=cfg.d_model,
                num_heads=cfg.num_heads,
                mem_head_dim=cfg.mem_head_dim,
                latent_dim=cfg.snapshot_latent_dim,
                top_k=cfg.mem_top_k,
                mem_latent_dim=cfg.mem_latent_dim,
                rope_base=cfg.rope_base,
                rope_scaling_type=cfg.rope_scaling_type,
                rope_scaling_factor=cfg.rope_scaling_factor,
                rope_original_max_position=cfg.rope_original_max_position,
                yarn_beta_fast=cfg.yarn_beta_fast,
                yarn_beta_slow=cfg.yarn_beta_slow,
                partial_rope_dim=cfg.partial_rope_dim,
                attention_sink=cfg.attention_sink,
                norm_eps=cfg.norm_eps,
            )
            self.mem_res = BlockAttentionResidual(
                d_model=cfg.d_model, eps=cfg.norm_eps,
            )

        self.ffn_norm = RMSNorm(eps=cfg.norm_eps)
        if cfg.use_moe:
            self.ffn = MoELayer(
                d_model=cfg.d_model,
                num_shared_experts=cfg.num_shared_experts,
                num_sparse_experts=cfg.num_sparse_experts,
                top_k=cfg.top_k,
                expert_d_ff=cfg.expert_d_ff,
                dropout=cfg.dropout,
                bias_update_rate=cfg.bias_update_rate,
                capacity_factor=cfg.capacity_factor,
                num_loops=self.num_loops,
                diversity_factor=cfg.moe_diversity_factor,
            )
            self.use_moe = True
        else:
            self.ffn = SwiGLU(d_ff=cfg.d_ff, dropout=cfg.dropout)
            self.use_moe = False

        self.ffn_res = BlockAttentionResidual(
            d_model=cfg.d_model, eps=cfg.norm_eps,
        )

    def __call__(
        self,
        blocks: List[jnp.ndarray],
        partial: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray] = None,
        is_causal: bool = True,
        loop_idx: int = 0,
        deterministic: bool = True,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, Optional[jnp.ndarray]]:
        if self.is_swa:
            h = self.attn_res(blocks, partial)
            attn_out = self.attn(
                self.attn_norm(h),
                attention_mask=attention_mask,
                is_causal=is_causal,
                deterministic=deterministic,
            )
            partial = partial + attn_out
        else:
            h = self.kda_res(blocks, partial)
            kda_out, snapshots, snap_positions = self.kda(
                self.kda_norm(h),
                attention_mask=attention_mask,
                deterministic=deterministic,
            )
            partial = partial + kda_out

            h = self.mem_res(blocks, partial)
            mem_out = self.mem(
                self.mem_norm(h),
                snapshots,
                snap_positions,
                token_offset=0,
                attention_mask=attention_mask,
                deterministic=deterministic,
            )
            partial = partial + mem_out

        h = self.ffn_res(blocks, partial)
        ffn_normed = self.ffn_norm(h)
        if self.use_moe:
            ffn_out, aux_loss, topk_indices = self.ffn(
                ffn_normed, loop_idx=loop_idx, deterministic=deterministic,
            )
            partial = partial + ffn_out
            return partial, aux_loss, topk_indices
        else:
            partial = partial + self.ffn(ffn_normed, deterministic=deterministic)
            return partial, jnp.zeros((), dtype=partial.dtype), None


class LogosTransformer(nn.Module):
    """Flax Logos transformer: Entry → Body (looped) → Exit with tied LM head."""

    config: LogosConfig

    @nn.compact
    def __call__(
        self,
        input_ids: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray] = None,
        labels: Optional[jnp.ndarray] = None,
        is_causal: bool = True,
        deterministic: bool = True,
    ) -> Dict[str, Any]:
        cfg = self.config
        kinit = default_kernel_init()

        # Token embedding
        x = nn.Embed(
            cfg.vocab_size, cfg.d_model, embedding_init=kinit, name="token_emb",
        )(input_ids)
        x = nn.Dropout(rate=cfg.dropout)(x, deterministic=deterministic)

        aux_loss = jnp.zeros((), dtype=x.dtype)
        topk_indices_list = []

        zero_partial = lambda: jnp.zeros_like(x)
        blocks: List[jnp.ndarray] = [x]
        partial = zero_partial()

        # Entry
        for i in range(cfg.num_entry_layers):
            block = LogosTransformerBlock(
                config=cfg, layer_idx=i, num_loops=1, name=f"entry_{i}",
            )
            partial, l_aux, l_topk = block(
                blocks, partial,
                attention_mask=attention_mask,
                is_causal=is_causal,
                loop_idx=0,
                deterministic=deterministic,
            )
            aux_loss = aux_loss + l_aux
            topk_indices_list.append(l_topk)
        blocks = blocks + [partial]
        partial = zero_partial()

        # Body (looped)
        body_offset = cfg.num_entry_layers
        for loop_idx in range(cfg.num_loops):
            for i in range(cfg.num_body_layers):
                block = LogosTransformerBlock(
                    config=cfg, layer_idx=body_offset + i,
                    num_loops=cfg.num_loops, name=f"body_{i}",
                )
                partial, l_aux, l_topk = block(
                    blocks, partial,
                    attention_mask=attention_mask,
                    is_causal=is_causal,
                    loop_idx=loop_idx,
                    deterministic=deterministic,
                )
                aux_loss = aux_loss + l_aux
                topk_indices_list.append(l_topk)
            blocks = blocks + [partial]
            partial = zero_partial()

        # Exit
        exit_offset = body_offset + cfg.num_body_layers
        for i in range(cfg.num_exit_layers):
            block = LogosTransformerBlock(
                config=cfg, layer_idx=exit_offset + i,
                num_loops=1, name=f"exit_{i}",
            )
            partial, l_aux, l_topk = block(
                blocks, partial,
                attention_mask=attention_mask,
                is_causal=is_causal,
                loop_idx=0,
                deterministic=deterministic,
            )
            aux_loss = aux_loss + l_aux
            topk_indices_list.append(l_topk)

        h_main = BlockAttentionResidual(
            d_model=cfg.d_model, eps=cfg.norm_eps, name="final_res",
        )(blocks, partial)
        x = RMSNorm(eps=cfg.norm_eps, name="final_norm")(h_main)

        # Tied LM head: reuse token_emb weight
        emb_weight = self.variables["params"]["token_emb"]["embedding"]
        logits = x @ emb_weight.T

        chunk_sz = int(getattr(cfg, "lm_head_chunk_size", 0) or 0)
        loss = None
        if labels is not None:
            if chunk_sz > 0:
                loss = chunked_linear_cross_entropy(
                    x, emb_weight,  # emb_weight is (vocab, d_model) — correct
                    labels,
                    chunk_size=chunk_sz,
                    ignore_index=-100,
                )
            else:
                loss = standard_lm_cross_entropy(
                    logits, labels, ignore_index=-100,
                )

        return {
            "logits": logits,
            "loss": loss,
            "aux_loss": aux_loss if cfg.use_moe else None,
            "topk_indices": topk_indices_list if cfg.use_moe else None,
        }
