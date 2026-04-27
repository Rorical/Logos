"""Hybrid: SuperLinear KDA + retrieval, with local sliding-window
attention layers placed every ``swa_every`` positions (Samba-style)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.superlinear import (
    SuperLinearConfig,
    SuperKimiDeltaAttention,
    SnapshotRetrieval,
)
from models.baseline import (
    RMSNorm,
    Attention,
    SwiGLU,
    MoELayer,
    count_parameters,
    model_summary,
)


@dataclass
class HybridConfig(SuperLinearConfig):
    swa_window: int = 256
    swa_every: int = 4
    swa_offset: int = 3

    def __post_init__(self):
        super().__post_init__()
        if self.swa_window < 1:
            raise ValueError("swa_window must be >= 1")
        if self.swa_every < 1:
            raise ValueError("swa_every must be >= 1")
        if not (0 <= self.swa_offset < self.swa_every):
            raise ValueError(
                f"swa_offset ({self.swa_offset}) must be in [0, swa_every={self.swa_every})"
            )


try:
    from torch.nn.attention.flex_attention import (
        flex_attention as _flex_attention,
        create_block_mask as _create_block_mask,
    )
    _HAS_FLEX = True
    # Without an explicit torch.compile wrapper FlexAttention falls back to an
    # eager implementation that materializes the full scores matrix and warns
    # on every call. Compiling once at import time gives the fused kernel and
    # caches recompiles per (shape, score_mod, block_mask) signature.
    _flex_attention_fused = torch.compile(_flex_attention, dynamic=False)
except ImportError:
    _HAS_FLEX = False
    _flex_attention_fused = None


class LocalAttention(Attention):
    """Causal sliding-window MHA.

    On CUDA we route through :func:`torch.nn.attention.flex_attention.flex_attention`
    with a compiled sliding-window ``BlockMask``, so attention is sparse:
    out-of-window key blocks are skipped entirely instead of being computed
    and then masked. This is roughly ``seq_len / window`` × cheaper than
    the dense-and-mask approach on long contexts.

    Attention-sink is preserved by prepending a single virtual key/value
    pair at position 0 (zero value vector) and using ``score_mod`` to
    override the pre-softmax score of that slot to the per-head learnable
    ``sink_logit``. This matches :func:`baseline.softmax_with_sink` exactly:
    the sink contributes mass to the softmax denominator while contributing
    zero to the output, biasing other weights to sum to ≤ 1.

    A dense fallback (parent's ``_build_mask`` + ``F.scaled_dot_product_attention``)
    runs on CPU and on PyTorch builds without FlexAttention so smoke tests
    work everywhere.
    """

    def __init__(self, config: HybridConfig):
        super().__init__(config)
        self.window = config.swa_window
        # BlockMask is shape-bound (Q_LEN, KV_LEN) but parameter-free, so we
        # can build once per (seq_len, has_sink, is_causal, device) and reuse.
        self._block_mask_cache: Dict[Tuple[int, bool, bool, str], Any] = {}

    def _build_mask(
        self,
        batch: int,
        seq_len: int,
        device: torch.device,
        attention_mask: Optional[torch.Tensor],
        is_causal: bool,
    ) -> Optional[torch.Tensor]:
        # Used only by the dense CPU fallback in ``forward``.
        idx = torch.arange(seq_len, device=device)
        rel = idx.unsqueeze(0) - idx.unsqueeze(1)
        if is_causal:
            window_mask = (rel <= 0) & (rel > -self.window)
        else:
            window_mask = rel.abs() < self.window
        mask = window_mask.unsqueeze(0).unsqueeze(0)

        if attention_mask is not None:
            key_mask = attention_mask.unsqueeze(1).unsqueeze(2).bool()
            key_mask = key_mask.expand(batch, 1, seq_len, seq_len)
            mask = mask & key_mask

        return mask

    def _get_block_mask(
        self,
        seq_len: int,
        has_sink: bool,
        is_causal: bool,
        device: torch.device,
    ):
        key = (seq_len, has_sink, is_causal, str(device))
        bm = self._block_mask_cache.get(key)
        if bm is not None:
            return bm

        window = self.window
        if has_sink:
            kv_len = seq_len + 1
            if is_causal:
                def mask_mod(b, h, q_idx, kv_idx):
                    is_sink = kv_idx == 0
                    real_kv = kv_idx - 1
                    in_window = (q_idx >= real_kv) & (q_idx - real_kv < window)
                    return is_sink | in_window
            else:
                def mask_mod(b, h, q_idx, kv_idx):
                    is_sink = kv_idx == 0
                    real_kv = kv_idx - 1
                    in_window = (real_kv - q_idx).abs() < window
                    return is_sink | in_window
        else:
            kv_len = seq_len
            if is_causal:
                def mask_mod(b, h, q_idx, kv_idx):
                    return (q_idx >= kv_idx) & (q_idx - kv_idx < window)
            else:
                def mask_mod(b, h, q_idx, kv_idx):
                    return (q_idx - kv_idx).abs() < window

        bm = _create_block_mask(
            mask_mod, B=None, H=None, Q_LEN=seq_len, KV_LEN=kv_len,
            device=device,
        )
        self._block_mask_cache[key] = bm
        return bm

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        is_causal: bool = True,
    ) -> torch.Tensor:
        # The optimized FlexAttention block mask is sequence-structural; it
        # does not include per-example padding. When a caller supplies an
        # attention_mask, use the dense parent path so padded keys are masked
        # correctly. The pretraining script passes ``None`` for its packed
        # fixed-length batches, preserving the sparse CUDA path there.
        if attention_mask is not None or not _HAS_FLEX or not x.is_cuda:
            return super().forward(x, attention_mask=attention_mask, is_causal=is_causal)

        batch, seq_len, _ = x.shape
        q = self.q_proj(x).view(batch, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch, seq_len, self.num_heads, self.head_dim)
        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        q = self._apply_rope(q, seq_len)
        k = self._apply_rope(k, seq_len)

        has_sink = self.attention_sink
        if has_sink:
            sink_k = torch.zeros(
                batch, self.num_heads, 1, self.head_dim,
                device=q.device, dtype=q.dtype,
            )
            sink_v = torch.zeros_like(sink_k)
            k = torch.cat([sink_k, k], dim=2)
            v = torch.cat([sink_v, v], dim=2)
            sink_logit = self.sink_logit

            def score_mod(score, b, h, q_idx, kv_idx):
                # The dot-product against the zero sink key is 0; replace
                # it with the per-head learnable logit so the sink only
                # affects the softmax denominator.
                sink = sink_logit[h].to(score.dtype)
                return torch.where(kv_idx == 0, sink, score)
        else:
            score_mod = None

        block_mask = self._get_block_mask(
            seq_len, has_sink=has_sink, is_causal=is_causal, device=q.device,
        )
        out = _flex_attention_fused(
            q, k, v, score_mod=score_mod, block_mask=block_mask,
        )

        out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
        return self.out_proj(out)


class HybridTransformerBlock(nn.Module):
    """Either a full SuperLinear block (KDA + retrieval) or an SWA + FFN block.

    SWA layers carry no snapshot compressor or retrieval sublayer.
    """

    def __init__(self, config: HybridConfig, layer_idx: int):
        super().__init__()
        self.use_moe = config.use_moe
        self.is_swa = self._is_swa_layer(layer_idx, config)

        if self.is_swa:
            self.attn_norm = RMSNorm(config.d_model, eps=config.norm_eps)
            self.attn = LocalAttention(config)
        else:
            self.kda_norm = RMSNorm(config.d_model, eps=config.norm_eps)
            self.kda = SuperKimiDeltaAttention(config)

            self.mem_norm = RMSNorm(config.d_model, eps=config.norm_eps)
            self.mem = SnapshotRetrieval(
                d_model=config.d_model,
                num_heads=config.num_heads,
                mem_head_dim=config.mem_head_dim,
                latent_dim=config.snapshot_latent_dim,
                top_k=config.mem_top_k,
                mem_latent_dim=config.mem_latent_dim,
                rope_base=config.rope_base,
                rope_scaling_type=config.rope_scaling_type,
                rope_scaling_factor=config.rope_scaling_factor,
                rope_original_max_position=config.rope_original_max_position,
                yarn_beta_fast=config.yarn_beta_fast,
                yarn_beta_slow=config.yarn_beta_slow,
                partial_rope_dim=config.partial_rope_dim,
                attention_sink=config.attention_sink,
                norm_eps=config.norm_eps,
            )

        self.ffn_norm = RMSNorm(config.d_model, eps=config.norm_eps)
        if config.use_moe:
            self.ffn = MoELayer(config)
        else:
            self.ffn = SwiGLU(config.d_model, config.d_ff, config.dropout)

    @staticmethod
    def _is_swa_layer(layer_idx: int, config: HybridConfig) -> bool:
        return (layer_idx % config.swa_every) == config.swa_offset

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        is_causal: bool = True,
        cache: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        if self.is_swa:
            attn_out = self.attn(
                self.attn_norm(x), attention_mask=attention_mask, is_causal=is_causal
            )
            x = x + attn_out
        else:
            kda_out, snapshots, snap_positions = self.kda(
                self.kda_norm(x), attention_mask=attention_mask, cache=cache
            )
            x = x + kda_out

            if cache is not None:
                token_offset = cache.get("n_processed", x.size(1)) - x.size(1)
            else:
                token_offset = 0
            mem_out = self.mem(
                self.mem_norm(x),
                snapshots,
                snap_positions,
                token_offset=token_offset,
                attention_mask=attention_mask,
            )
            x = x + mem_out

        if self.use_moe:
            ffn_out, aux_loss, topk_indices = self.ffn(self.ffn_norm(x))
            x = x + ffn_out
            return x, aux_loss, topk_indices
        else:
            x = x + self.ffn(self.ffn_norm(x))
            zero = torch.zeros((), device=x.device, dtype=x.dtype)
            return x, zero, None


class HybridTransformer(nn.Module):
    def __init__(self, config: HybridConfig):
        super().__init__()
        self.config = config

        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

        self.layers = nn.ModuleList([
            HybridTransformerBlock(config, i) for i in range(config.num_layers)
        ])

        self.final_norm = RMSNorm(config.d_model, eps=config.norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

        for module in self.modules():
            if isinstance(module, SnapshotRetrieval):
                nn.init.zeros_(module.out_up.weight)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        is_causal: bool = True,
        caches: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        x = self.token_emb(input_ids)
        x = self.dropout(x)

        aux_loss = torch.zeros((), device=input_ids.device, dtype=x.dtype)
        topk_indices_list: List[Optional[torch.Tensor]] = []
        for i, layer in enumerate(self.layers):
            # SWA layers don't read per-layer caches.
            layer_cache = (
                caches[i] if (caches is not None and not layer.is_swa) else None
            )
            x, layer_aux, layer_topk = layer(
                x,
                attention_mask=attention_mask,
                is_causal=is_causal,
                cache=layer_cache,
            )
            aux_loss = aux_loss + layer_aux
            topk_indices_list.append(layer_topk)

        x = self.final_norm(x)
        logits = self.lm_head(x)

        loss: Optional[torch.Tensor] = None
        if labels is not None:
            shift_logits = logits[..., :-1, :]
            shift_labels = labels[..., 1:]
            flat_labels = shift_labels.reshape(-1)
            loss_sum = F.cross_entropy(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                flat_labels,
                ignore_index=-100,
                reduction="sum",
            )
            loss = loss_sum / (flat_labels != -100).sum().clamp_min(1)

        return {
            "logits": logits,
            "loss": loss,
            "aux_loss": aux_loss if self.config.use_moe else None,
            "topk_indices": topk_indices_list if self.config.use_moe else None,
        }

    def update_router_biases(self, topk_indices_list: List[Optional[torch.Tensor]]) -> None:
        if not self.config.use_moe:
            return
        for layer, topk_indices in zip(self.layers, topk_indices_list):
            if topk_indices is not None and isinstance(layer.ffn, MoELayer):
                layer.ffn.update_bias(topk_indices)

    @torch.no_grad()
    def get_balance_stats(self) -> Dict[str, float]:
        if not self.config.use_moe:
            return {}
        stats: Dict[str, float] = {}
        for idx, layer in enumerate(self.layers):
            if hasattr(layer.ffn, "bias"):
                bias = layer.ffn.bias
                kind = "swa" if layer.is_swa else "kda"
                stats[f"layer{idx}_{kind}_bias_mean"] = bias.abs().mean().item()
                stats[f"layer{idx}_{kind}_bias_max"] = bias.abs().max().item()
        return stats

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        attention_mask: Optional[torch.Tensor] = None,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """Naive autoregressive generation (no per-step cache reuse)."""
        self.train(False)
        batch_size = input_ids.size(0)

        for _ in range(max_new_tokens):
            outputs = self.forward(
                input_ids, attention_mask=attention_mask, is_causal=True,
            )
            logits = outputs["logits"][:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits = logits.masked_fill(logits < v[:, [-1]], float("-inf"))

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=-1)

            if attention_mask is not None:
                attention_mask = torch.cat([
                    attention_mask,
                    torch.ones(
                        (batch_size, 1),
                        device=attention_mask.device,
                        dtype=attention_mask.dtype,
                    ),
                ], dim=-1)

            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

        return input_ids
