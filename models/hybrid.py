"""Hybrid attention modules: KDA, local SWA, CSA, and HCA.

CSA/HCA are DeepSeek-V4-style compressed global attentions. They compress
sequence-dimension KV entries with learned per-dimension pooling, then run
shared-KV MQA over the compressed entries. CSA uses light compression plus a
two-stage sparse recall path; HCA uses heavier compression plus dense global
recall. No snapshot memory path is used.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.lm_loss import (
    lm_cross_entropy_from_logits,
    token_superposition_attention_mask,
    token_superposition_embeddings,
)
from models.linear import LinearConfig, KimiDeltaAttention
from models.baseline import (
    RMSNorm,
    Attention,
    RotaryEmbedding,
    SwiGLU,
    MoELayer,
    combine_lm_and_aux_loss,
    init_moe_router_weights,
    count_parameters,
    model_summary,
)


_ATTN_TYPES = ("kda", "swa", "csa", "hca")


def normalize_attention_type(kind: str) -> str:
    kind = kind.strip().lower()
    aliases = {
        "linear": "kda",
        "local": "swa",
        "sliding": "swa",
        "sliding_window": "swa",
        "compressed_sparse": "csa",
        "compressed": "csa",
        "heavily_compressed": "hca",
        "global": "hca",
    }
    kind = aliases.get(kind, kind)
    if kind not in _ATTN_TYPES:
        raise ValueError(
            f"Unknown attention type {kind!r}; expected one of {_ATTN_TYPES}."
        )
    return kind


def parse_attention_pattern(pattern: Optional[str]) -> List[str]:
    if pattern is None:
        return []
    pattern = pattern.strip()
    if not pattern:
        return []
    for sep in (";", "|"):
        pattern = pattern.replace(sep, ",")
    return [
        normalize_attention_type(part)
        for part in pattern.split(",")
        if part.strip()
    ]


def expand_attention_pattern(
    pattern: Optional[str],
    length: int,
    *,
    default: str,
) -> List[str]:
    if length < 0:
        raise ValueError("length must be >= 0")
    values = parse_attention_pattern(pattern)
    if not values:
        values = [normalize_attention_type(default)]
    return [values[i % len(values)] for i in range(length)]


def _local_swa_kind(layer_idx: int, swa_every: int, swa_offset: int) -> str:
    return "swa" if (layer_idx % swa_every) == swa_offset else "kda"


def default_hybrid_attention_pattern(config: "HybridConfig", length: int) -> List[str]:
    return [
        _local_swa_kind(i, config.swa_every, config.swa_offset)
        for i in range(length)
    ]


@dataclass
class HybridConfig(LinearConfig):
    swa_window: int = 256
    swa_every: int = 4
    swa_offset: int = 3

    # Compressed global attention. CSA defaults to 4-token compression and
    # sparse top-k recall; HCA defaults to 128-token compression and dense
    # recall over all compressed entries.
    csa_compression: int = 4
    csa_top_k: int = 1024
    csa_indexer_heads: int = 4
    csa_indexer_dim: int = 32
    # Weight on the CSA indexer's attention-aligned KL loss. The indexer is a
    # separate sparse-recall selector whose top-k is non-differentiable, so it
    # receives zero gradient otherwise. The loss trains only indexer params
    # (trunk inputs are detached), so the weight is forgiving; 1.0 matches the
    # DeepSeek-V3.2 lightning-indexer recipe and is not decayed.
    csa_indexer_loss_weight: float = 1.0
    hca_compression: int = 128
    compressed_query_dim: Optional[int] = None
    compressed_head_dim: Optional[int] = None
    # Apply rotary position embedding to the CSA/HCA compressed-attention q/k
    # (and the CSA indexer q/k). Queries rotate at their true token position;
    # pooled keys rotate at a per-group representative position (the group's
    # last token). Default ON, following NSA/DSA evidence that the indexer and
    # compressed attention need positional geometry (a missing/mismatched
    # indexer RoPE was a published DSA bug). Off => bit-identical to no-RoPE.
    compressed_rope: bool = True

    # Comma/semicolon-separated pattern, e.g. "hca,csa,csa,swa".
    # If unset, Hybrid preserves the old structural KDA/SWA schedule.
    attn_pattern: Optional[str] = None

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
        if self.csa_compression < 1:
            raise ValueError("csa_compression must be >= 1")
        if self.csa_top_k < 1:
            raise ValueError("csa_top_k must be >= 1")
        if self.csa_indexer_heads < 1:
            raise ValueError("csa_indexer_heads must be >= 1")
        if self.csa_indexer_dim < 1:
            raise ValueError("csa_indexer_dim must be >= 1")
        if self.csa_indexer_loss_weight < 0:
            raise ValueError("csa_indexer_loss_weight must be >= 0")
        if self.hca_compression < 1:
            raise ValueError("hca_compression must be >= 1")
        if self.compressed_query_dim is not None and self.compressed_query_dim < 1:
            raise ValueError("compressed_query_dim must be >= 1")
        if self.compressed_head_dim is not None and self.compressed_head_dim < 1:
            raise ValueError("compressed_head_dim must be >= 1")
        parse_attention_pattern(self.attn_pattern)


try:
    from torch.nn.attention.flex_attention import (
        flex_attention as _flex_attention,
        create_block_mask as _create_block_mask,
    )
    _HAS_FLEX = True
    _flex_attention_fused = torch.compile(_flex_attention, dynamic=True)
except ImportError:
    _HAS_FLEX = False
    _flex_attention_fused = None
try:
    import torch._dynamo.config as _dynamo_config
    for _attr in ("recompile_limit", "cache_size_limit"):
        if hasattr(_dynamo_config, _attr):
            setattr(_dynamo_config, _attr, max(64, getattr(_dynamo_config, _attr)))
except Exception:
    pass


class LocalAttention(Attention):
    """Causal sliding-window MHA with a dense CPU fallback."""

    def __init__(self, config: HybridConfig):
        super().__init__(config)
        self.window = config.swa_window
        self._block_mask_cache: Dict[Tuple[int, bool, bool, str], Any] = {}

    def _build_mask(
        self,
        batch: int,
        seq_len: int,
        device: torch.device,
        attention_mask: Optional[torch.Tensor],
        is_causal: bool,
    ) -> Optional[torch.Tensor]:
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


class TokenCompressor(nn.Module):
    """Learned per-dimension compression over fixed-size token groups."""

    def __init__(self, d_model: int, head_dim: int, compression: int, overlap: bool):
        super().__init__()
        self.head_dim = head_dim
        self.compression = compression
        self.overlap = overlap
        self.kv_proj_a = nn.Linear(d_model, head_dim, bias=False)
        self.weight_proj_a = nn.Linear(d_model, head_dim, bias=False)
        self.pos_bias_a = nn.Parameter(torch.zeros(compression, head_dim))
        if overlap:
            self.kv_proj_b = nn.Linear(d_model, head_dim, bias=False)
            self.weight_proj_b = nn.Linear(d_model, head_dim, bias=False)
            self.pos_bias_b = nn.Parameter(torch.zeros(compression, head_dim))

    def _group(
        self,
        x: torch.Tensor,
        z: torch.Tensor,
        mask: Optional[torch.Tensor],
        pos_bias: torch.Tensor,
        pad_front: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, D = x.shape
        m = self.compression
        if mask is None:
            mask = torch.ones(B, T, device=x.device, dtype=torch.bool)
        if pad_front > 0:
            x = F.pad(x, (0, 0, pad_front, 0))
            z = F.pad(z, (0, 0, pad_front, 0), value=float("-inf"))
            mask = F.pad(mask, (pad_front, 0), value=0)
            T = x.size(1)
        pad = (m - T % m) % m
        if pad > 0:
            x = F.pad(x, (0, 0, 0, pad))
            z = F.pad(z, (0, 0, 0, pad), value=float("-inf"))
            mask = F.pad(mask, (0, pad), value=0)

        G = x.size(1) // m
        x_g = x.view(B, G, m, D)
        z_g = z.view(B, G, m, D) + pos_bias.view(1, 1, m, D)
        valid = mask.view(B, G, m).bool()
        z_g = z_g.masked_fill(~valid.unsqueeze(-1), float("-inf"))
        return x_g, z_g, valid

    def forward(
        self,
        hidden: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        kv_a = self.kv_proj_a(hidden)
        z_a = self.weight_proj_a(hidden)
        a_x, a_z, a_valid = self._group(
            kv_a, z_a, attention_mask, self.pos_bias_a,
        )

        if self.overlap:
            kv_b = self.kv_proj_b(hidden)
            z_b = self.weight_proj_b(hidden)
            b_x, b_z, b_valid = self._group(
                kv_b, z_b, attention_mask, self.pos_bias_b,
                pad_front=self.compression,
            )
            # b group 0 is pure left padding; b group i+1 overlaps a group i.
            b_x = b_x[:, :a_x.size(1)]
            b_z = b_z[:, :a_z.size(1)]
            b_valid = b_valid[:, :a_valid.size(1)]
            values = torch.cat([a_x, b_x], dim=2)
            logits = torch.cat([a_z, b_z], dim=2)
            valid = torch.cat([a_valid, b_valid], dim=2)
        else:
            values = a_x
            logits = a_z
            valid = a_valid

        logits = logits.masked_fill(~valid.unsqueeze(-1), float("-inf"))
        all_invalid = ~valid.any(dim=2, keepdim=True)
        logits = torch.where(
            all_invalid.unsqueeze(-1),
            torch.zeros_like(logits),
            logits,
        )
        weights = torch.softmax(logits.float(), dim=2).to(values.dtype)
        weights = torch.where(valid.unsqueeze(-1), weights, torch.zeros_like(weights))
        compressed = torch.sum(weights * values, dim=2)
        group_valid = valid.any(dim=2)
        return compressed, group_valid


class CompressedGlobalAttention(nn.Module):
    """Shared-KV compressed attention used for CSA and HCA."""

    def __init__(self, config: HybridConfig, *, mode: str):
        super().__init__()
        self.mode = mode
        self.d_model = config.d_model
        self.num_heads = config.num_heads
        self.head_dim = config.compressed_head_dim or config.head_dim
        self.query_dim = config.compressed_query_dim or self.head_dim

        if mode == "csa":
            self.compression = config.csa_compression
            self.top_k = config.csa_top_k
            self.sparse = True
            overlap = True
        elif mode == "hca":
            self.compression = config.hca_compression
            self.top_k = 0
            self.sparse = False
            overlap = False
        else:
            raise ValueError(f"Unknown compressed attention mode {mode!r}")

        self.q_down = nn.Linear(config.d_model, self.query_dim, bias=False)
        self.q_up = nn.Linear(self.query_dim, config.num_heads * self.head_dim, bias=False)
        self.q_norm = RMSNorm(self.head_dim, eps=config.norm_eps)
        self.kv_norm = RMSNorm(self.head_dim, eps=config.norm_eps)
        self.compressor = TokenCompressor(
            config.d_model, self.head_dim, self.compression, overlap=overlap,
        )

        # Partial RoPE for queries (rotated at true token positions) and pooled
        # keys (rotated at a per-group representative position). Mirrors the
        # baseline Attention partial-rope geometry; default rotates the full
        # compressed head dim. Even-dim required by the rotate-half layout.
        self.compressed_rope = config.compressed_rope
        if self.compressed_rope:
            rope_dim = config.partial_rope_dim
            if rope_dim is None or rope_dim > self.head_dim:
                rope_dim = self.head_dim
            rope_dim -= rope_dim % 2
            self.rope_dim = rope_dim
            self.rotary = RotaryEmbedding(
                rope_dim, config.max_seq_len, config.rope_base,
            )

        if self.sparse:
            self.indexer_q_down = nn.Linear(config.d_model, self.query_dim, bias=False)
            self.indexer_q_up = nn.Linear(
                self.query_dim,
                config.csa_indexer_heads * config.csa_indexer_dim,
                bias=False,
            )
            self.indexer_k_proj = nn.Linear(self.head_dim, config.csa_indexer_dim, bias=False)
            self.indexer_w = nn.Linear(config.d_model, config.csa_indexer_heads, bias=False)
            self.indexer_heads = config.csa_indexer_heads
            self.indexer_dim = config.csa_indexer_dim
            self.indexer_loss_weight = config.csa_indexer_loss_weight
            if self.compressed_rope:
                idx_rope_dim = min(self.rope_dim, self.indexer_dim)
                idx_rope_dim -= idx_rope_dim % 2
                self.indexer_rope_dim = idx_rope_dim
                if idx_rope_dim > 0:
                    self.indexer_rotary = RotaryEmbedding(
                        idx_rope_dim, config.max_seq_len, config.rope_base,
                    )

        self.out_proj = nn.Linear(config.num_heads * self.head_dim, config.d_model, bias=False)
        self.attention_sink = config.attention_sink
        if self.attention_sink:
            self.sink_logit = nn.Parameter(torch.zeros(config.num_heads))

    def _group_positions(self, G: int, T: int, device: torch.device) -> torch.Tensor:
        """Representative absolute position for each compressed group.

        Group ``g`` aggregates raw tokens spanning the a-window ``[g*c, g*c+c)``
        (and, when overlapping, the b-window to its left); its representative is
        the a-window's LAST token ``g*c+c-1``, clamped to the final real token
        ``T-1`` for the trailing partial group. This matches the causal boundary
        used in scoring, so a query at ``t`` sees only groups whose rep position
        ``<= t``. ``torch.arange`` keeps this torch.compile/XLA-safe.
        """
        g_idx = torch.arange(G, device=device, dtype=torch.long)
        pos = (g_idx + 1) * self.compression - 1
        return pos.clamp_max(T - 1)

    def _rope_partial(
        self, x: torch.Tensor, positions: torch.Tensor, rotary: nn.Module, rope_dim: int
    ) -> torch.Tensor:
        """Partial RoPE on the last ``rope_dim`` channels of ``x`` at ``positions``."""
        if rope_dim >= x.shape[-1]:
            return rotary.forward_at_positions(x, positions)
        no_rope = x[..., :-rope_dim]
        rope = rotary.forward_at_positions(x[..., -rope_dim:], positions)
        return torch.cat([no_rope, rope], dim=-1)

    def _build_scores(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        kv_index: torch.Tensor,
        group_valid: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        is_causal: bool,
        hidden: torch.Tensor,
        positions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        scores = torch.einsum("bhtd,bnd->bhtn", q, kv) / (self.head_dim ** 0.5)
        B, H, T, G = scores.shape
        index_loss = torch.zeros((), device=scores.device, dtype=scores.dtype)
        if is_causal:
            t_idx = torch.arange(T, device=hidden.device, dtype=torch.long)
            g_idx = torch.arange(G, device=hidden.device, dtype=torch.long)
            complete_groups = torch.div(
                t_idx + 1, self.compression, rounding_mode="floor",
            )
            causal = g_idx.view(1, 1, 1, G) < complete_groups.view(1, 1, T, 1)
            if T % self.compression:
                # Match the old clamped end-position behavior for the final
                # partial compression group without forming compression*g+c.
                causal = causal | (
                    (t_idx.view(1, 1, T, 1) == (T - 1))
                    & (g_idx.view(1, 1, 1, G) == (G - 1))
                )
            scores = scores.masked_fill(~causal, float("-inf"))
        scores = scores.masked_fill(~group_valid.view(B, 1, 1, G), float("-inf"))
        if attention_mask is not None:
            scores = scores.masked_fill(~attention_mask.view(B, 1, T, 1).bool(), float("-inf"))

        if self.sparse:
            # Per-query validity over groups (all heads masked => no valid key).
            # The top-k mask below is non-differentiable, so the indexer would
            # otherwise receive zero gradient. Feed it DETACHED trunk inputs so
            # its KL loss trains only the indexer params (DSA-style isolation);
            # this also makes the mask itself identical to the attached version.
            invalid = torch.isinf(scores).all(dim=1)  # [B, T, G]
            q_i = self.indexer_q_up(self.indexer_q_down(hidden.detach()))
            q_i = q_i.view(B, T, self.indexer_heads, self.indexer_dim)
            k_i = self.indexer_k_proj(kv_index.detach())
            if self.compressed_rope and self.indexer_rope_dim > 0:
                # Same positional geometry as the compressed attention so the
                # indexer ranks blocks the way the dense attention would: rotate
                # indexer-q at per-token positions, indexer-k at group positions.
                t_pos = torch.arange(T, device=q_i.device, dtype=torch.long)
                q_i = self._rope_partial(
                    q_i.transpose(1, 2), t_pos, self.indexer_rotary, self.indexer_rope_dim,
                ).transpose(1, 2)
                k_i = self._rope_partial(
                    k_i, positions, self.indexer_rotary, self.indexer_rope_dim,
                )
            idx_scores = torch.einsum("bthd,bnd->bthn", q_i, k_i)
            idx_scores = F.relu(idx_scores)
            idx_weights = self.indexer_w(hidden.detach()).transpose(1, 2).unsqueeze(-1)
            idx_scores = (idx_scores.transpose(1, 2) * idx_weights).sum(dim=1)
            idx_scores = idx_scores.masked_fill(invalid, float("-inf"))

            if self.training:
                index_loss = self._indexer_kl_loss(scores, idx_scores, invalid)

            k_sel = min(self.top_k, G)
            if k_sel < G:
                _, top_idx = idx_scores.topk(k_sel, dim=-1)
                keep = torch.zeros_like(idx_scores, dtype=torch.bool)
                keep.scatter_(-1, top_idx, True)
                scores = scores.masked_fill(~keep.unsqueeze(1), float("-inf"))
        return scores, index_loss

    def _indexer_kl_loss(
        self,
        scores: torch.Tensor,
        idx_scores: torch.Tensor,
        invalid: torch.Tensor,
    ) -> torch.Tensor:
        """KL(teacher || indexer) aligning the indexer to the dense attention.

        Teacher: per-head softmax over groups of the dense (pre-top-k) scores,
        summed across heads then L1-renormalized over groups, in fp32 and
        DETACHED. Student: log-softmax of the head-combined indexer scores over
        groups. Averaged over query rows that have at least one valid group.
        """
        B, H, T, G = scores.shape
        valid_row = (~invalid).any(dim=-1)  # [B, T]
        n_valid = valid_row.sum()
        if n_valid == 0:
            return torch.zeros((), device=scores.device, dtype=scores.dtype)

        # Teacher distribution over groups (detached). Fully-masked rows softmax
        # to NaN (all -inf); zero them out and exclude via valid_row below.
        per_head = F.softmax(scores.float(), dim=-1)  # [B, H, T, G]
        per_head = torch.nan_to_num(per_head, nan=0.0)
        target = per_head.sum(dim=1)  # [B, T, G]
        target = target / target.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        target = target.detach()

        student_logp = F.log_softmax(
            torch.where(invalid, torch.full_like(idx_scores, float("-inf")), idx_scores).float(),
            dim=-1,
        )
        student_logp = torch.nan_to_num(student_logp, neginf=0.0)

        kl = (target * (target.clamp_min(1e-9).log() - student_logp)).sum(dim=-1)  # [B, T]
        kl = kl * valid_row.to(kl.dtype)
        loss = kl.sum() / n_valid.to(kl.dtype)
        return loss.to(scores.dtype)

    def forward(
        self,
        hidden: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        is_causal: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, _ = hidden.shape
        kv, group_valid = self.compressor(hidden, attention_mask)
        kv = self.kv_norm(kv)
        q = self.q_up(self.q_down(hidden)).view(B, T, self.num_heads, self.head_dim)
        q = self.q_norm(q).transpose(1, 2)

        # RoPE (after q_norm/kv_norm, matching baseline order): queries rotate at
        # their true token position; pooled keys rotate at a per-group
        # representative position. The rotated kv is used ONLY for the score
        # dot-product; the value aggregation below keeps the un-rotated kv.
        positions = self._group_positions(kv.size(1), T, hidden.device)
        if self.compressed_rope:
            t_pos = torch.arange(T, device=q.device, dtype=torch.long)
            q_score = self._rope_partial(q, t_pos, self.rotary, self.rope_dim)
            kv_score = self._rope_partial(kv, positions, self.rotary, self.rope_dim)
        else:
            q_score = q
            kv_score = kv

        scores, index_loss = self._build_scores(
            q_score, kv_score, kv, group_valid, attention_mask, is_causal, hidden,
            positions,
        )
        all_masked = torch.isinf(scores).all(dim=-1, keepdim=True)
        if self.attention_sink:
            sink = self.sink_logit.float().view(1, self.num_heads, 1, 1).expand(B, -1, T, -1)
            aug = torch.cat([scores.float(), sink], dim=-1)
            weights = F.softmax(aug, dim=-1)[..., :scores.size(-1)].to(kv.dtype)
        else:
            safe_scores = torch.where(all_masked, torch.zeros_like(scores), scores)
            weights = F.softmax(safe_scores.float(), dim=-1).to(kv.dtype)
            weights = torch.where(all_masked, torch.zeros_like(weights), weights)
        out = torch.einsum("bhtn,bnd->bhtd", weights, kv)
        if attention_mask is not None:
            out = out * attention_mask.view(B, 1, T, 1).to(out.dtype)
        out = out.transpose(1, 2).reshape(B, T, self.num_heads * self.head_dim)
        out = self.out_proj(out)
        # Return the raw (unweighted) indexer KL so the model can both surface
        # it for logging and weight it once before adding to the train loss.
        return out, index_loss


class HybridAttentionLayer(nn.Module):
    """Owns the requested attention variants and selects one per call."""

    def __init__(self, config: HybridConfig, kinds: List[str]):
        super().__init__()
        unique = sorted(set(normalize_attention_type(k) for k in kinds))
        self.layers = nn.ModuleDict()
        for kind in unique:
            if kind == "kda":
                self.layers[kind] = KimiDeltaAttention(config)
            elif kind == "swa":
                self.layers[kind] = LocalAttention(config)
            elif kind in ("csa", "hca"):
                self.layers[kind] = CompressedGlobalAttention(config, mode=kind)

    def forward(
        self,
        kind: str,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        is_causal: bool = True,
        cache: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        kind = normalize_attention_type(kind)
        layer = self.layers[kind]
        zero = torch.zeros((), device=x.device, dtype=x.dtype)
        if kind == "kda":
            return layer(x, attention_mask=attention_mask, cache=cache), zero
        if kind in ("csa", "hca"):
            return layer(x, attention_mask=attention_mask, is_causal=is_causal)
        return layer(x, attention_mask=attention_mask, is_causal=is_causal), zero


class HybridTransformerBlock(nn.Module):
    def __init__(self, config: HybridConfig, attention_kinds: List[str]):
        super().__init__()
        self.use_moe = config.use_moe
        self.attention_kinds = [normalize_attention_type(k) for k in attention_kinds]
        self.attn_norm = RMSNorm(config.d_model, eps=config.norm_eps)
        self.attn = HybridAttentionLayer(config, self.attention_kinds)

        self.ffn_norm = RMSNorm(config.d_model, eps=config.norm_eps)
        if config.use_moe:
            self.ffn = MoELayer(config)
        else:
            self.ffn = SwiGLU(config.d_model, config.d_ff)

    def forward(
        self,
        x: torch.Tensor,
        attention_kind: str,
        attention_mask: Optional[torch.Tensor] = None,
        is_causal: bool = True,
        cache: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        attn_out, index_loss = self.attn(
            attention_kind,
            self.attn_norm(x),
            attention_mask=attention_mask,
            is_causal=is_causal,
            cache=cache,
        )
        x = x + attn_out

        if self.use_moe:
            ffn_out, aux_loss, topk_indices = self.ffn(self.ffn_norm(x))
            x = x + ffn_out
            return x, aux_loss, topk_indices, index_loss
        x = x + self.ffn(self.ffn_norm(x))
        zero = torch.zeros((), device=x.device, dtype=x.dtype)
        return x, zero, None, index_loss


class HybridTransformer(nn.Module):
    def __init__(self, config: HybridConfig):
        super().__init__()
        self.config = config

        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)

        if config.attn_pattern:
            self.attn_schedule = expand_attention_pattern(
                config.attn_pattern, config.num_layers, default="kda",
            )
        else:
            self.attn_schedule = default_hybrid_attention_pattern(
                config, config.num_layers,
            )

        self.layers = nn.ModuleList([
            HybridTransformerBlock(config, [self.attn_schedule[i]])
            for i in range(config.num_layers)
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
        init_moe_router_weights(self, self.config.router_init_std)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        is_causal: bool = True,
        caches: Optional[List[Dict[str, Any]]] = None,
        token_superposition_bag_size: int = 1,
    ) -> Dict[str, Any]:
        x = token_superposition_embeddings(
            self.token_emb, input_ids, token_superposition_bag_size,
        )
        attention_mask = token_superposition_attention_mask(
            attention_mask, token_superposition_bag_size,
        )

        aux_loss = torch.zeros((), device=input_ids.device, dtype=x.dtype)
        index_loss = torch.zeros((), device=input_ids.device, dtype=x.dtype)
        topk_indices_list: List[Optional[torch.Tensor]] = []
        for i, layer in enumerate(self.layers):
            kind = self.attn_schedule[i]
            layer_cache = caches[i] if (caches is not None and kind == "kda") else None
            x, layer_aux, layer_topk, layer_index = layer(
                x,
                attention_kind=kind,
                attention_mask=attention_mask,
                is_causal=is_causal,
                cache=layer_cache,
            )
            aux_loss = aux_loss + layer_aux
            index_loss = index_loss + layer_index
            topk_indices_list.append(layer_topk)

        x = self.final_norm(x)
        logits = self.lm_head(x)

        lm_loss: Optional[torch.Tensor] = None
        if labels is not None:
            lm_loss = lm_cross_entropy_from_logits(
                logits,
                labels,
                token_superposition_bag_size=token_superposition_bag_size,
                ignore_index=-100,
            )
        loss = combine_lm_and_aux_loss(
            lm_loss,
            aux_loss if self.config.use_moe else None,
            self.training,
        )
        if loss is not None and self.training:
            loss = loss + self.config.csa_indexer_loss_weight * index_loss

        return {
            "logits": logits,
            "loss": loss,
            "lm_loss": lm_loss,
            "aux_loss": aux_loss if self.config.use_moe else None,
            "indexer_loss": index_loss,
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
                kind = self.attn_schedule[idx]
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


__all__ = [
    "HybridConfig",
    "LocalAttention",
    "TokenCompressor",
    "CompressedGlobalAttention",
    "HybridAttentionLayer",
    "HybridTransformerBlock",
    "HybridTransformer",
    "normalize_attention_type",
    "parse_attention_pattern",
    "expand_attention_pattern",
    "count_parameters",
    "model_summary",
]
