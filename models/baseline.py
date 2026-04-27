"""Baseline decoder-only transformer with shared building blocks (RMSNorm,
SwiGLU, MoE, RoPE, sink softmax) reused across every other variant."""

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


def _maybe_all_reduce_load(load: torch.Tensor) -> torch.Tensor:
    """Sum a per-expert load tensor across DDP ranks in place. No-op when
    distributed is not initialised so single-GPU training is unchanged."""
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(load, op=dist.ReduceOp.SUM)
    return load


def _expert_load_from_topk(
    topk_indices: torch.Tensor,
    num_experts: int,
) -> torch.Tensor:
    """Count top-k expert assignments without materialising one-hot tensors."""
    return torch.bincount(
        topk_indices.reshape(-1),
        minlength=num_experts,
    ).to(torch.float32)


def _validate_moe_config(config) -> None:
    if not config.use_moe:
        return
    if config.num_shared_experts < 1:
        raise ValueError("num_shared_experts must be >= 1 when use_moe=True")
    if config.num_sparse_experts < 1:
        raise ValueError("num_sparse_experts must be >= 1 when use_moe=True")
    if not (1 <= config.top_k <= config.num_sparse_experts):
        raise ValueError(
            f"top_k ({config.top_k}) must be in [1, num_sparse_experts="
            f"{config.num_sparse_experts}] when use_moe=True"
        )
    if config.expert_d_ff < 1:
        raise ValueError("expert_d_ff must be >= 1 when use_moe=True")
    if config.capacity_factor <= 0:
        raise ValueError("capacity_factor must be > 0 when use_moe=True")


@dataclass
class BaselineConfig:
    vocab_size: int = 32000
    d_model: int = 512
    max_seq_len: int = 2048

    num_layers: int = 12
    num_heads: int = 8
    dropout: float = 0.0
    norm_eps: float = 1e-6

    # SwiGLU has 3 matmuls; ~(8/3) * d_model matches a 4*d_model 2-matmul FFN.
    d_ff: int = 1364

    use_moe: bool = True
    num_shared_experts: int = 2
    num_sparse_experts: int = 64
    top_k: int = 6
    expert_d_ff: int = 256
    bias_update_rate: float = 0.01
    capacity_factor: float = 2.0
    # 0 keeps the standard full-logits CE. Positive values enable the
    # memory-efficient chunked LM-head CE in models that support it.
    lm_head_chunk_size: int = 0
    # Cross-loop expert-diversity weight; only acts when an MoE layer is
    # reused across loop iterations (recursive / logos body stack).
    moe_diversity_factor: float = 0.0

    rope_base: float = 10000.0

    qk_norm: bool = True
    partial_rope_dim: Optional[int] = None
    attention_sink: bool = True

    # When True, route the BlockAttentionResidual depth-softmax + weighted
    # sum through an opaque torch.library.custom_op so torch.compile can't
    # fuse softmax_backward with the upstream stack / RMSNorm / dot-product
    # chain. Needed on SMEM-constrained GPUs (sm_120 / Ada-class consumer
    # cards, ~99 KB SMEM/SM) where Inductor's persistent-reduction fused
    # backward exceeds the per-block shared-memory cap. Adds ~one graph
    # break per BlockAttentionResidual call (~97 in Logos at default
    # depth) — typically <2% throughput on cards where the unfused path
    # compiles fine.
    block_residual_isolate_softmax: bool = False

    def __post_init__(self):
        if self.d_model % self.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        head_dim = self.d_model // self.num_heads
        if self.partial_rope_dim is not None:
            if self.partial_rope_dim > head_dim:
                raise ValueError(
                    f"partial_rope_dim ({self.partial_rope_dim}) must be "
                    f"<= head_dim ({head_dim})"
                )
            if self.partial_rope_dim % 2 != 0:
                raise ValueError(
                    f"partial_rope_dim ({self.partial_rope_dim}) must be even"
                )
        _validate_moe_config(self)


# Fused C++ kernel (PyTorch >= 2.4) — pow+mean+rsqrt+mul in a single pass,
# vs the previous 3-kernel python implementation. Same ``.weight`` parameter
# so existing checkpoints load unchanged. Note that nn.RMSNorm puts eps
# inside the sqrt (1/sqrt(mean(x^2)+eps)) where the old impl had it outside
# (1/(sqrt(mean(x^2))+eps)) — a tiny numerical difference, not a behavior
# change at training scale.
RMSNorm = nn.RMSNorm


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len

        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)

        self.register_buffer("cos", emb.cos().unsqueeze(0).unsqueeze(0), persistent=False)
        self.register_buffer("sin", emb.sin().unsqueeze(0).unsqueeze(0), persistent=False)

    @staticmethod
    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        # Fail loudly when slicing the cos/sin table would silently truncate;
        # see SnapshotRetrieval._apply_rope for a tableless variant.
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"RotaryEmbedding: seq_len ({seq_len}) exceeds the "
                f"precomputed max_seq_len ({self.max_seq_len})."
            )
        cos = self.cos[:, :, :seq_len, :].to(x.device)
        sin = self.sin[:, :, :seq_len, :].to(x.device)
        return x * cos + self.rotate_half(x) * sin


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w_down(F.silu(self.w_gate(x)) * self.w_up(x)))


class Expert(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.ffn = SwiGLU(d_model, d_ff, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(x)


class SparseExpertBank(nn.Module):
    """Packed sparse-expert SwiGLU weights.

    The master parameters stay 2D so the existing Muon parameter split still
    picks them up. ``packed_weights`` returns zero-copy 3D views grouped by
    expert.
    """

    def __init__(
        self,
        num_experts: int,
        d_model: int,
        d_ff: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.d_model = d_model
        self.d_ff = d_ff
        self.w_gate = nn.Parameter(torch.empty(num_experts * d_ff, d_model))
        self.w_up = nn.Parameter(torch.empty(num_experts * d_ff, d_model))
        self.w_down = nn.Parameter(torch.empty(num_experts * d_model, d_ff))
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.w_gate, mean=0.0, std=0.02)
        nn.init.normal_(self.w_up, mean=0.0, std=0.02)
        nn.init.normal_(self.w_down, mean=0.0, std=0.02)

    def packed_weights(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.w_gate.view(self.num_experts, self.d_ff, self.d_model),
            self.w_up.view(self.num_experts, self.d_ff, self.d_model),
            self.w_down.view(self.num_experts, self.d_model, self.d_ff),
        )

    def forward_batched(self, expert_in: torch.Tensor) -> torch.Tensor:
        """SwiGLU over a static-shape ``(E, C, d_model)`` expert-grouped batch.

        Three batched GEMMs replace ``E`` Python iterations of three small
        ``F.linear`` calls — fewer kernel launches and better SM utilisation
        when per-expert capacity ``C`` is small.
        """
        w_gate, w_up, w_down = self.packed_weights()
        h_gate = torch.bmm(expert_in, w_gate.transpose(-1, -2))
        h_up = torch.bmm(expert_in, w_up.transpose(-1, -2))
        hidden = F.silu(h_gate) * h_up
        return self.dropout(torch.bmm(hidden, w_down.transpose(-1, -2)))


class Router(nn.Module):
    def __init__(self, d_model: int, num_experts: int):
        super().__init__()
        self.linear = nn.Linear(d_model, num_experts, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class MoELayer(nn.Module):
    """Shared experts + top-k sparse experts with DeepSeek-style aux-loss-free
    bias balancing. Static-shape dispatch keeps it torch.compile-clean.

    ``num_loops`` > 1 gives the bias buffer a row per loop iteration so the
    same weights can specialise differently when reused across loops.
    """

    def __init__(self, config: BaselineConfig, num_loops: int = 1):
        super().__init__()
        self.d_model = config.d_model
        self.num_shared_experts = config.num_shared_experts
        self.num_sparse_experts = config.num_sparse_experts
        self.top_k = config.top_k
        self.bias_update_rate = config.bias_update_rate
        self.capacity_factor = config.capacity_factor
        self.num_loops = num_loops
        self.diversity_factor = float(getattr(config, "moe_diversity_factor", 0.0))

        self.router = Router(config.d_model, config.num_sparse_experts)

        self.register_buffer(
            "bias",
            torch.zeros(num_loops, config.num_sparse_experts),
            persistent=True,
        )

        self.shared_experts = nn.ModuleList([
            Expert(config.d_model, config.expert_d_ff, config.dropout)
            for _ in range(config.num_shared_experts)
        ])
        self.sparse_experts = SparseExpertBank(
            config.num_sparse_experts,
            config.d_model,
            config.expert_d_ff,
            config.dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        loop_idx: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, seq_len, d_model = x.shape
        N = batch * seq_len
        device = x.device
        dtype = x.dtype
        E = self.num_sparse_experts
        K = self.top_k

        router_logits = self.router(x) + self.bias[loop_idx]

        shared_out = sum(expert(x) for expert in self.shared_experts) / self.num_shared_experts

        capacity = max(1, int(N * K * self.capacity_factor / E))
        C = capacity

        x_flat = x.view(-1, d_model)

        router_probs = F.softmax(router_logits, dim=-1)

        topk_probs, topk_indices = torch.topk(router_probs, K, dim=-1)
        topk_probs = topk_probs / (
            topk_probs.sum(dim=-1, keepdim=True) + 1e-9
        )

        topk_indices_flat = topk_indices.view(-1)
        topk_probs_flat = topk_probs.view(-1)
        token_ids = torch.arange(
            N, device=device,
        ).unsqueeze(1).expand(-1, K).reshape(-1)

        sorted_expert_ids, sort_idx = torch.sort(topk_indices_flat)
        sorted_token_ids = token_ids[sort_idx]
        sorted_gates = topk_probs_flat[sort_idx]

        # Per-expert slot index via cummax over (position * is_first);
        # avoids the dynamic-shape ``nonzero`` that would graph-break
        # under compile.
        M = sorted_expert_ids.size(0)
        positions = torch.arange(M, device=device)
        diff = sorted_expert_ids[1:] != sorted_expert_ids[:-1]
        is_first = torch.cat(
            [torch.ones(1, dtype=torch.bool, device=device), diff]
        )
        group_starts = (positions * is_first.long()).cummax(dim=0).values
        slot_indices = positions - group_starts

        sorted_x = x_flat[sorted_token_ids]
        # Over-capacity tokens are routed to a sentinel slot C and trimmed.
        valid = slot_indices < C
        safe_slot = torch.where(
            valid, slot_indices, torch.full_like(slot_indices, C)
        )

        expert_in = torch.zeros(E, C + 1, d_model, device=device, dtype=dtype)
        expert_gate = torch.zeros(E, C + 1, device=device, dtype=dtype)
        expert_tok = torch.full(
            (E, C + 1), -1, dtype=torch.long, device=device
        )
        expert_mask = torch.zeros(E, C + 1, dtype=torch.bool, device=device)

        expert_in[sorted_expert_ids, safe_slot] = sorted_x
        expert_gate[sorted_expert_ids, safe_slot] = sorted_gates
        expert_tok[sorted_expert_ids, safe_slot] = sorted_token_ids
        expert_mask[sorted_expert_ids, safe_slot] = valid

        expert_in = expert_in[:, :C].contiguous()
        expert_gate = expert_gate[:, :C].contiguous()
        expert_tok = expert_tok[:, :C].contiguous()
        expert_mask = expert_mask[:, :C].contiguous()

        expert_out = self.sparse_experts.forward_batched(expert_in)

        # Static-shape scatter: invalid slots route to sentinel index N and
        # are trimmed off after the index_add_.
        flat_mask = expert_mask.view(-1)
        flat_tok = expert_tok.view(-1)
        flat_gate = expert_gate.view(-1)
        flat_src = expert_out.view(-1, d_model)

        safe_dst = torch.where(
            flat_mask, flat_tok, torch.full_like(flat_tok, N)
        )
        safe_gate = torch.where(
            flat_mask, flat_gate, torch.zeros_like(flat_gate)
        )

        sparse_out_ext = torch.zeros(
            N + 1, d_model, device=device, dtype=dtype
        )
        sparse_out_ext.index_add_(
            0, safe_dst, safe_gate.unsqueeze(-1) * flat_src
        )
        sparse_out = sparse_out_ext[:N].view(batch, seq_len, d_model)

        return shared_out + sparse_out, torch.tensor(
            0.0, device=device, dtype=dtype
        ), topk_indices

    def update_bias(self, topk_indices: torch.Tensor, loop_idx: int = 0) -> None:
        """Per-row balance update for one loop iteration. Call after
        ``optimizer.step()``."""
        with torch.no_grad():
            load = _expert_load_from_topk(
                topk_indices, self.num_sparse_experts
            )
            load = _maybe_all_reduce_load(load)
            total = load.sum() + 1e-9
            load_fraction = load / total
            target_fraction = 1.0 / self.num_sparse_experts
            self.bias[loop_idx] += self.bias_update_rate * (
                target_fraction - load_fraction
            )

    def update_bias_per_loop(
        self,
        topk_per_loop: List[torch.Tensor],
    ) -> None:
        """Combined balance + cross-loop diversity update.

        With ``diversity_factor > 0`` and ``num_loops > 1``, swaps per-row
        balance for an aggregate balance plus a diversity term that pushes
        each row away from experts the other loops over-use. The
        specialisation mode has growth coefficient ``+beta / (num_loops-1)``
        — unstable for any beta > 0, so any starting asymmetry amplifies.
        """
        if len(topk_per_loop) != self.num_loops:
            raise ValueError(
                f"update_bias_per_loop expected {self.num_loops} loop "
                f"entries, got {len(topk_per_loop)}"
            )
        with torch.no_grad():
            loads = torch.stack([
                _expert_load_from_topk(topk, self.num_sparse_experts)
                for topk in topk_per_loop
            ], dim=0)
            loads = _maybe_all_reduce_load(loads)
            loads = loads / (loads.sum(dim=1, keepdim=True) + 1e-9)

            target = 1.0 / self.num_sparse_experts

            if self.num_loops > 1 and self.diversity_factor > 0:
                agg_load = loads.mean(dim=0)
                agg_term = (target - agg_load).unsqueeze(0).expand_as(loads)
                other_mean = (loads.sum(dim=0, keepdim=True) - loads) / (
                    self.num_loops - 1
                )
                diversity_term = -self.diversity_factor * (other_mean - target)
                update = agg_term + diversity_term
            else:
                update = target - loads

            self.bias += self.bias_update_rate * update


def softmax_with_sink(scores: torch.Tensor, sink_logit: torch.Tensor) -> torch.Tensor:
    """Softmax with a per-head learnable sink logit appended to the
    denominator; weights sum to <= 1 (StreamingLLM / GPT-OSS-style)."""
    B, H, T_q, T_k = scores.shape
    out_dtype = scores.dtype
    sink = sink_logit.to(torch.float32).view(1, H, 1, 1).expand(B, H, T_q, 1)
    aug = torch.cat([scores.to(torch.float32), sink], dim=-1)
    weights = F.softmax(aug, dim=-1)[..., :T_k]
    return weights.to(out_dtype)


def manual_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    sink_logit: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    training: bool = False,
) -> torch.Tensor:
    D = q.shape[-1]
    scale = D ** -0.5
    scores = (q @ k.transpose(-2, -1)) * scale
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    if sink_logit is not None:
        weights = softmax_with_sink(scores, sink_logit)
    else:
        weights = F.softmax(scores, dim=-1)
    if training and dropout_p > 0.0:
        weights = F.dropout(weights, p=dropout_p)
    return weights @ v


class Attention(nn.Module):
    """Rotary MHA with optional Q/K RMSNorm, partial RoPE, and a per-head
    learnable attention sink. Falls back to SDPA when sink is disabled."""

    def __init__(self, config: BaselineConfig):
        super().__init__()
        self.d_model = config.d_model
        self.num_heads = config.num_heads
        self.head_dim = config.d_model // config.num_heads
        self.dropout = config.dropout

        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)

        self.qk_norm = config.qk_norm
        if self.qk_norm:
            self.q_norm = RMSNorm(self.head_dim, eps=config.norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=config.norm_eps)

        rope_dim = config.partial_rope_dim if config.partial_rope_dim is not None else self.head_dim
        self.rope_dim = rope_dim
        self.rotary = RotaryEmbedding(rope_dim, config.max_seq_len, config.rope_base)

        self.attention_sink = config.attention_sink
        if self.attention_sink:
            self.sink_logit = nn.Parameter(torch.zeros(self.num_heads))

    def _apply_rope(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        if self.rope_dim >= x.shape[-1]:
            return self.rotary(x, seq_len)
        no_rope = x[..., :-self.rope_dim]
        rope = x[..., -self.rope_dim:]
        rope = self.rotary(rope, seq_len)
        return torch.cat([no_rope, rope], dim=-1)

    def _build_mask(
        self,
        batch: int,
        seq_len: int,
        device: torch.device,
        attention_mask: Optional[torch.Tensor],
        is_causal: bool,
    ) -> Optional[torch.Tensor]:
        if attention_mask is not None:
            key_mask = attention_mask.unsqueeze(1).unsqueeze(2).bool()
            key_mask = key_mask.expand(batch, 1, seq_len, seq_len)
            if is_causal:
                causal_mask = torch.tril(
                    torch.ones(seq_len, seq_len, device=device, dtype=torch.bool)
                )
                return key_mask & causal_mask.unsqueeze(0).unsqueeze(0)
            return key_mask
        if is_causal and self.attention_sink:
            # The manual sink path needs an explicit causal mask.
            causal_mask = torch.tril(
                torch.ones(seq_len, seq_len, device=device, dtype=torch.bool)
            )
            return causal_mask.unsqueeze(0).unsqueeze(0)
        return None

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        is_causal: bool = True,
    ) -> torch.Tensor:
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

        mask = self._build_mask(batch, seq_len, x.device, attention_mask, is_causal)

        if self.attention_sink:
            out = manual_attention(
                q, k, v,
                mask=mask,
                sink_logit=self.sink_logit,
                dropout_p=self.dropout,
                training=self.training,
            )
        else:
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=(mask is None and is_causal),
            )

        out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    """Pre-norm Transformer block: RMSNorm -> Attn -> RMSNorm -> FFN/MoE.

    ``num_loops`` is forwarded to the optional MoE layer for weight-shared
    body stacks (recursive / logos).
    """

    def __init__(self, config: BaselineConfig, num_loops: int = 1):
        super().__init__()
        self.use_moe = config.use_moe

        self.attn_norm = RMSNorm(config.d_model, eps=config.norm_eps)
        self.attn = Attention(config)

        self.ffn_norm = RMSNorm(config.d_model, eps=config.norm_eps)
        if config.use_moe:
            self.ffn = MoELayer(config, num_loops=num_loops)
        else:
            self.ffn = SwiGLU(config.d_model, config.d_ff, config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        is_causal: bool = True,
        loop_idx: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        x = x + self.attn(self.attn_norm(x), attention_mask=attention_mask, is_causal=is_causal)

        if self.use_moe:
            ffn_out, aux_loss, topk_indices = self.ffn(self.ffn_norm(x), loop_idx=loop_idx)
            x = x + ffn_out
            return x, aux_loss, topk_indices
        else:
            x = x + self.ffn(self.ffn_norm(x))
            return x, torch.zeros((), device=x.device, dtype=x.dtype), None


class BaselineTransformer(nn.Module):
    def __init__(self, config: BaselineConfig):
        super().__init__()
        self.config = config

        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

        self.layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.num_layers)
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

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        is_causal: bool = True,
    ) -> Dict[str, Any]:
        x = self.token_emb(input_ids)
        x = self.dropout(x)

        aux_loss = torch.zeros((), device=input_ids.device, dtype=x.dtype)
        topk_indices_list: List[Optional[torch.Tensor]] = []
        for layer in self.layers:
            x, layer_aux, layer_topk = layer(x, attention_mask=attention_mask, is_causal=is_causal)
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
        stats = {}
        for idx, layer in enumerate(self.layers):
            if hasattr(layer.ffn, "bias"):
                bias = layer.ffn.bias
                stats[f"layer{idx}_bias_mean"] = bias.abs().mean().item()
                stats[f"layer{idx}_bias_max"] = bias.abs().max().item()
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
        self.eval()
        batch_size = input_ids.size(0)

        for _ in range(max_new_tokens):
            outputs = self.forward(
                input_ids,
                attention_mask=attention_mask,
                is_causal=True,
            )
            logits = outputs["logits"][:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            input_ids = torch.cat([input_ids, next_token], dim=-1)

            if attention_mask is not None:
                attention_mask = torch.cat([
                    attention_mask,
                    torch.ones((batch_size, 1), device=attention_mask.device, dtype=attention_mask.dtype),
                ], dim=-1)

            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

        return input_ids


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_summary(model: nn.Module) -> str:
    lines = ["Model Summary", "=" * 50]
    total = 0
    for name, module in model.named_children():
        n = sum(p.numel() for p in module.parameters())
        total += n
        lines.append(f"{name:25s} {n:>15,} params")
    lines.append("-" * 50)
    lines.append(f"{'Total':25s} {total:>15,} params")
    return "\n".join(lines)
