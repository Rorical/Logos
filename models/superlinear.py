"""SuperLinear: KDA + per-layer snapshot retrieval memory.

The KDA scan emits MLA-compressed state snapshots at regular intervals;
a sparse top-k attention sublayer retrieves them with RoPE on (q, k)
encoding the relative distance ``t - snap_position``. Memory branch is
zero-init no-op so training begins identical to a KDA-only model.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from models.linear import (
    LinearConfig,
    _ShortConvolution,
    _RMSNormGatedSigmoid,
    _kda_gate,
    _kda_recurrent_step,
)
from models.baseline import (
    RMSNorm,
    SwiGLU,
    MoELayer,
    count_parameters,
    model_summary,
)


@dataclass
class SuperLinearConfig(LinearConfig):
    snapshot_interval: int = 256
    snapshot_latent_dim: int = 128

    mem_top_k: int = 16
    mem_head_dim: int = 64
    mem_latent_dim: int = 128

    rope_scaling_type: str = "none"
    rope_scaling_factor: float = 1.0
    rope_original_max_position: Optional[int] = None
    yarn_beta_fast: float = 32.0
    yarn_beta_slow: float = 1.0

    def __post_init__(self):
        super().__post_init__()
        if self.snapshot_interval <= 0:
            raise ValueError("snapshot_interval must be positive")
        if self.snapshot_interval % self.chunk_size != 0:
            raise ValueError(
                f"snapshot_interval ({self.snapshot_interval}) must be a "
                f"multiple of chunk_size ({self.chunk_size})"
            )
        if self.mem_top_k < 1:
            raise ValueError("mem_top_k must be >= 1")
        if self.snapshot_latent_dim < 1:
            raise ValueError("snapshot_latent_dim must be >= 1")
        if self.mem_latent_dim < 1:
            raise ValueError("mem_latent_dim must be >= 1")
        if self.mem_head_dim % 2 != 0:
            raise ValueError(
                f"mem_head_dim ({self.mem_head_dim}) must be even (RoPE)."
            )
        if self.rope_scaling_type not in ("none", "ntk", "yarn"):
            raise ValueError(
                f"rope_scaling_type must be 'none', 'ntk', or 'yarn', "
                f"got {self.rope_scaling_type!r}"
            )
        if self.rope_scaling_factor <= 0:
            raise ValueError("rope_scaling_factor must be positive")
        if self.rope_original_max_position is None:
            self.rope_original_max_position = self.max_seq_len
        # NTK 'base * s**(d/(d-2))' divides by zero at d=2.
        if self.rope_scaling_type == "ntk" and self.rope_scaling_factor != 1.0:
            rope_dim = (
                self.partial_rope_dim
                if self.partial_rope_dim is not None
                else self.mem_head_dim
            )
            if rope_dim <= 2:
                raise ValueError(
                    f"NTK scaling requires rotated dim > 2 (got {rope_dim})."
                )


class _SnapshotCompressor(nn.Module):
    """MLA-style compressor of per-head KDA state ``(B, H, K, V)`` -> ``(B, H, r)``."""

    def __init__(self, head_dim: int, latent_dim: int, eps: float = 1e-6):
        super().__init__()
        self.head_dim = head_dim
        self.latent_dim = latent_dim
        flat_dim = head_dim * head_dim
        self.norm = RMSNorm(flat_dim, eps=eps)
        self.down = nn.Linear(flat_dim, latent_dim, bias=False)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        B, H, K, V = state.shape
        flat = state.reshape(B, H, K * V)
        # Cast to projection dtype: the scan runs autocast-disabled (fp32).
        w_dtype = self.down.weight.dtype
        return self.down(self.norm(flat.to(w_dtype)))


def _kda_chunk_scan_with_snapshots(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    log_g: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int = 64,
    use_qk_l2norm: bool = True,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    compressor: Optional[_SnapshotCompressor] = None,
    chunks_per_snapshot: Optional[int] = None,
    chunk_offset: int = 0,
):
    """KDA chunkwise scan that also emits compressed state snapshots at every
    ``snapshot_interval`` boundary (in the global token frame).

    ``chunk_offset`` is the count of chunks already processed before this
    call. Chunk ``n`` emits a snapshot iff ``(chunk_offset + n + 1) %
    chunks_per_snapshot == 0``. Exact alignment with global snapshot
    boundaries requires ``n_before % chunk_size == 0``.

    ``snap_end_positions`` are returned in the local frame; the caller adds
    the global token offset.
    """
    B, T, H, K = q.shape
    V = v.shape[-1]
    orig_dtype = v.dtype
    device = q.device

    do_snapshots = (compressor is not None) and (chunks_per_snapshot is not None)

    with torch.autocast(device_type=device.type, enabled=False):
        if use_qk_l2norm:
            q = F.normalize(q, dim=-1)
            k = F.normalize(k, dim=-1)
        scale = K ** -0.5

        q = q.float() * scale
        k = k.float()
        v = v.float()
        log_g = log_g.float()
        beta = beta.float()

        pad = (chunk_size - T % chunk_size) % chunk_size
        if pad > 0:
            q = F.pad(q, (0, 0, 0, 0, 0, pad))
            k = F.pad(k, (0, 0, 0, 0, 0, pad))
            v = F.pad(v, (0, 0, 0, 0, 0, pad))
            log_g = F.pad(log_g, (0, 0, 0, 0, 0, pad))
            beta = F.pad(beta, (0, 0, 0, pad))
        Nc = (T + pad) // chunk_size
        C = chunk_size

        q = rearrange(q, "b (n c) h k -> b h n c k", c=C)
        k = rearrange(k, "b (n c) h k -> b h n c k", c=C)
        v = rearrange(v, "b (n c) h v -> b h n c v", c=C)
        log_g = rearrange(log_g, "b (n c) h k -> b h n c k", c=C)
        beta = rearrange(beta, "b (n c) h -> b h n c", c=C)

        cum_log_g = log_g.cumsum(dim=-2).clamp(min=-15.0)
        W = cum_log_g.exp()
        W_inv = (-cum_log_g).exp()

        u_mat = k * W_inv
        w_mat = k * W
        q_tilde = q * W

        beta_e = beta.unsqueeze(-1)
        beta_w = beta_e * w_mat
        beta_v = beta_e * v

        L = torch.einsum("bhnik,bhnjk->bhnij", beta_w, u_mat)
        upper_incl_diag = torch.triu(
            torch.ones(C, C, dtype=torch.bool, device=device), diagonal=0
        )
        L = L.masked_fill(upper_incl_diag, 0)

        I_plus_L = L + torch.eye(C, dtype=L.dtype, device=device)
        effective_v = torch.linalg.solve_triangular(
            I_plus_L, beta_v, upper=False, unitriangular=True
        )
        effective_w = torch.linalg.solve_triangular(
            I_plus_L, beta_w, upper=False, unitriangular=True
        )

        intra_attn = torch.einsum("bhnik,bhnjk->bhnij", q_tilde, u_mat)
        strict_upper = torch.triu(
            torch.ones(C, C, dtype=torch.bool, device=device), diagonal=1
        )
        intra_attn = intra_attn.masked_fill(strict_upper, 0)

        if initial_state is not None:
            S = initial_state.to(dtype=q.dtype, device=q.device)
        else:
            S = q.new_zeros(B, H, K, V)

        outputs: List[torch.Tensor] = []
        snapshot_latents: List[torch.Tensor] = []
        snapshot_end_positions: List[int] = []

        for n in range(Nc):
            delta = effective_v[:, :, n] - effective_w[:, :, n] @ S
            o_inter = q_tilde[:, :, n] @ S
            o_chunk = o_inter + intra_attn[:, :, n] @ delta
            outputs.append(o_chunk)

            state_update = torch.einsum(
                "bhck,bhcv->bhkv", u_mat[:, :, n], delta
            )
            S = W[:, :, n, -1].unsqueeze(-1) * (S + state_update)

            if do_snapshots and ((chunk_offset + n + 1) % chunks_per_snapshot == 0):
                latent = compressor(S).to(orig_dtype)
                snapshot_latents.append(latent)
                snapshot_end_positions.append((n + 1) * C - 1)

        out = torch.stack(outputs, dim=2)
        out = rearrange(out, "b h n c v -> b (n c) h v")
        if pad > 0:
            out = out[:, :T]
        out = out.to(orig_dtype)

    # Phantom snapshots in the padded suffix (snap_position >= T) are kept
    # rather than filtered: the causal mask in SnapshotRetrieval hides them
    # from every real query, and filtering would introduce a data-dependent
    # shape that breaks torch.compile.
    if do_snapshots and snapshot_latents:
        snaps = torch.stack(snapshot_latents, dim=1)
        positions = torch.tensor(
            snapshot_end_positions, device=device, dtype=torch.long
        )
    else:
        snaps, positions = None, None

    if output_final_state:
        return out, S, snaps, positions
    return out, snaps, positions


def _compute_rope_setup(
    head_dim: int,
    rope_base: float,
    scaling_type: str,
    scaling_factor: float,
    original_max_position: int,
    yarn_beta_fast: float,
    yarn_beta_slow: float,
) -> Tuple[torch.Tensor, float]:
    """Return ``(inv_freq, attention_scaling)`` for the chosen RoPE variant.

    - ``none``: vanilla RoPE.
    - ``ntk``: ``base * factor**(d/(d-2))`` so high-freq pairs stay; low
      freqs implicitly interpolate.
    - ``yarn``: per-pair mix of extrapolation and interpolation with a
      linear ramp + attention temperature factor on cos/sin.
    """
    d = head_dim
    arange = torch.arange(0, d, 2, dtype=torch.float32)
    inv_freq_base = 1.0 / (rope_base ** (arange / d))

    if scaling_type == "none" or scaling_factor == 1.0:
        return inv_freq_base, 1.0

    if scaling_type == "ntk":
        if d <= 2:
            raise ValueError(
                f"NTK scaling requires rotated dim > 2 (got d={d})."
            )
        new_base = rope_base * (scaling_factor ** (d / (d - 2)))
        inv_freq = 1.0 / (new_base ** (arange / d))
        return inv_freq, 1.0

    log_base = math.log(rope_base)
    n_pairs = d // 2

    def find_correction_dim(num_rot: float) -> float:
        return (d * math.log(original_max_position / (num_rot * 2 * math.pi))) / (2 * log_base)

    # Clamp ``high`` to ``n_pairs - 1`` (pair space). Reference YaRN
    # implementations clamp to ``d - 1`` (channel space) — a unit error
    # that prevents the lowest-freq pairs from reaching full interpolation.
    low = max(math.floor(find_correction_dim(yarn_beta_fast)), 0)
    high = min(math.ceil(find_correction_dim(yarn_beta_slow)), n_pairs - 1)
    if low == high:
        high = low + 1

    idx = torch.arange(n_pairs, dtype=torch.float32)
    ramp = torch.clamp((idx - low) / (high - low), 0.0, 1.0)

    inv_freq_extrap = inv_freq_base
    inv_freq_interp = inv_freq_base / scaling_factor
    inv_freq = inv_freq_extrap * (1.0 - ramp) + inv_freq_interp * ramp

    attention_scaling = 0.1 * math.log(scaling_factor) + 1.0
    return inv_freq, attention_scaling


class SnapshotRetrieval(nn.Module):
    """Causal top-k attention over compressed state snapshots.

    K/V are MLA up-projections from the stored latent. Q rotated by token
    position, K rotated by snapshot end-position, so the dot product
    encodes ``t - snap_position`` directly. ``out_up`` is zero-initialised
    so the module starts as an exact no-op (LoRA-style).
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        mem_head_dim: int,
        latent_dim: int,
        top_k: int,
        mem_latent_dim: int = 128,
        rope_base: float = 10000.0,
        rope_scaling_type: str = "none",
        rope_scaling_factor: float = 1.0,
        rope_original_max_position: int = 2048,
        yarn_beta_fast: float = 32.0,
        yarn_beta_slow: float = 1.0,
        partial_rope_dim: Optional[int] = None,
        attention_sink: bool = True,
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        if mem_head_dim % 2 != 0:
            raise ValueError(f"mem_head_dim ({mem_head_dim}) must be even for RoPE")
        self.num_heads = num_heads
        self.mem_head_dim = mem_head_dim
        self.top_k = top_k

        rope_dim = partial_rope_dim if partial_rope_dim is not None else mem_head_dim
        if rope_dim > mem_head_dim:
            raise ValueError(
                f"partial_rope_dim ({rope_dim}) must be <= mem_head_dim ({mem_head_dim})"
            )
        if rope_dim % 2 != 0:
            raise ValueError(f"partial_rope_dim ({rope_dim}) must be even")
        self.rope_dim = rope_dim

        self.attention_sink = attention_sink

        projection_size = num_heads * mem_head_dim

        # MLA-style low-rank bottleneck on the three D^2-sized projections.
        self.q_down = nn.Linear(d_model, mem_latent_dim, bias=False)
        self.q_up = nn.Linear(mem_latent_dim, projection_size, bias=False)

        self.gate_down = nn.Linear(d_model, mem_latent_dim, bias=False)
        self.gate_up = nn.Linear(mem_latent_dim, projection_size, bias=False)

        self.out_down = nn.Linear(projection_size, mem_latent_dim, bias=False)
        self.out_up = nn.Linear(mem_latent_dim, d_model, bias=False)

        self.k_up = nn.Linear(latent_dim, mem_head_dim, bias=False)
        self.v_up = nn.Linear(latent_dim, mem_head_dim, bias=False)

        self.q_norm = RMSNorm(mem_head_dim, eps=norm_eps)
        self.k_norm = RMSNorm(mem_head_dim, eps=norm_eps)

        inv_freq, attention_scaling = _compute_rope_setup(
            head_dim=self.rope_dim,
            rope_base=rope_base,
            scaling_type=rope_scaling_type,
            scaling_factor=rope_scaling_factor,
            original_max_position=rope_original_max_position,
            yarn_beta_fast=yarn_beta_fast,
            yarn_beta_slow=yarn_beta_slow,
        )
        self.register_buffer("rope_inv_freq", inv_freq, persistent=False)
        self.rope_attention_scaling = float(attention_scaling)

        if self.attention_sink:
            self.sink_logit = nn.Parameter(torch.zeros(num_heads))

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def _apply_rope(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        if self.rope_dim < x.shape[-1]:
            no_rope = x[..., :-self.rope_dim]
            target = x[..., -self.rope_dim:]
        else:
            no_rope = None
            target = x

        freqs = positions.to(torch.float32).unsqueeze(-1) * self.rope_inv_freq
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().to(target.dtype).unsqueeze(0).unsqueeze(0)
        sin = emb.sin().to(target.dtype).unsqueeze(0).unsqueeze(0)
        rotated = target * cos + self._rotate_half(target) * sin
        if self.rope_attention_scaling != 1.0:
            rotated = rotated * self.rope_attention_scaling

        if no_rope is None:
            return rotated
        return torch.cat([no_rope, rotated], dim=-1)

    def forward(
        self,
        x: torch.Tensor,
        snapshots: Optional[torch.Tensor],
        snap_positions: Optional[torch.Tensor],
        token_offset: int = 0,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run retrieval. ``token_offset`` is the absolute position of
        ``x[:, 0]`` (0 during pure training, ``n_processed - T`` during
        cached generation)."""
        if snapshots is None or snapshots.size(1) == 0:
            return torch.zeros_like(x)

        B, T, D = x.shape
        _, N, H, r = snapshots.shape
        Dh = self.mem_head_dim

        q = self.q_up(self.q_down(x)).view(B, T, H, Dh)
        q = self.q_norm(q).transpose(1, 2)

        k = self.k_up(snapshots)
        v = self.v_up(snapshots)
        k = self.k_norm(k).transpose(1, 2)
        v = v.transpose(1, 2)

        t_abs = token_offset + torch.arange(T, device=x.device)
        q = self._apply_rope(q, t_abs)
        k = self._apply_rope(k, snap_positions)

        scores = torch.einsum("bhtd,bhnd->bhtn", q, k) / (Dh ** 0.5)

        # Causal: snapshot covering up to position p is visible at token t
        # iff p < t.
        valid = snap_positions.unsqueeze(0) < t_abs.unsqueeze(1)
        valid = valid.unsqueeze(0).unsqueeze(0)
        scores = scores.masked_fill(~valid, float("-inf"))

        k_sel = min(self.top_k, N)
        if k_sel < N:
            _, top_idx = scores.topk(k_sel, dim=-1)
            keep_mask = torch.zeros_like(scores, dtype=torch.bool)
            keep_mask.scatter_(-1, top_idx, True)
            scores = scores.masked_fill(~keep_mask, float("-inf"))

        # Sink absorbs all mass when every key is masked (early tokens), so
        # no all-masked guard needed.
        if self.attention_sink:
            scores_f = scores.float()
            sink_f = self.sink_logit.float().view(1, H, 1, 1).expand(B, H, T, 1)
            aug = torch.cat([scores_f, sink_f], dim=-1)
            weights = F.softmax(aug, dim=-1)[..., :N].to(v.dtype)
        else:
            all_masked = torch.isinf(scores).all(dim=-1, keepdim=True)
            scores = torch.where(all_masked, torch.zeros_like(scores), scores)
            weights = F.softmax(scores.float(), dim=-1).to(v.dtype)
            weights = torch.where(all_masked, torch.zeros_like(weights), weights)

        out = torch.einsum("bhtn,bhnd->bhtd", weights, v)
        # ``reshape`` skips the layout-fixing copy when the post-transpose
        # strides happen to permit a view; ``contiguous().view(...)`` always
        # copied. Compile can additionally fold the reshape into the
        # surrounding matmul fusion.
        out = out.transpose(1, 2).reshape(B, T, H * Dh)

        gate = torch.sigmoid(self.gate_up(self.gate_down(x)))
        out = out * gate

        out = self.out_up(self.out_down(out))

        if attention_mask is not None:
            out = out * attention_mask.unsqueeze(-1).to(out.dtype)
        return out


class SuperKimiDeltaAttention(nn.Module):
    """KDA layer that also emits compressed snapshots and (optionally)
    caches a growing snapshot history."""

    def __init__(self, config: SuperLinearConfig):
        super().__init__()
        self.hidden_size = config.d_model
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.head_k_dim = self.head_dim
        self.conv_size = config.conv_size
        self.chunk_size = config.chunk_size
        self.snapshot_interval = config.snapshot_interval
        self.chunks_per_snapshot = config.snapshot_interval // config.chunk_size

        projection_size = self.num_heads * self.head_dim

        self.q_proj = nn.Linear(self.hidden_size, projection_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, projection_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, projection_size, bias=False)

        self.q_conv1d = _ShortConvolution(projection_size, self.conv_size, "silu")
        self.k_conv1d = _ShortConvolution(projection_size, self.conv_size, "silu")
        self.v_conv1d = _ShortConvolution(projection_size, self.conv_size, "silu")

        A = torch.empty(self.num_heads, dtype=torch.float32).uniform_(
            *config.A_init_range
        )
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True

        self.dt_bias = nn.Parameter(torch.empty(projection_size, dtype=torch.float32))
        self.dt_bias._no_weight_decay = True

        self.f_a_proj = nn.Linear(self.hidden_size, self.head_dim, bias=False)
        self.f_b_proj = nn.Linear(self.head_dim, projection_size, bias=False)

        self.b_proj = nn.Linear(self.hidden_size, self.num_heads, bias=False)

        self.g_a_proj = nn.Linear(self.hidden_size, self.head_dim, bias=False)
        self.g_b_proj = nn.Linear(self.head_dim, projection_size, bias=True)

        self.o_norm = _RMSNormGatedSigmoid(self.head_dim, eps=config.norm_eps)
        self.o_proj = nn.Linear(projection_size, self.hidden_size, bias=False)

        self.snapshot_compressor = _SnapshotCompressor(
            head_dim=self.head_dim,
            latent_dim=config.snapshot_latent_dim,
            eps=config.norm_eps,
        )

        self._reset_parameters()

    def _reset_parameters(self):
        dt = torch.exp(
            torch.rand(self.num_heads * self.head_dim)
            * (math.log(0.1) - math.log(0.001))
            + math.log(0.001)
        )
        dt = torch.clamp(dt, min=1e-4)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_bias.copy_(inv_dt)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        cache: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        use_cache = cache is not None
        T = x.size(1)

        q_in = self.q_proj(x)
        k_in = self.k_proj(x)
        v_in = self.v_proj(x)
        if use_cache:
            q, cache["conv_state_q"] = self.q_conv1d(
                q_in, cache=cache.get("conv_state_q"), return_cache=True
            )
            k, cache["conv_state_k"] = self.k_conv1d(
                k_in, cache=cache.get("conv_state_k"), return_cache=True
            )
            v, cache["conv_state_v"] = self.v_conv1d(
                v_in, cache=cache.get("conv_state_v"), return_cache=True
            )
        else:
            q = self.q_conv1d(q_in)
            k = self.k_conv1d(k_in)
            v = self.v_conv1d(v_in)

        q = rearrange(q, "... (h d) -> ... h d", d=self.head_dim)
        k = rearrange(k, "... (h d) -> ... h d", d=self.head_dim)
        v = rearrange(v, "... (h d) -> ... h d", d=self.head_dim)

        g_raw = self.f_b_proj(self.f_a_proj(x))
        g_raw = rearrange(g_raw, "... (h d) -> ... h d", d=self.head_dim)
        log_g = _kda_gate(g_raw, self.A_log, self.dt_bias)

        beta = self.b_proj(x).float().sigmoid()

        if attention_mask is not None:
            mask_4d = attention_mask.unsqueeze(-1).unsqueeze(-1)
            q = q * mask_4d.to(q.dtype)
            k = k * mask_4d.to(k.dtype)
            v = v * mask_4d.to(v.dtype)
            log_g = log_g * mask_4d.to(log_g.dtype)
            beta = beta * attention_mask.unsqueeze(-1).to(beta.dtype)

        prev_state = cache.get("recurrent_state") if use_cache else None
        n_before = cache.get("n_processed", 0) if use_cache else 0

        if use_cache and prev_state is not None and T == 1:
            from models.linear import _kda_recurrent_step
            o, new_state = _kda_recurrent_step(
                q, k, v, log_g, beta, prev_state, use_qk_l2norm=True
            )
            new_snapshots: Optional[torch.Tensor] = None
            new_snap_positions: Optional[torch.Tensor] = None
            n_after = n_before + 1
            if n_after % self.snapshot_interval == 0:
                latent = self.snapshot_compressor(new_state).to(x.dtype)
                new_snapshots = latent.unsqueeze(1)
                new_snap_positions = torch.tensor(
                    [n_after - 1], device=x.device, dtype=torch.long
                )
        else:
            chunk_offset = n_before // self.chunk_size
            if use_cache and (n_before % self.chunk_size != 0):
                warnings.warn(
                    f"SuperKimiDeltaAttention: chunked cached prefill with "
                    f"n_before={n_before} not aligned to chunk_size="
                    f"{self.chunk_size}; snapshot boundaries will misalign "
                    f"by {n_before % self.chunk_size} tokens.",
                    stacklevel=2,
                )
            o, new_state, scan_snaps, scan_positions = _kda_chunk_scan_with_snapshots(
                q=q, k=k, v=v, log_g=log_g, beta=beta,
                chunk_size=self.chunk_size,
                use_qk_l2norm=True,
                initial_state=prev_state,
                output_final_state=True,
                compressor=self.snapshot_compressor,
                chunks_per_snapshot=self.chunks_per_snapshot,
                chunk_offset=chunk_offset,
            )
            new_snapshots = scan_snaps
            new_snap_positions = scan_positions
            n_after = n_before + T
            if new_snap_positions is not None:
                new_snap_positions = new_snap_positions + n_before

        if use_cache:
            prev_snaps = cache.get("mem_latents")
            prev_positions = cache.get("mem_positions")
            if new_snapshots is not None and prev_snaps is not None:
                all_snaps = torch.cat([prev_snaps, new_snapshots], dim=1)
                all_positions = torch.cat([prev_positions, new_snap_positions])
            elif new_snapshots is not None:
                all_snaps = new_snapshots
                all_positions = new_snap_positions
            else:
                all_snaps = prev_snaps
                all_positions = prev_positions

            cache["recurrent_state"] = new_state
            cache["mem_latents"] = all_snaps
            cache["mem_positions"] = all_positions
            cache["n_processed"] = n_after
            snapshots_out = all_snaps
            positions_out = all_positions
        else:
            snapshots_out = new_snapshots
            positions_out = new_snap_positions

        gate = self.g_b_proj(self.g_a_proj(x))
        gate = rearrange(gate, "... (h d) -> ... h d", d=self.head_dim)
        o = self.o_norm(o, gate)

        o = rearrange(o, "b t h d -> b t (h d)")
        return self.o_proj(o), snapshots_out, positions_out


class SuperLinearTransformerBlock(nn.Module):
    """KDA -> snapshot retrieval -> FFN/MoE."""

    def __init__(self, config: SuperLinearConfig):
        super().__init__()
        self.use_moe = config.use_moe

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

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        is_causal: bool = True,
        cache: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
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
            return x, torch.zeros((), device=x.device, dtype=x.dtype), None


class SuperLinearTransformer(nn.Module):
    def __init__(self, config: SuperLinearConfig):
        super().__init__()
        self.config = config

        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

        self.layers = nn.ModuleList([
            SuperLinearTransformerBlock(config) for _ in range(config.num_layers)
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

        # Zero ``out_up`` (LoRA-style): out_up @ out_down(x) = 0 at init, but
        # gradient still flows back so the memory path can start learning.
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
            layer_cache = caches[i] if caches is not None else None
            x, layer_aux, layer_topk = layer(
                x, attention_mask=attention_mask,
                is_causal=is_causal, cache=layer_cache,
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
        stats = {}
        for idx, layer in enumerate(self.layers):
            if hasattr(layer.ffn, "bias"):
                bias = layer.ffn.bias
                stats[f"layer{idx}_bias_mean"] = bias.abs().mean().item()
                stats[f"layer{idx}_bias_max"] = bias.abs().max().item()
        return stats

    def _make_caches(self) -> List[Dict[str, Any]]:
        return [
            {
                "recurrent_state": None,
                "conv_state_q": None,
                "conv_state_k": None,
                "conv_state_v": None,
                "mem_latents": None,
                "mem_positions": None,
                "n_processed": 0,
            }
            for _ in self.layers
        ]

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
        caches = self._make_caches()

        def _sample(logits: torch.Tensor) -> torch.Tensor:
            logits = logits / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits = logits.masked_fill(logits < v[:, [-1]], float("-inf"))
            probs = F.softmax(logits, dim=-1)
            return torch.multinomial(probs, num_samples=1)

        outputs = self.forward(input_ids, is_causal=True, caches=caches)
        next_token = _sample(outputs["logits"][:, -1, :])
        input_ids = torch.cat([input_ids, next_token], dim=-1)

        if eos_token_id is not None and (next_token == eos_token_id).all():
            return input_ids

        for _ in range(max_new_tokens - 1):
            outputs = self.forward(next_token, is_causal=True, caches=caches)
            next_token = _sample(outputs["logits"][:, -1, :])
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

        return input_ids
