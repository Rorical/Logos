"""KDA (Kimi Delta Attention) for Flax — chunkwise-parallel scan with snapshots."""

from __future__ import annotations

import functools
import math
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn

from models_jax.base import default_kernel_init


def _kda_gate(g: jnp.ndarray, A_log: jnp.ndarray, dt_bias: jnp.ndarray) -> jnp.ndarray:
    H, K = g.shape[-2], g.shape[-1]
    g = g.astype(jnp.float32) + dt_bias.astype(jnp.float32).reshape(H, K)
    dt = jax.nn.softplus(g)
    A = A_log.astype(jnp.float32).reshape(1, 1, H, 1)
    return -jnp.exp(A) * dt


def _causal_depthwise_conv1d(
    x: jnp.ndarray, weight: jnp.ndarray, activation: str = "silu",
) -> jnp.ndarray:
    B, T, D = x.shape
    K = weight.shape[-1]
    x_t = x.transpose(0, 2, 1)
    pad = K - 1
    x_pad = jnp.pad(x_t, ((0, 0), (0, 0), (pad, 0)))

    w_flat = weight.reshape(D, 1, K)

    y = jax.lax.conv_general_dilated(
        x_pad, w_flat,
        window_strides=(1,), padding="VALID",
        dimension_numbers=("NCH", "OIH", "NCH"),
        feature_group_count=D,
    )
    y = y.transpose(0, 2, 1)

    if activation == "silu":
        y = jax.nn.silu(y)
    return y


@functools.partial(jax.jit, static_argnames=(
    "chunk_size", "use_qk_l2norm", "do_snapshots",
    "chunks_per_snapshot", "compressor_norm_eps", "out_dtype",
))
def _kda_chunk_scan(
    q: jnp.ndarray, k: jnp.ndarray, v: jnp.ndarray,
    log_g: jnp.ndarray, beta: jnp.ndarray,
    chunk_size: int,
    use_qk_l2norm: bool = True,
    initial_state: Optional[jnp.ndarray] = None,
    do_snapshots: bool = False,
    chunks_per_snapshot: int = 1,
    compressor_norm_w=None, compressor_norm_eps=None, compressor_down_k=None,
    out_dtype=None,
):
    """Chunkwise-parallel KDA scan. Returns (output, final_state, snap_latents, snap_positions)."""
    B, T, H, K = q.shape
    V = v.shape[-1]
    orig_dtype = v.dtype if out_dtype is None else out_dtype

    if use_qk_l2norm:
        q = q / (jnp.linalg.norm(q, axis=-1, keepdims=True) + 1e-12)
        k = k / (jnp.linalg.norm(k, axis=-1, keepdims=True) + 1e-12)

    scale = K ** -0.5
    q = q.astype(jnp.float32) * scale
    k = k.astype(jnp.float32)
    v = v.astype(jnp.float32)
    log_g = log_g.astype(jnp.float32)
    beta = beta.astype(jnp.float32)

    pad = (chunk_size - T % chunk_size) % chunk_size
    if pad > 0:
        q = jnp.pad(q, ((0, 0), (0, pad), (0, 0), (0, 0)))
        k = jnp.pad(k, ((0, 0), (0, pad), (0, 0), (0, 0)))
        v = jnp.pad(v, ((0, 0), (0, pad), (0, 0), (0, 0)))
        log_g = jnp.pad(log_g, ((0, 0), (0, pad), (0, 0), (0, 0)))
        beta = jnp.pad(beta, ((0, 0), (0, pad), (0, 0)))

    Tp = T + pad
    Nc = Tp // chunk_size
    C = chunk_size

    q_c = q.reshape(B, Nc, C, H, K).transpose(0, 3, 1, 2, 4)
    k_c = k.reshape(B, Nc, C, H, K).transpose(0, 3, 1, 2, 4)
    v_c = v.reshape(B, Nc, C, H, V).transpose(0, 3, 1, 2, 4)
    log_g_c = log_g.reshape(B, Nc, C, H, K).transpose(0, 3, 1, 2, 4)
    beta_c = beta.reshape(B, Nc, C, H).transpose(0, 3, 1, 2)

    cum_log_g = jnp.cumsum(log_g_c, axis=3).clip(min=-15.0)
    W = jnp.exp(cum_log_g)
    W_inv = jnp.exp(-cum_log_g)

    u_mat = k_c * W_inv
    w_mat = k_c * W
    q_tilde = q_c * W

    beta_e = beta_c[..., None]
    beta_w = beta_e * w_mat
    beta_v = beta_e * v_c

    triu_mask = jnp.triu(jnp.ones((C, C), dtype=jnp.bool_), k=0)
    L = jnp.einsum("bhnik,bhnjk->bhnij", beta_w, u_mat)
    L = jnp.where(triu_mask, 0.0, L)

    I_plus_L = L + jnp.eye(C, dtype=L.dtype)
    effective_v = jnp.linalg.solve(I_plus_L, beta_v)
    effective_w = jnp.linalg.solve(I_plus_L, beta_w)

    strict_upper = jnp.triu(jnp.ones((C, C), dtype=jnp.bool_), k=1)
    intra_attn = jnp.einsum("bhnik,bhnjk->bhnij", q_tilde, u_mat)
    intra_attn = jnp.where(strict_upper, 0.0, intra_attn)

    if initial_state is not None:
        S0 = initial_state.astype(jnp.float32)
    else:
        S0 = jnp.zeros((B, H, K, V), dtype=jnp.float32)

    # Pre-allocate output and snapshot arrays
    outputs = jnp.zeros((Nc, B, H, C, V), dtype=jnp.float32)

    if do_snapshots:
        num_snaps = Nc // chunks_per_snapshot
        snap_latents = jnp.zeros((B, num_snaps, H, compressor_down_k.shape[-1]), dtype=orig_dtype)
    else:
        num_snaps = 0
        snap_latents = jnp.zeros((B, 0, H, 0), dtype=orig_dtype)

    def body_fn(n, carry):
        S, outputs, snap_latents = carry

        delta = effective_v[:, :, n] - jnp.einsum("bhkv,bhck->bhcv", S, effective_w[:, :, n])
        o_inter = jnp.einsum("bhkv,bhck->bhcv", S, q_tilde[:, :, n])
        o_chunk = o_inter + jnp.einsum("bhij,bhjv->bhiv", intra_attn[:, :, n], delta)

        outputs = outputs.at[n].set(o_chunk)

        state_update = jnp.einsum("bhck,bhcv->bhkv", u_mat[:, :, n], delta)
        S = W[:, :, n, -1, None] * (S + state_update)

        if do_snapshots:
            Bx, Hx, Kx, Vx = S.shape
            flat = S.reshape(Bx, Hx, Kx * Vx).astype(jnp.float32)
            rrms = jax.lax.rsqrt(
                jnp.mean(flat ** 2, axis=-1, keepdims=True) + compressor_norm_eps
            )
            flat = flat * rrms * compressor_norm_w.astype(jnp.float32)
            latent = (flat @ compressor_down_k.astype(jnp.float32)).astype(orig_dtype)

            # Always write somewhere; mask via jnp.where so non-snap iterations
            # write back the existing value (effective no-op). Avoids lax.cond
            # inside fori_loop, which has produced fragile HLO for scatter ops.
            is_snap_step = (n + 1) % chunks_per_snapshot == 0
            safe_snap_idx = jnp.maximum((n + 1) // chunks_per_snapshot - 1, 0)
            current = snap_latents[:, safe_snap_idx]
            new_val = jnp.where(is_snap_step, latent, current)
            snap_latents = snap_latents.at[:, safe_snap_idx].set(new_val)

        return (S, outputs, snap_latents)

    S, outputs, snap_latents = jax.lax.fori_loop(
        0, Nc, body_fn, (S0, outputs, snap_latents)
    )

    out = outputs.transpose(1, 0, 3, 2, 4).reshape(B, Tp, H, V)
    if pad > 0:
        out = out[:, :T]
    out = out.astype(orig_dtype)

    if do_snapshots and num_snaps > 0:
        snap_positions = jnp.arange(1, num_snaps + 1, dtype=jnp.int32) * chunks_per_snapshot * C - 1
    else:
        snap_latents = None
        snap_positions = None

    return out, S, snap_latents, snap_positions


class SuperKimiDeltaAttention(nn.Module):
    """KDA with snapshot compression (SuperLinear variant used in Logos)."""

    d_model: int
    num_heads: int
    head_dim: int = 64
    conv_size: int = 4
    chunk_size: int = 64
    snapshot_interval: int = 256
    snapshot_latent_dim: int = 128
    A_init_min: float = 1.0
    A_init_max: float = 16.0
    norm_eps: float = 1e-6

    def setup(self):
        PS = self.num_heads * self.head_dim
        HD = self.head_dim
        kinit = default_kernel_init()

        self.q_proj = nn.Dense(PS, use_bias=False, kernel_init=kinit)
        self.k_proj = nn.Dense(PS, use_bias=False, kernel_init=kinit)
        self.v_proj = nn.Dense(PS, use_bias=False, kernel_init=kinit)

        self.q_conv_w = self.param("q_conv_w", kinit, (PS, self.conv_size))
        self.k_conv_w = self.param("k_conv_w", kinit, (PS, self.conv_size))
        self.v_conv_w = self.param("v_conv_w", kinit, (PS, self.conv_size))

        self.A_log = self._param_A("A_log", self.num_heads)
        self.dt_bias = self.param(
            "dt_bias", lambda *_args: self._init_dt_bias(PS), (PS,),
        )

        self.f_a = nn.Dense(HD, use_bias=False, kernel_init=kinit)
        self.f_b = nn.Dense(PS, use_bias=False, kernel_init=kinit)
        self.b_proj = nn.Dense(self.num_heads, use_bias=False, kernel_init=kinit)
        self.g_a = nn.Dense(HD, use_bias=False, kernel_init=kinit)
        self.g_b = nn.Dense(PS, use_bias=True, kernel_init=kinit)

        self.o_norm_w = self.param("o_norm_w", nn.initializers.ones, (HD,))

        self.o_proj = nn.Dense(self.d_model, use_bias=False, kernel_init=kinit)

        flat_dim = HD * HD
        self.comp_norm_w = self.param(
            "comp_norm_w", nn.initializers.ones, (flat_dim,),
        )
        self.comp_down_k = self.param(
            "comp_down_k", kinit, (flat_dim, self.snapshot_latent_dim),
        )

    @property
    def chunks_per_snapshot(self):
        return self.snapshot_interval // self.chunk_size

    def _init_dt_bias(self, size):
        key = jax.random.PRNGKey(0)
        dt = jnp.exp(
            jax.random.uniform(key, (size,), dtype=jnp.float32)
            * (math.log(0.1) - math.log(0.001)) + math.log(0.001)
        )
        dt = jnp.clip(dt, min=1e-4)
        return dt + jnp.log(-jnp.expm1(-dt))

    def __call__(
        self, x: jnp.ndarray, *,
        attention_mask: Optional[jnp.ndarray] = None,
        deterministic: bool = True,
    ):
        B, T, D = x.shape
        H = self.num_heads
        HK = self.head_dim
        PS = H * HK

        q_in = self.q_proj(x)
        k_in = self.k_proj(x)
        v_in = self.v_proj(x)

        q = _causal_depthwise_conv1d(q_in, self.q_conv_w)
        k = _causal_depthwise_conv1d(k_in, self.k_conv_w)
        v = _causal_depthwise_conv1d(v_in, self.v_conv_w)

        q = q.reshape(B, T, H, HK)
        k = k.reshape(B, T, H, HK)
        v = v.reshape(B, T, H, HK)

        g_raw = self.f_b(self.f_a(x)).reshape(B, T, H, HK)
        log_g = _kda_gate(g_raw, self.A_log, self.dt_bias)
        beta = jax.nn.sigmoid(self.b_proj(x).astype(jnp.float32))

        if attention_mask is not None:
            m4 = attention_mask[..., None, None].astype(q.dtype)
            q = q * m4
            k = k * m4
            v = v * m4
            log_g = log_g * m4
            beta = beta * attention_mask[..., None].astype(beta.dtype)

        do_snaps = (self.chunks_per_snapshot > 0)

        o, _final_state, snaps, snap_positions = _kda_chunk_scan(
            q, k, v, log_g, beta,
            chunk_size=self.chunk_size,
            use_qk_l2norm=True,
            do_snapshots=do_snaps,
            chunks_per_snapshot=self.chunks_per_snapshot,
            compressor_norm_w=self.comp_norm_w,
            compressor_norm_eps=self.norm_eps,
            compressor_down_k=self.comp_down_k,
            out_dtype=x.dtype,
        )

        gate = self.g_b(self.g_a(x)).reshape(B, T, H, HK)
        x_f32 = o.astype(jnp.float32)
        rrms = jax.lax.rsqrt(
            jnp.mean(x_f32 ** 2, axis=-1, keepdims=True) + self.norm_eps
        )
        o = (x_f32 * rrms) * self.o_norm_w.astype(jnp.float32)
        o = o * jax.nn.sigmoid(gate.astype(jnp.float32))
        o = o.reshape(B, T, PS)
        o = self.o_proj(o.astype(x.dtype))

        return o, snaps, snap_positions

    # Override A_log param init
    def _param_A(self, name, num_heads, minval=1.0, maxval=16.0):
        return self.param(
            name,
            lambda *_args: jnp.log(
                jax.random.uniform(
                    jax.random.PRNGKey(0), (num_heads,),
                    dtype=jnp.float32, minval=minval, maxval=maxval,
                )
            ),
            (num_heads,),
        )
