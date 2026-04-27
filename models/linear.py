"""Linear (Kimi Delta Attention) decoder-only transformer.

Pure-PyTorch chunkwise-parallel KDA scan.
"""

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

from models.baseline import (
    BaselineConfig,
    RMSNorm,
    SwiGLU,
    MoELayer,
    _validate_moe_config,
    count_parameters,
    model_summary,
)


class _ShortConvolution(nn.Module):
    """Causal depthwise 1-D conv with optional cached state for O(1) decode."""

    def __init__(
        self,
        hidden_size: int,
        kernel_size: int,
        activation: str = "silu",
        bias: bool = False,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(
            hidden_size,
            hidden_size,
            kernel_size=kernel_size,
            groups=hidden_size,
            padding=kernel_size - 1,
            bias=bias,
        )
        self.activation = activation

    def forward(
        self,
        x: torch.Tensor,
        cache: Optional[torch.Tensor] = None,
        return_cache: bool = False,
    ):
        T = x.size(1)
        K = self.kernel_size

        if cache is None:
            y = self.conv(x.transpose(1, 2))[..., :T].transpose(1, 2)
        else:
            x_full = torch.cat([cache, x], dim=1)
            y = F.conv1d(
                x_full.transpose(1, 2),
                self.conv.weight,
                self.conv.bias,
                stride=1,
                padding=0,
                groups=self.conv.groups,
            ).transpose(1, 2)

        if self.activation == "silu":
            y = F.silu(y)

        if not return_cache:
            return y

        if K <= 1:
            new_cache = x.new_zeros(x.size(0), 0, x.size(-1))
        else:
            combined = torch.cat([cache, x], dim=1) if cache is not None else x
            if combined.size(1) >= K - 1:
                new_cache = combined[:, -(K - 1):].contiguous()
            else:
                pad = combined.new_zeros(
                    combined.size(0), (K - 1) - combined.size(1), combined.size(-1)
                )
                new_cache = torch.cat([pad, combined], dim=1)
        return y, new_cache


class _RMSNormGatedSigmoid(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x_f = x.float()
        rms_inv = x_f.pow(2).mean(dim=-1, keepdim=True).add_(self.eps).rsqrt()
        y = (x_f * rms_inv).to(dtype) * self.weight
        return y * torch.sigmoid(gate.to(dtype))


def _kda_gate(
    g: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
) -> torch.Tensor:
    """Log-space decay gate: ``-exp(A_log) * softplus(g + dt_bias)``."""
    H, K = g.shape[-2], g.shape[-1]
    g = g.float() + dt_bias.float().view(H, K)
    dt = F.softplus(g)
    A = A_log.float().view(1, 1, H, 1)
    return -A.exp() * dt


def _kda_chunk_scan(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    log_g: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int = 64,
    use_qk_l2norm: bool = True,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
):
    """Chunkwise-parallel KDA scan in pure PyTorch.

    Recurrence: ``S_i = (I - beta_i k_i k_i^T) D_i S_{i-1} + beta_i k_i v_i^T``,
    ``o_i = q_i @ S_i``, with ``D_i = diag(exp(log_g_i))``. Chunk-level
    parallelism comes from the similarity transform ``~S_i = W_i^{-1} S_i``
    plus a single triangular solve per chunk.
    """
    B, T, H, K = q.shape
    V = v.shape[-1]
    orig_dtype = v.dtype
    device = q.device

    # The body runs in fp32: per-channel decays accumulate aggressively and
    # CUDA's triangular_solve has no bf16/fp16 kernel.
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

        # Clamp the cumulative log-decay to [-15, 0]: at default A/dt_bias
        # ranges a 64-token cumsum can drop below -80, and exp(-cum) then
        # overflows fp32 and NaNs the triangular solve.
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
        for n in range(Nc):
            delta = effective_v[:, :, n] - effective_w[:, :, n] @ S
            o_inter = q_tilde[:, :, n] @ S
            o_chunk = o_inter + intra_attn[:, :, n] @ delta
            outputs.append(o_chunk)

            state_update = torch.einsum(
                "bhck,bhcv->bhkv", u_mat[:, :, n], delta
            )
            S = W[:, :, n, -1].unsqueeze(-1) * (S + state_update)

        out = torch.stack(outputs, dim=2)
        out = rearrange(out, "b h n c v -> b (n c) h v")
        if pad > 0:
            out = out[:, :T]
        out = out.to(orig_dtype)

    if output_final_state:
        # State stays fp32 so cached decode preserves precision.
        return out, S
    return out


def _kda_recurrent_step(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    log_g: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor,
    use_qk_l2norm: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Single-token KDA step matching ``_kda_chunk_scan`` for ``T == 1``."""
    assert q.size(1) == 1 and k.size(1) == 1 and v.size(1) == 1
    orig_dtype = v.dtype
    K = q.size(-1)
    device = q.device

    with torch.autocast(device_type=device.type, enabled=False):
        if use_qk_l2norm:
            q = F.normalize(q, dim=-1)
            k = F.normalize(k, dim=-1)
        scale = K ** -0.5

        q_t = (q[:, 0].float()) * scale
        k_t = k[:, 0].float()
        v_t = v[:, 0].float()
        g_t = log_g[:, 0].float()
        b_t = beta[:, 0].float()

        S = state.to(torch.float32)
        S = S * g_t.exp().unsqueeze(-1)
        kS = torch.einsum("bhk,bhkv->bhv", k_t, S)
        update = torch.einsum(
            "bhk,bhv->bhkv", (b_t.unsqueeze(-1) * k_t), (v_t - kS)
        )
        S = S + update
        o = torch.einsum("bhk,bhkv->bhv", q_t, S).unsqueeze(1)
    return o.to(orig_dtype), S


@dataclass
class LinearConfig(BaselineConfig):
    head_dim: int = 64
    conv_size: int = 4
    chunk_size: int = 64
    A_init_range: Tuple[float, float] = (1, 16)

    expand: int = 2
    rope_base: float = 10000.0

    def __post_init__(self):
        if self.d_model % self.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        if self.partial_rope_dim is not None:
            if self.partial_rope_dim % 2 != 0:
                raise ValueError(
                    f"partial_rope_dim ({self.partial_rope_dim}) must be even"
                )
        if self.head_dim < 1:
            raise ValueError("head_dim must be >= 1")
        if self.chunk_size < 1:
            raise ValueError("chunk_size must be >= 1")
        if self.conv_size < 1:
            raise ValueError("conv_size must be >= 1")
        _validate_moe_config(self)


class KimiDeltaAttention(nn.Module):
    def __init__(self, config: LinearConfig):
        super().__init__()
        self.hidden_size = config.d_model
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.head_k_dim = self.head_dim
        self.conv_size = config.conv_size
        self.chunk_size = config.chunk_size

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

        self._reset_parameters()

    def _reset_parameters(self):
        # Inverse-softplus init (Mamba-2 / KDA scheme).
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
        cache: Optional[Dict[str, Optional[torch.Tensor]]] = None,
    ) -> torch.Tensor:
        use_cache = cache is not None

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

        # Zero q/k/v, log_g, beta at padded positions so they contribute no
        # content and no decay to the recurrent state.
        if attention_mask is not None:
            mask_4d = attention_mask.unsqueeze(-1).unsqueeze(-1)
            q = q * mask_4d.to(q.dtype)
            k = k * mask_4d.to(k.dtype)
            v = v * mask_4d.to(v.dtype)
            log_g = log_g * mask_4d.to(log_g.dtype)
            beta = beta * attention_mask.unsqueeze(-1).to(beta.dtype)

        if use_cache:
            prev_state = cache.get("recurrent_state")
            if prev_state is not None and x.size(1) == 1:
                o, new_state = _kda_recurrent_step(
                    q, k, v, log_g, beta, prev_state, use_qk_l2norm=True
                )
            else:
                o, new_state = _kda_chunk_scan(
                    q=q, k=k, v=v, log_g=log_g, beta=beta,
                    chunk_size=self.chunk_size,
                    use_qk_l2norm=True,
                    initial_state=prev_state,
                    output_final_state=True,
                )
            cache["recurrent_state"] = new_state
        else:
            o = _kda_chunk_scan(
                q=q, k=k, v=v, log_g=log_g, beta=beta,
                chunk_size=self.chunk_size,
                use_qk_l2norm=True,
            )

        gate = self.g_b_proj(self.g_a_proj(x))
        gate = rearrange(gate, "... (h d) -> ... h d", d=self.head_dim)
        o = self.o_norm(o, gate)

        o = rearrange(o, "b t h d -> b t (h d)")
        return self.o_proj(o)


class LinearTransformerBlock(nn.Module):
    def __init__(self, config: LinearConfig):
        super().__init__()
        self.use_moe = config.use_moe

        self.kda_norm = RMSNorm(config.d_model, eps=config.norm_eps)
        self.kda = KimiDeltaAttention(config)

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
        cache: Optional[Dict[str, Optional[torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        x = x + self.kda(self.kda_norm(x), attention_mask=attention_mask, cache=cache)

        if self.use_moe:
            ffn_out, aux_loss, topk_indices = self.ffn(self.ffn_norm(x))
            x = x + ffn_out
            return x, aux_loss, topk_indices
        else:
            x = x + self.ffn(self.ffn_norm(x))
            return x, torch.zeros((), device=x.device, dtype=x.dtype), None


class LinearTransformer(nn.Module):
    def __init__(self, config: LinearConfig):
        super().__init__()
        self.config = config

        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

        self.layers = nn.ModuleList([
            LinearTransformerBlock(config) for _ in range(config.num_layers)
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
        caches: Optional[List[Dict[str, Optional[torch.Tensor]]]] = None,
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

        caches: List[Dict[str, Optional[torch.Tensor]]] = [
            {
                "recurrent_state": None,
                "conv_state_q": None,
                "conv_state_k": None,
                "conv_state_v": None,
            }
            for _ in self.layers
        ]

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
