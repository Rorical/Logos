"""Decoder-only transformer with Block Attention Residuals (MoonshotAI):
each sublayer's input is a learned softmax over completed-block
representations plus the running partial sum, replacing the standard
additive residual."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.baseline import (
    BaselineConfig,
    RMSNorm,
    Attention,
    SwiGLU,
    MoELayer,
    count_parameters,
    model_summary,
)


@dataclass
class ResidualConfig(BaselineConfig):
    num_blocks: int = 4

    def __post_init__(self):
        super().__post_init__()
        if self.num_blocks < 1:
            raise ValueError("num_blocks must be >= 1")
        if self.num_blocks > self.num_layers:
            raise ValueError("num_blocks must be <= num_layers")
        if self.num_layers % self.num_blocks != 0:
            # Otherwise the last block silently gets the leftover layers and
            # the depth-wise softmax's candidates become asymmetric.
            raise ValueError(
                f"num_layers ({self.num_layers}) must be divisible by "
                f"num_blocks ({self.num_blocks})."
            )


def _block_residual_softmax_sum(
    values: torch.Tensor,
    logits: torch.Tensor,
) -> torch.Tensor:
    """``softmax(logits, dim=0) * values`` summed along the depth axis."""
    weights = torch.softmax(logits.float(), dim=0).to(values.dtype)
    return torch.sum(weights * values, dim=0)


# Opaque op so ``torch.compile`` doesn't fuse softmax_backward with the
# upstream stack / RMSNorm / dot-product chain. On Ada-class consumer
# GPUs (RTX PRO 6000 Blackwell, sm_120; ~99 KB SMEM/SM) Inductor
# materialises the persistent-reduction fused backward at >128 KB SMEM
# and fails to compile. Routing through a custom_op gives Dynamo a
# fusion boundary so each piece (softmax_backward, weighted_sum_backward,
# RMSNorm_backward) compiles into its own kernel that fits in SMEM.
@torch.library.custom_op(
    "logos::block_residual_softmax_sum", mutates_args=(),
)
def _block_residual_softmax_sum_op(
    values: torch.Tensor, logits: torch.Tensor,
) -> torch.Tensor:
    return _block_residual_softmax_sum(values, logits)


@_block_residual_softmax_sum_op.register_fake
def _block_residual_softmax_sum_fake(
    values: torch.Tensor, logits: torch.Tensor,
) -> torch.Tensor:
    return torch.empty(
        values.shape[1:], dtype=values.dtype, device=values.device,
    )


def _block_residual_softmax_sum_setup_context(ctx, inputs, output):
    values, logits = inputs
    ctx.save_for_backward(values, logits)


def _block_residual_softmax_sum_backward(ctx, grad_output):
    values, logits = ctx.saved_tensors
    # Recompute the small forward under autograd to extract per-input
    # gradients. Each contributing op (softmax, mul, sum) becomes its own
    # eager kernel — the whole point of routing through this opaque op.
    v_d = values.detach().requires_grad_(values.requires_grad)
    l_d = logits.detach().requires_grad_(logits.requires_grad)
    with torch.enable_grad():
        out = _block_residual_softmax_sum(v_d, l_d)
    grads = torch.autograd.grad(
        outputs=out, inputs=[v_d, l_d],
        grad_outputs=grad_output, allow_unused=True,
    )
    return grads[0], grads[1]


_block_residual_softmax_sum_op.register_autograd(
    _block_residual_softmax_sum_backward,
    setup_context=_block_residual_softmax_sum_setup_context,
)


class BlockAttentionResidual(nn.Module):
    """Depth-wise softmax over completed block states + the current partial.

    ``proj`` is zero-initialised so the layer starts as a uniform average,
    matching a standard residual sum in expectation. Softmax runs in fp32.

    ``isolate_softmax`` routes the depth-softmax + weighted-sum step
    through an opaque custom_op so ``torch.compile`` can't fuse it with
    the upstream stack / RMSNorm / dot-product chain. Set this on
    SMEM-constrained GPUs where Inductor's persistent-reduction fused
    backward exceeds the per-block shared-memory cap.
    """

    def __init__(
        self,
        d_model: int,
        eps: float = 1e-6,
        isolate_softmax: bool = False,
    ):
        super().__init__()
        self.norm = RMSNorm(d_model, eps=eps)
        # 1D Parameter rather than ``nn.Linear(d_model, 1, bias=False)``.
        # torch.compile's captured-graph backward strips the leading 1 dim
        # off the gradient of a ``(1, d_model)`` Linear weight, then fails
        # autograd's shape contract:
        #   "got [d_model] but expected shape compatible with [1, d_model]".
        # A 1D parameter sidesteps the bug — gradient and expected shape
        # are both ``(d_model,)`` with no singleton to mishandle.
        self.proj = nn.Parameter(torch.zeros(d_model))
        self.isolate_softmax = bool(isolate_softmax)

    def forward(
        self,
        blocks: List[torch.Tensor],
        partial_block: torch.Tensor,
    ) -> torch.Tensor:
        if len(blocks) == 0:
            return partial_block

        values = torch.stack(blocks + [partial_block], dim=0)
        keys = self.norm(values)
        # Dot keys against a 1D weight; keepdim=True so the broadcast
        # against ``values`` doesn't need a separate unsqueeze.
        logits = (keys * self.proj).sum(dim=-1, keepdim=True)  # [N, B, T, 1]
        if self.isolate_softmax:
            return _block_residual_softmax_sum_op(values, logits)
        return _block_residual_softmax_sum(values, logits)


class ResidualTransformerBlock(nn.Module):
    def __init__(self, config: ResidualConfig):
        super().__init__()
        self.use_moe = config.use_moe

        self.attn_norm = RMSNorm(config.d_model, eps=config.norm_eps)
        self.attn = Attention(config)

        self.ffn_norm = RMSNorm(config.d_model, eps=config.norm_eps)
        if config.use_moe:
            self.ffn = MoELayer(config)
        else:
            self.ffn = SwiGLU(config.d_model, config.d_ff, config.dropout)

        isolate = getattr(config, "block_residual_isolate_softmax", False)
        self.attn_res = BlockAttentionResidual(
            config.d_model, eps=config.norm_eps, isolate_softmax=isolate,
        )
        self.ffn_res = BlockAttentionResidual(
            config.d_model, eps=config.norm_eps, isolate_softmax=isolate,
        )

    def forward(
        self,
        blocks: List[torch.Tensor],
        partial_block: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        is_causal: bool = True,
    ) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        h = self.attn_res(blocks, partial_block)
        attn_out = self.attn(self.attn_norm(h), attention_mask=attention_mask, is_causal=is_causal)
        partial_block = partial_block + attn_out

        h = self.ffn_res(blocks, partial_block)
        if self.use_moe:
            ffn_out, aux_loss, topk_indices = self.ffn(self.ffn_norm(h))
            partial_block = partial_block + ffn_out
            return blocks, partial_block, aux_loss, topk_indices
        else:
            partial_block = partial_block + self.ffn(self.ffn_norm(h))
            zero = torch.zeros((), device=partial_block.device, dtype=partial_block.dtype)
            return blocks, partial_block, zero, None


class ResidualTransformer(nn.Module):
    def __init__(self, config: ResidualConfig):
        super().__init__()
        self.config = config

        self.layers_per_block = max(1, config.num_layers // config.num_blocks)

        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

        self.layers = nn.ModuleList([
            ResidualTransformerBlock(config) for _ in range(config.num_layers)
        ])

        self.final_res = BlockAttentionResidual(
            config.d_model, eps=config.norm_eps,
            isolate_softmax=getattr(config, "block_residual_isolate_softmax", False),
        )
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
            if isinstance(module, BlockAttentionResidual):
                nn.init.zeros_(module.proj)

    def _is_block_boundary(self, layer_idx: int) -> bool:
        return layer_idx > 0 and (layer_idx % self.layers_per_block == 0)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        is_causal: bool = True,
    ) -> Dict[str, Any]:
        x = self.token_emb(input_ids)
        x = self.dropout(x)

        # Embedding is "block 0"; partial accumulates a fresh block.
        blocks: List[torch.Tensor] = [x]
        partial_block = torch.zeros_like(x)

        aux_loss = torch.zeros((), device=input_ids.device, dtype=x.dtype)
        topk_indices_list: List[Optional[torch.Tensor]] = []

        for layer_idx, layer in enumerate(self.layers):
            # Non-in-place block-list update for compile/checkpointing safety.
            if self._is_block_boundary(layer_idx):
                blocks = blocks + [partial_block]
                partial_block = torch.zeros_like(partial_block)

            blocks, partial_block, layer_aux, layer_topk = layer(
                blocks,
                partial_block,
                attention_mask=attention_mask,
                is_causal=is_causal,
            )
            aux_loss = aux_loss + layer_aux
            topk_indices_list.append(layer_topk)

        x = self.final_res(blocks, partial_block)
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
        batch_size = input_ids.size(0)

        for _ in range(max_new_tokens):
            outputs = self.forward(
                input_ids, attention_mask=attention_mask, is_causal=True,
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
                    torch.ones(
                        (batch_size, 1),
                        device=attention_mask.device,
                        dtype=attention_mask.dtype,
                    ),
                ], dim=-1)

            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

        return input_ids
