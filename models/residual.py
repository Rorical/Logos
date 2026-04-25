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


class BlockAttentionResidual(nn.Module):
    """Depth-wise softmax over completed block states + the current partial.

    ``proj`` is zero-initialised so the layer starts as a uniform average,
    matching a standard residual sum in expectation. Softmax runs in fp32.
    """

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.norm = RMSNorm(d_model, eps=eps)
        self.proj = nn.Linear(d_model, 1, bias=False)
        nn.init.zeros_(self.proj.weight)

    def forward(
        self,
        blocks: List[torch.Tensor],
        partial_block: torch.Tensor,
    ) -> torch.Tensor:
        if len(blocks) == 0:
            return partial_block

        values = torch.stack(blocks + [partial_block], dim=0)
        keys = self.norm(values)
        logits = self.proj(keys).squeeze(-1)
        weights = torch.softmax(logits.float(), dim=0).to(values.dtype)
        return torch.sum(weights.unsqueeze(-1) * values, dim=0)


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

        self.attn_res = BlockAttentionResidual(config.d_model, eps=config.norm_eps)
        self.ffn_res = BlockAttentionResidual(config.d_model, eps=config.norm_eps)

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
            zero = torch.tensor(0.0, device=partial_block.device, dtype=partial_block.dtype)
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

        self.final_res = BlockAttentionResidual(config.d_model, eps=config.norm_eps)
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
                nn.init.zeros_(module.proj.weight)

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

        aux_loss = torch.tensor(0.0, device=input_ids.device, dtype=x.dtype)
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
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            if (shift_labels != -100).any():
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100,
                )
            else:
                loss = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)

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
