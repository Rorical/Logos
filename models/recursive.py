"""Recursive (looped-depth) decoder-only transformer.

Three sections — entry / body / exit — where the body is a small stack of
shared weights applied ``num_loops`` times per forward. The loop update is
``h_{t+1} = A * h_t + B * e + R(h_t + e)`` with per-channel injection
gates A, B initialised to zero (so the loop starts as a weight-shared
transformer stack on h+e). Optional cross-loop expert diversity for shared
MoE routers via ``moe_diversity_factor``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.baseline import (
    BaselineConfig,
    RMSNorm,
    TransformerBlock,
    MoELayer,
    count_parameters,
    model_summary,
)


@dataclass
class RecursiveConfig(BaselineConfig):
    # Auto-derived from entry + body + exit in __post_init__.
    num_layers: int = 0

    num_entry_layers: int = 2
    num_body_layers: int = 4
    num_exit_layers: int = 2
    num_loops: int = 4

    # Std of the random init for the per-channel A gate. 0 (default)
    # leaves the loop's residual mixing inert at step 0; small positive
    # values (e.g. 0.02) break that symmetry. B always starts at zero.
    body_gate_init_std: float = 0.0

    def __post_init__(self):
        super().__post_init__()
        if self.num_body_layers <= 0 or self.num_loops <= 0:
            raise ValueError(
                "num_body_layers and num_loops must both be > 0; set "
                "num_entry_layers / num_exit_layers to 0 if you want a "
                "purely-body model."
            )
        if self.body_gate_init_std < 0:
            raise ValueError("body_gate_init_std must be >= 0")
        self.num_layers = (
            self.num_entry_layers
            + self.num_body_layers
            + self.num_exit_layers
        )


class RecursiveBlock(nn.Module):
    """One iteration of the body loop: ``h_{t+1} = A*h + B*e + R(h+e)``.

    The body's transformer blocks are reused ``num_loops`` times, so MoE
    layers carry per-loop bias rows and the cross-loop diversity term.
    """

    def __init__(self, config: RecursiveConfig):
        super().__init__()
        self.blocks = nn.ModuleList([
            TransformerBlock(config, num_loops=config.num_loops)
            for _ in range(config.num_body_layers)
        ])
        if config.body_gate_init_std > 0:
            self.A = nn.Parameter(
                torch.randn(config.d_model) * config.body_gate_init_std
            )
        else:
            self.A = nn.Parameter(torch.zeros(config.d_model))
        self.B = nn.Parameter(torch.zeros(config.d_model))

    def forward(
        self,
        h: torch.Tensor,
        e: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        is_causal: bool = True,
        loop_idx: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[Optional[torch.Tensor]]]:
        x = h + e
        aux_loss = torch.zeros((), device=x.device, dtype=x.dtype)
        topk_list: List[Optional[torch.Tensor]] = []
        for block in self.blocks:
            x, block_aux, block_topk = block(
                x,
                attention_mask=attention_mask,
                is_causal=is_causal,
                loop_idx=loop_idx,
            )
            aux_loss = aux_loss + block_aux
            topk_list.append(block_topk)
        h_next = self.A * h + self.B * e + x
        return h_next, aux_loss, topk_list


class RecursiveTransformer(nn.Module):
    def __init__(self, config: RecursiveConfig):
        super().__init__()
        self.config = config

        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

        self.entry = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.num_entry_layers)
        ])
        self.body = RecursiveBlock(config)
        self.exit = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.num_exit_layers)
        ])

        self.final_norm = RMSNorm(config.d_model, eps=config.norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

        self._init_weights()

    def _init_weights(self):
        # ``RecursiveBlock.A`` and ``.B`` stay at their zero init — they are
        # nn.Parameter (not Linear/Embedding) and so are skipped by this pass.
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

        for layer in self.entry:
            x, layer_aux, layer_topk = layer(
                x, attention_mask=attention_mask, is_causal=is_causal
            )
            aux_loss = aux_loss + layer_aux
            topk_indices_list.append(layer_topk)
        e = x

        h = torch.zeros_like(e)
        for loop_idx in range(self.config.num_loops):
            h, block_aux, block_topks = self.body(
                h,
                e,
                attention_mask=attention_mask,
                is_causal=is_causal,
                loop_idx=loop_idx,
            )
            aux_loss = aux_loss + block_aux
            topk_indices_list.extend(block_topks)
        x = h

        for layer in self.exit:
            x, layer_aux, layer_topk = layer(
                x, attention_mask=attention_mask, is_causal=is_causal
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
        """Apply DeepSeek-style bias updates. Index layout:

            [entry_0..E-1,
             loop_0.b_0..B-1, loop_1.b_0..B-1, ..., loop_{L-1}.b_0..B-1,
             exit_0..X-1]

        Each body block is updated once per parameter set with all its
        loop iterations grouped, so the cross-loop diversity term sees
        them together.
        """
        if not self.config.use_moe:
            return

        n_entry = self.config.num_entry_layers
        n_body = self.config.num_body_layers
        n_loops = self.config.num_loops

        for i, layer in enumerate(self.entry):
            topk = topk_indices_list[i]
            if topk is not None and isinstance(layer.ffn, MoELayer):
                layer.ffn.update_bias(topk, loop_idx=0)

        body_offset = n_entry
        for r, block in enumerate(self.body.blocks):
            if not isinstance(block.ffn, MoELayer):
                continue
            topk_per_loop: List[torch.Tensor] = []
            valid = True
            for l in range(n_loops):
                idx = body_offset + l * n_body + r
                topk = topk_indices_list[idx]
                if topk is None:
                    valid = False
                    break
                topk_per_loop.append(topk)
            if valid:
                block.ffn.update_bias_per_loop(topk_per_loop)

        exit_offset = n_entry + n_loops * n_body
        for i, layer in enumerate(self.exit):
            topk = topk_indices_list[exit_offset + i]
            if topk is not None and isinstance(layer.ffn, MoELayer):
                layer.ffn.update_bias(topk, loop_idx=0)

    @torch.no_grad()
    def get_balance_stats(self) -> Dict[str, float]:
        """One entry per parameter set — body sub-blocks appear once each
        (not ``num_loops`` times)."""
        if not self.config.use_moe:
            return {}

        stats: Dict[str, float] = {}

        def _record(name: str, ffn: nn.Module) -> None:
            if hasattr(ffn, "bias"):
                bias = ffn.bias
                stats[f"{name}_bias_mean"] = bias.abs().mean().item()
                stats[f"{name}_bias_max"] = bias.abs().max().item()

        for idx, layer in enumerate(self.entry):
            _record(f"entry{idx}", layer.ffn)
        for idx, block in enumerate(self.body.blocks):
            _record(f"body{idx}", block.ffn)
        for idx, layer in enumerate(self.exit):
            _record(f"exit{idx}", layer.ffn)
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


__all__ = [
    "RecursiveConfig",
    "RecursiveBlock",
    "RecursiveTransformer",
    "count_parameters",
    "model_summary",
]
