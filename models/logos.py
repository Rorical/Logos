"""Logos — a sub-quadratic decoder-only transformer.

Each parameter-block is one of two kinds:

* **KDA + Retrieval**  Kimi Delta Attention scan, followed by sparse
  top-k attention over MLA-compressed state snapshots.
* **Local SWA**        Causal sliding-window softmax attention with a
  fixed window of ``swa_window`` tokens.

Block kind is determined structurally by
``layer_idx % swa_every == swa_offset``.

The model is partitioned into three sections:

* **Entry**   ``num_entry_layers`` blocks, run once.
* **Body**    ``num_body_layers`` blocks with weights shared across
  ``num_loops`` iterations per forward.
* **Exit**    ``num_exit_layers`` blocks, run once.

Sublayer inputs are computed by Block Attention Residual: a learned
depth-wise softmax over the list of completed-block representations and
the current partial sum. Section and loop boundaries close partial sums
into new completed blocks; a final Block Attention Residual produces the
LM-head input.

Body MoE layers carry per-loop router-bias rows and a cross-loop
expert-diversity term (``moe_diversity_factor``), so shared weights
specialise differently across loop iterations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.baseline import (
    RMSNorm,
    SwiGLU,
    MoELayer,
    count_parameters,
    model_summary,
)
from models.superlinear import (
    SuperKimiDeltaAttention,
    SnapshotRetrieval,
)
from models.hybrid import HybridConfig, LocalAttention
from models.residual import BlockAttentionResidual


@dataclass
class LogosConfig(HybridConfig):
    # Auto-derived from entry + body + exit in __post_init__.
    num_layers: int = 0

    num_entry_layers: int = 2
    num_body_layers: int = 4
    num_exit_layers: int = 2
    num_loops: int = 4

    def __post_init__(self):
        super().__post_init__()
        if self.num_body_layers <= 0 or self.num_loops <= 0:
            raise ValueError(
                "num_body_layers and num_loops must both be > 0"
            )
        if self.num_entry_layers < 0 or self.num_exit_layers < 0:
            raise ValueError(
                "num_entry_layers and num_exit_layers must be >= 0"
            )
        self.num_layers = (
            self.num_entry_layers
            + self.num_body_layers
            + self.num_exit_layers
        )


class LogosTransformerBlock(nn.Module):
    """A single Logos parameter-block.

    SWA blocks consist of LocalAttention + FFN with two Block Attention
    Residual modules. KDA blocks consist of Kimi Delta Attention +
    Snapshot Retrieval + FFN with three Block Attention Residual modules.
    The MoE FFN exposes per-loop router-bias rows when ``num_loops > 1``.
    """

    def __init__(self, config: LogosConfig, layer_idx: int, num_loops: int = 1):
        super().__init__()
        self.use_moe = config.use_moe
        self.is_swa = (layer_idx % config.swa_every) == config.swa_offset

        if self.is_swa:
            self.attn_norm = RMSNorm(config.d_model, eps=config.norm_eps)
            self.attn = LocalAttention(config)
            self.attn_res = BlockAttentionResidual(config.d_model, eps=config.norm_eps)
        else:
            self.kda_norm = RMSNorm(config.d_model, eps=config.norm_eps)
            self.kda = SuperKimiDeltaAttention(config)
            self.kda_res = BlockAttentionResidual(config.d_model, eps=config.norm_eps)

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
            self.mem_res = BlockAttentionResidual(config.d_model, eps=config.norm_eps)

        self.ffn_norm = RMSNorm(config.d_model, eps=config.norm_eps)
        if config.use_moe:
            self.ffn = MoELayer(config, num_loops=num_loops)
        else:
            self.ffn = SwiGLU(config.d_model, config.d_ff, config.dropout)
        self.ffn_res = BlockAttentionResidual(config.d_model, eps=config.norm_eps)

    def forward(
        self,
        blocks: List[torch.Tensor],
        partial: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        is_causal: bool = True,
        cache: Optional[Dict[str, Any]] = None,
        loop_idx: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        if self.is_swa:
            h = self.attn_res(blocks, partial)
            attn_out = self.attn(
                self.attn_norm(h), attention_mask=attention_mask, is_causal=is_causal,
            )
            partial = partial + attn_out
        else:
            h = self.kda_res(blocks, partial)
            kda_out, snapshots, snap_positions = self.kda(
                self.kda_norm(h), attention_mask=attention_mask, cache=cache,
            )
            partial = partial + kda_out

            h = self.mem_res(blocks, partial)
            if cache is not None:
                token_offset = cache.get("n_processed", h.size(1)) - h.size(1)
            else:
                token_offset = 0
            mem_out = self.mem(
                self.mem_norm(h),
                snapshots,
                snap_positions,
                token_offset=token_offset,
                attention_mask=attention_mask,
            )
            partial = partial + mem_out

        h = self.ffn_res(blocks, partial)
        if self.use_moe:
            ffn_out, aux_loss, topk_indices = self.ffn(self.ffn_norm(h), loop_idx=loop_idx)
            partial = partial + ffn_out
            return partial, aux_loss, topk_indices
        else:
            partial = partial + self.ffn(self.ffn_norm(h))
            zero = torch.tensor(0.0, device=partial.device, dtype=partial.dtype)
            return partial, zero, None


class LogosTransformer(nn.Module):
    def __init__(self, config: LogosConfig):
        super().__init__()
        self.config = config

        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

        # Layer indices flow contiguously through entry, body, and exit, so
        # SWA placement is determined by a single global index space.
        self.entry = nn.ModuleList([
            LogosTransformerBlock(config, layer_idx=i, num_loops=1)
            for i in range(config.num_entry_layers)
        ])

        body_offset = config.num_entry_layers
        self.body = nn.ModuleList([
            LogosTransformerBlock(
                config,
                layer_idx=body_offset + i,
                num_loops=config.num_loops,
            )
            for i in range(config.num_body_layers)
        ])

        exit_offset = config.num_entry_layers + config.num_body_layers
        self.exit = nn.ModuleList([
            LogosTransformerBlock(config, layer_idx=exit_offset + i, num_loops=1)
            for i in range(config.num_exit_layers)
        ])

        self.final_res = BlockAttentionResidual(config.d_model, eps=config.norm_eps)
        self.final_norm = RMSNorm(config.d_model, eps=config.norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

        self._init_weights()

    def _init_weights(self):
        # Block Attention Residual proj weights are zero-initialised so each
        # depth-wise softmax starts uniform. SnapshotRetrieval.out_up is
        # zero-initialised so the memory branch is an exact no-op at step 0.
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

        for module in self.modules():
            if isinstance(module, SnapshotRetrieval):
                nn.init.zeros_(module.out_up.weight)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        is_causal: bool = True,
    ) -> Dict[str, Any]:
        x = self.token_emb(input_ids)
        x = self.dropout(x)

        aux_loss = torch.tensor(0.0, device=input_ids.device, dtype=x.dtype)
        topk_indices_list: List[Optional[torch.Tensor]] = []

        blocks: List[torch.Tensor] = [x]
        partial = torch.zeros_like(x)

        for layer in self.entry:
            partial, layer_aux, layer_topk = layer(
                blocks, partial,
                attention_mask=attention_mask, is_causal=is_causal,
                cache=None, loop_idx=0,
            )
            aux_loss = aux_loss + layer_aux
            topk_indices_list.append(layer_topk)
        blocks = blocks + [partial]
        partial = torch.zeros_like(x)

        for loop_idx in range(self.config.num_loops):
            for block in self.body:
                partial, layer_aux, layer_topk = block(
                    blocks, partial,
                    attention_mask=attention_mask, is_causal=is_causal,
                    cache=None, loop_idx=loop_idx,
                )
                aux_loss = aux_loss + layer_aux
                topk_indices_list.append(layer_topk)
            blocks = blocks + [partial]
            partial = torch.zeros_like(x)

        for layer in self.exit:
            partial, layer_aux, layer_topk = layer(
                blocks, partial,
                attention_mask=attention_mask, is_causal=is_causal,
                cache=None, loop_idx=0,
            )
            aux_loss = aux_loss + layer_aux
            topk_indices_list.append(layer_topk)

        x = self.final_res(blocks, partial)
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
        """Apply DeepSeek-style bias updates to all MoE routers.

        ``topk_indices_list`` is laid out as
        ``[entry, loop_0.body, loop_1.body, ..., loop_{L-1}.body, exit]``.
        Body blocks receive a single grouped update across all loop
        iterations so the cross-loop diversity term observes them jointly.
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
        for r, block in enumerate(self.body):
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
        """Per-parameter-set router-bias statistics. Body sub-blocks appear
        once each (regardless of ``num_loops``)."""
        if not self.config.use_moe:
            return {}

        stats: Dict[str, float] = {}

        def _record(name: str, layer: nn.Module) -> None:
            ffn = layer.ffn
            if not hasattr(ffn, "bias"):
                return
            kind = "swa" if layer.is_swa else "kda"
            bias = ffn.bias
            stats[f"{name}_{kind}_bias_mean"] = bias.abs().mean().item()
            stats[f"{name}_{kind}_bias_max"] = bias.abs().max().item()

        for idx, layer in enumerate(self.entry):
            _record(f"entry{idx}", layer)
        for idx, block in enumerate(self.body):
            _record(f"body{idx}", block)
        for idx, layer in enumerate(self.exit):
            _record(f"exit{idx}", layer)
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
        """Autoregressive generation by full re-forward over the growing
        prefix. Per-step cache reuse is not implemented: the body re-runs
        its KDA scan on every loop iteration, so a cache would have to
        either reset state per loop or carry it across loops, which would
        change the model's semantics."""
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
    "LogosConfig",
    "LogosTransformerBlock",
    "LogosTransformer",
    "count_parameters",
    "model_summary",
]
