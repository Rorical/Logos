"""Logos — looped decoder-only transformer with hybrid attention variants.

Each block selects one attention mechanism per execution from:

* ``kda`` — Kimi Delta Attention, with no snapshot memory branch.
* ``swa`` — local sliding-window softmax attention.
* ``csa`` — 4-token compressed sparse global attention.
* ``hca`` — heavily compressed dense global attention.

The model is partitioned into Entry -> Body -> Exit. Body blocks are shared
across ``num_loops`` iterations, and their attention kind can vary per loop
using a flattened loop-major ``body_attn_pattern``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as ckpt_utils

from models.baseline import (
    RMSNorm,
    SwiGLU,
    MoELayer,
    count_parameters,
    model_summary,
)
from models.hybrid import (
    HybridConfig,
    HybridAttentionLayer,
    expand_attention_pattern,
    normalize_attention_type,
    parse_attention_pattern,
)
from models.residual import BlockAttentionResidual
from models.lm_loss import (
    chunked_linear_cross_entropy,
    standard_lm_cross_entropy,
)


def _xla_checkpoint(function, *args):
    try:
        from torch_xla.utils.checkpoint import checkpoint as xla_checkpoint
    except ImportError as exc:
        raise ImportError(
            "XLA gradient checkpointing requires torch_xla.utils.checkpoint. "
            "Install a torch_xla build matching your PyTorch version."
        ) from exc
    return xla_checkpoint(function, *args, use_reentrant=True)


def _legacy_kind(layer_idx: int, config: "LogosConfig") -> str:
    return "swa" if (layer_idx % config.swa_every) == config.swa_offset else "kda"


def _default_entry_schedule(config: "LogosConfig") -> List[str]:
    return [_legacy_kind(i, config) for i in range(config.num_entry_layers)]


def _default_body_schedule(config: "LogosConfig") -> List[str]:
    out: List[str] = []
    body_offset = config.num_entry_layers
    for _ in range(config.num_loops):
        for r in range(config.num_body_layers):
            out.append(_legacy_kind(body_offset + r, config))
    return out


def _default_exit_schedule(config: "LogosConfig") -> List[str]:
    exit_offset = config.num_entry_layers + config.num_body_layers
    return [
        _legacy_kind(exit_offset + i, config)
        for i in range(config.num_exit_layers)
    ]


def _resolve_logos_attention_schedules(
    config: "LogosConfig",
) -> Tuple[List[str], List[str], List[str]]:
    n_entry = config.num_entry_layers
    n_body_exec = config.num_body_layers * config.num_loops
    n_exit = config.num_exit_layers

    section_patterns = (
        config.entry_attn_pattern,
        config.body_attn_pattern,
        config.exit_attn_pattern,
    )
    if config.attn_pattern and all(p is None for p in section_patterns):
        full = expand_attention_pattern(
            config.attn_pattern,
            n_entry + n_body_exec + n_exit,
            default="kda",
        )
        entry = full[:n_entry]
        body = full[n_entry:n_entry + n_body_exec]
        exit_ = full[n_entry + n_body_exec:]
        return entry, body, exit_

    entry = (
        expand_attention_pattern(config.entry_attn_pattern, n_entry, default="kda")
        if config.entry_attn_pattern is not None
        else _default_entry_schedule(config)
    )
    body = (
        expand_attention_pattern(config.body_attn_pattern, n_body_exec, default="kda")
        if config.body_attn_pattern is not None
        else _default_body_schedule(config)
    )
    exit_ = (
        expand_attention_pattern(config.exit_attn_pattern, n_exit, default="kda")
        if config.exit_attn_pattern is not None
        else _default_exit_schedule(config)
    )
    return entry, body, exit_


@dataclass
class LogosConfig(HybridConfig):
    # Auto-derived from entry + body + exit in __post_init__.
    num_layers: int = 0

    num_entry_layers: int = 2
    num_body_layers: int = 4
    num_exit_layers: int = 2
    num_loops: int = 4

    # Fine-grained attention schedules. ``body_attn_pattern`` is expanded to
    # ``num_loops * num_body_layers`` entries in loop-major order:
    # loop0.block0, loop0.block1, ..., loop1.block0, ...
    entry_attn_pattern: Optional[str] = None
    body_attn_pattern: Optional[str] = None
    exit_attn_pattern: Optional[str] = None

    entry_top_k: Optional[int] = None
    exit_top_k: Optional[int] = None

    gradient_checkpointing: bool = False
    ckpt_granularity: str = "per-block"

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
        if self.ckpt_granularity not in ("per-block", "per-loop"):
            raise ValueError(
                "ckpt_granularity must be 'per-block' or 'per-loop', "
                f"got {self.ckpt_granularity!r}"
            )
        for pattern in (
            self.entry_attn_pattern,
            self.body_attn_pattern,
            self.exit_attn_pattern,
        ):
            parse_attention_pattern(pattern)
        self.num_layers = (
            self.num_entry_layers
            + self.num_body_layers
            + self.num_exit_layers
        )


class LogosTransformerBlock(nn.Module):
    """A Logos parameter-block with selectable attention kind per call."""

    def __init__(
        self,
        config: LogosConfig,
        attention_kinds: List[str],
        num_loops: int = 1,
        top_k: Optional[int] = None,
    ):
        super().__init__()
        self.use_moe = config.use_moe
        self.attention_kinds = [normalize_attention_type(k) for k in attention_kinds]

        isolate_res = getattr(
            config, "block_residual_isolate_softmax", False,
        )
        self.attn_norm = RMSNorm(config.d_model, eps=config.norm_eps)
        self.attn = HybridAttentionLayer(config, self.attention_kinds)
        self.attn_res = BlockAttentionResidual(
            config.d_model, eps=config.norm_eps,
            isolate_softmax=isolate_res,
        )

        self.ffn_norm = RMSNorm(config.d_model, eps=config.norm_eps)
        if config.use_moe:
            self.ffn = MoELayer(config, num_loops=num_loops, top_k=top_k)
        else:
            self.ffn = SwiGLU(config.d_model, config.d_ff, config.dropout)
        self.ffn_res = BlockAttentionResidual(
            config.d_model, eps=config.norm_eps,
            isolate_softmax=isolate_res,
        )

    def forward(
        self,
        blocks: List[torch.Tensor],
        partial: Optional[torch.Tensor],
        attention_kind: str,
        attention_mask: Optional[torch.Tensor] = None,
        is_causal: bool = True,
        cache: Optional[Dict[str, Any]] = None,
        loop_idx: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        h = self.attn_res(blocks, partial)
        attn_out = self.attn(
            attention_kind,
            self.attn_norm(h),
            attention_mask=attention_mask,
            is_causal=is_causal,
            cache=cache,
        )
        if partial is None:
            partial = attn_out
        else:
            partial = partial + attn_out

        h = self.ffn_res(blocks, partial)
        if self.use_moe:
            ffn_out, aux_loss, topk_indices = self.ffn(self.ffn_norm(h), loop_idx=loop_idx)
            partial = partial + ffn_out
            return partial, aux_loss, topk_indices

        partial = partial + self.ffn(self.ffn_norm(h))
        zero = torch.zeros((), device=partial.device, dtype=partial.dtype)
        return partial, zero, None


class LogosTransformer(nn.Module):
    def __init__(self, config: LogosConfig):
        super().__init__()
        self.config = config

        self.entry_attn_schedule, self.body_attn_schedule, self.exit_attn_schedule = (
            _resolve_logos_attention_schedules(config)
        )

        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

        self.entry = nn.ModuleList([
            LogosTransformerBlock(
                config,
                attention_kinds=[self.entry_attn_schedule[i]],
                num_loops=1,
                top_k=config.entry_top_k,
            )
            for i in range(config.num_entry_layers)
        ])

        self.body = nn.ModuleList([
            LogosTransformerBlock(
                config,
                attention_kinds=[
                    self.body_attn_schedule[l * config.num_body_layers + i]
                    for l in range(config.num_loops)
                ],
                num_loops=config.num_loops,
            )
            for i in range(config.num_body_layers)
        ])

        self.exit = nn.ModuleList([
            LogosTransformerBlock(
                config,
                attention_kinds=[self.exit_attn_schedule[i]],
                num_loops=1,
                top_k=config.exit_top_k,
            )
            for i in range(config.num_exit_layers)
        ])

        self.final_res = BlockAttentionResidual(
            config.d_model, eps=config.norm_eps,
            isolate_softmax=getattr(
                config, "block_residual_isolate_softmax", False,
            ),
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

    def _lm_loss(
        self,
        hidden: torch.Tensor,
        labels: torch.Tensor,
        logits: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        chunk_size = int(getattr(self.config, "lm_head_chunk_size", 0) or 0)
        if chunk_size > 0 and logits is None:
            return chunked_linear_cross_entropy(
                hidden,
                self.lm_head.weight,
                labels,
                chunk_size=chunk_size,
                ignore_index=-100,
            )
        if logits is None:
            logits = self.lm_head(hidden)
        return standard_lm_cross_entropy(
            logits, labels, ignore_index=-100,
        )

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

        blocks: List[torch.Tensor] = [x]
        partial: Optional[torch.Tensor] = None

        use_ckpt = self.config.gradient_checkpointing and self.training
        use_xla_ckpt = use_ckpt and input_ids.device.type == "xla"
        per_loop_ckpt = (
            use_ckpt and not use_xla_ckpt
            and getattr(self.config, "ckpt_granularity", "per-block") == "per-loop"
        )
        per_block_ckpt = use_ckpt and not use_xla_ckpt and not per_loop_ckpt

        def _call_block(block_module, blocks_in, partial_in, attention_kind, loop_idx):
            return block_module(
                blocks_in, partial_in, attention_kind,
                attention_mask=attention_mask, is_causal=is_causal,
                cache=None, loop_idx=loop_idx,
            )

        def _ckpt_block(block_module, blocks_in, partial_in, attention_kind, loop_idx):
            return ckpt_utils.checkpoint(
                block_module, blocks_in, partial_in, attention_kind,
                attention_mask=attention_mask, is_causal=is_causal,
                cache=None, loop_idx=loop_idx,
                use_reentrant=False,
            )

        def _xla_ckpt_block(block_module, blocks_in, partial_in, attention_kind, loop_idx):
            def _fn(*flat_inputs):
                blocks_flat = list(flat_inputs[:-1])
                partial_flat = flat_inputs[-1]
                return block_module(
                    blocks_flat, partial_flat, attention_kind,
                    attention_mask=attention_mask, is_causal=is_causal,
                    cache=None, loop_idx=loop_idx,
                )
            if partial_in is None:
                return _fn(*blocks_in, None)
            return _xla_checkpoint(_fn, *blocks_in, partial_in)

        for idx, layer in enumerate(self.entry):
            partial, layer_aux, layer_topk = _call_block(
                layer, blocks, partial, self.entry_attn_schedule[idx], 0,
            )
            aux_loss = aux_loss + layer_aux
            topk_indices_list.append(layer_topk)
        if self.config.num_entry_layers > 0:
            assert partial is not None, "entry produced no partial block"
            blocks = blocks + [partial]
            partial = None

        for loop_idx in range(self.config.num_loops):
            if per_loop_ckpt:
                _li = loop_idx

                def _body_loop(blks, p):
                    aux_sum = torch.zeros((), device=blks[0].device, dtype=blks[0].dtype)
                    topks: List[Optional[torch.Tensor]] = []
                    for r, block in enumerate(self.body):
                        kind = self.body_attn_schedule[_li * self.config.num_body_layers + r]
                        p, la, lt = block(
                            blks, p, kind,
                            attention_mask=attention_mask, is_causal=is_causal,
                            cache=None, loop_idx=_li,
                        )
                        aux_sum = aux_sum + la
                        topks.append(lt)
                    return p, aux_sum, topks

                partial, loop_aux, loop_topks = ckpt_utils.checkpoint(
                    _body_loop, blocks, partial, use_reentrant=False,
                )
                aux_loss = aux_loss + loop_aux
                topk_indices_list.extend(loop_topks)
            else:
                if per_block_ckpt:
                    runner = _ckpt_block
                elif use_xla_ckpt:
                    runner = _xla_ckpt_block
                else:
                    runner = _call_block
                for r, block in enumerate(self.body):
                    kind = self.body_attn_schedule[
                        loop_idx * self.config.num_body_layers + r
                    ]
                    partial, layer_aux, layer_topk = runner(
                        block, blocks, partial, kind, loop_idx,
                    )
                    aux_loss = aux_loss + layer_aux
                    topk_indices_list.append(layer_topk)
            assert partial is not None, f"body loop {loop_idx} produced no partial block"
            blocks = blocks + [partial]
            partial = None

        for idx, layer in enumerate(self.exit):
            partial, layer_aux, layer_topk = _call_block(
                layer, blocks, partial, self.exit_attn_schedule[idx], 0,
            )
            aux_loss = aux_loss + layer_aux
            topk_indices_list.append(layer_topk)

        h_main = self.final_res(blocks, partial)
        x = self.final_norm(h_main)
        use_chunked_lm_loss = (
            labels is not None
            and int(getattr(self.config, "lm_head_chunk_size", 0) or 0) > 0
        )
        logits = None if use_chunked_lm_loss else self.lm_head(x)

        loss: Optional[torch.Tensor] = None
        if labels is not None:
            loss = self._lm_loss(x, labels, logits=logits)

        return {
            "logits": logits,
            "loss": loss,
            "aux_loss": aux_loss if self.config.use_moe else None,
            "topk_indices": topk_indices_list if self.config.use_moe else None,
        }

    def update_router_biases(self, topk_indices_list: List[Optional[torch.Tensor]]) -> None:
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
        if not self.config.use_moe:
            return {}

        stats: Dict[str, float] = {}

        def _record(name: str, layer: nn.Module, kind: str) -> None:
            ffn = layer.ffn
            if not hasattr(ffn, "bias"):
                return
            bias = ffn.bias
            stats[f"{name}_{kind}_bias_mean"] = bias.abs().mean().item()
            stats[f"{name}_{kind}_bias_max"] = bias.abs().max().item()

        for idx, layer in enumerate(self.entry):
            _record(f"entry{idx}", layer, self.entry_attn_schedule[idx])
        for idx, block in enumerate(self.body):
            kinds = sorted({
                self.body_attn_schedule[l * self.config.num_body_layers + idx]
                for l in range(self.config.num_loops)
            })
            _record(f"body{idx}", block, "+".join(kinds))
        for idx, layer in enumerate(self.exit):
            _record(f"exit{idx}", layer, self.exit_attn_schedule[idx])
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
    "LogosConfig",
    "LogosTransformerBlock",
    "LogosTransformer",
    "count_parameters",
    "model_summary",
]
