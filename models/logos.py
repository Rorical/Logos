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
import torch.utils.checkpoint as ckpt_utils

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
from models.lm_loss import (
    chunked_linear_cross_entropy,
    standard_lm_cross_entropy,
)


def _xla_checkpoint(function, *args):
    """Run activation checkpointing through PyTorch/XLA's wrapper.

    PyTorch 2.9's non-reentrant checkpoint path infers device type ``xla``
    and then tries to access ``torch.xla``, which current torch_xla builds do
    not expose. The XLA wrapper supports reentrant checkpointing instead.
    """
    try:
        from torch_xla.utils.checkpoint import checkpoint as xla_checkpoint
    except ImportError as exc:
        raise ImportError(
            "XLA gradient checkpointing requires torch_xla.utils.checkpoint. "
            "Install a torch_xla build matching your PyTorch version."
        ) from exc
    return xla_checkpoint(function, *args, use_reentrant=True)


@dataclass
class LogosConfig(HybridConfig):
    # Auto-derived from entry + body + exit in __post_init__.
    num_layers: int = 0

    num_entry_layers: int = 2
    num_body_layers: int = 4
    num_exit_layers: int = 2
    num_loops: int = 4

    # Per-stack ``top_k`` override. ``None`` falls back to ``config.top_k``.
    # Boundary stacks (entry/exit) see less effective routing data per step
    # than the loop-shared body, so their bias-balancer can take much
    # longer to spread load. Raising their top_k buys headroom against
    # the per-expert capacity = N * top_k * capacity_factor / num_experts.
    entry_top_k: Optional[int] = None
    exit_top_k: Optional[int] = None

    # Re-compute body activations during backward instead of storing them.
    # Trades wall time for lower activation memory on the body — usually
    # the difference between OOM and fitting at 4K+ context. Entry and
    # exit layers are never checkpointed (small share of activations,
    # losing their compile fusion costs more than the memory saves).
    # Granularity is controlled separately by ``ckpt_granularity``.
    gradient_checkpointing: bool = False

    # ``per-block`` (default): wrap each body block individually. Backward
    # recompute holds only one block's internal activations at a time, so
    # peak memory is the lowest available. Costs torch.compile fusion at
    # every block boundary (one HOP per body block).
    # ``per-loop``: wrap one full body-loop iteration (``num_body_layers``
    # blocks) per HOP. Inductor regains fusion across the body, but the
    # backward recompute pass holds all ``num_body_layers`` blocks worth
    # of activations simultaneously — only viable when memory has
    # headroom. On XLA, per-block reentrant ckpt is always used because
    # torch_xla's wrapper requires tensor-flat in/out and can't pass a
    # per-block list of MoE topk tensors through cleanly.
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

    def __init__(
        self,
        config: LogosConfig,
        layer_idx: int,
        num_loops: int = 1,
        top_k: Optional[int] = None,
    ):
        super().__init__()
        self.use_moe = config.use_moe
        self.is_swa = (layer_idx % config.swa_every) == config.swa_offset

        isolate_res = getattr(
            config, "block_residual_isolate_softmax", False,
        )
        if self.is_swa:
            self.attn_norm = RMSNorm(config.d_model, eps=config.norm_eps)
            self.attn = LocalAttention(config)
            self.attn_res = BlockAttentionResidual(
                config.d_model, eps=config.norm_eps,
                isolate_softmax=isolate_res,
            )
        else:
            self.kda_norm = RMSNorm(config.d_model, eps=config.norm_eps)
            self.kda = SuperKimiDeltaAttention(config)
            self.kda_res = BlockAttentionResidual(
                config.d_model, eps=config.norm_eps,
                isolate_softmax=isolate_res,
            )

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
            self.mem_res = BlockAttentionResidual(
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
            zero = torch.zeros((), device=partial.device, dtype=partial.dtype)
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
            LogosTransformerBlock(
                config, layer_idx=i, num_loops=1,
                top_k=config.entry_top_k,
            )
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
            LogosTransformerBlock(
                config, layer_idx=exit_offset + i, num_loops=1,
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
                nn.init.zeros_(module.proj)

        for module in self.modules():
            if isinstance(module, SnapshotRetrieval):
                nn.init.zeros_(module.out_up.weight)

    def _lm_loss(
        self,
        hidden: torch.Tensor,
        labels: torch.Tensor,
        logits: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Standard next-token cross-entropy: ``hidden[i]`` predicts
        ``labels[i+1]``."""
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

        # Block outputs require_grad in training, so seed ``partial`` with the
        # same flag at every reset. Otherwise Dynamo recompiles the body /
        # exit call sites whenever ``partial`` flips from a fresh ``zeros_like``
        # (False) to a block output (True), and burns through the recompile
        # budget.
        zero_partial = lambda: torch.zeros_like(x, requires_grad=x.requires_grad)
        blocks: List[torch.Tensor] = [x]
        partial = zero_partial()

        # Recompute placement is asymmetric. Entry / exit are small and
        # always run un-checkpointed so torch.compile can fuse them; the
        # body — ``num_loops`` × ``num_body_layers`` block calls — is the
        # only section that gets ckpt'd. Body granularity is config-driven:
        #
        #  * ``per-block`` (default, lowest backward peak): one HOP per
        #    body block. The recompute pass holds at most one block's
        #    internal activations at a time. Costs Inductor fusion at
        #    every block boundary.
        #  * ``per-loop`` (opt-in, fewer HOPs): one HOP per loop
        #    iteration. Inductor regains fusion across body blocks within
        #    a loop, but backward holds ``num_body_layers`` worth of
        #    activations simultaneously — only viable with memory headroom.
        #
        # XLA always uses per-block reentrant ckpt: torch_xla's wrapper
        # requires tensor-flat in/out and can't pass a per-block list of
        # MoE topk tensors through the per-loop wrapper.
        use_ckpt = self.config.gradient_checkpointing and self.training
        use_xla_ckpt = use_ckpt and input_ids.device.type == "xla"
        per_loop_ckpt = (
            use_ckpt and not use_xla_ckpt
            and getattr(self.config, "ckpt_granularity", "per-block") == "per-loop"
        )
        per_block_ckpt = use_ckpt and not use_xla_ckpt and not per_loop_ckpt

        def _call_block(block_module, blocks_in, partial_in, loop_idx):
            return block_module(
                blocks_in, partial_in,
                attention_mask=attention_mask, is_causal=is_causal,
                cache=None, loop_idx=loop_idx,
            )

        def _ckpt_block(block_module, blocks_in, partial_in, loop_idx):
            return ckpt_utils.checkpoint(
                block_module, blocks_in, partial_in,
                attention_mask=attention_mask, is_causal=is_causal,
                cache=None, loop_idx=loop_idx,
                use_reentrant=False,
            )

        def _xla_ckpt_block(block_module, blocks_in, partial_in, loop_idx):
            def _fn(*flat_inputs):
                blocks_flat = list(flat_inputs[:-1])
                partial_flat = flat_inputs[-1]
                return block_module(
                    blocks_flat, partial_flat,
                    attention_mask=attention_mask, is_causal=is_causal,
                    cache=None, loop_idx=loop_idx,
                )
            return _xla_checkpoint(_fn, *blocks_in, partial_in)

        for layer in self.entry:
            partial, layer_aux, layer_topk = _call_block(layer, blocks, partial, 0)
            aux_loss = aux_loss + layer_aux
            topk_indices_list.append(layer_topk)
        blocks = blocks + [partial]
        partial = zero_partial()

        for loop_idx in range(self.config.num_loops):
            if per_loop_ckpt:
                # Aux losses are summed and topk indices collected inside
                # the region; both flow out through the HOP and are
                # accumulated into the outer lists. The recompute pass
                # produces fresh aux/topk that the ckpt machinery discards
                # (only the original-forward values reach the loss /
                # router-bias update), so MoE load-balancing stats are not
                # double-counted. List order within the region matches the
                # per-block append order used by ``update_router_biases``.
                _li = loop_idx
                def _body_loop(blks, p):
                    aux_sum = torch.zeros((), device=p.device, dtype=p.dtype)
                    topks: List[Optional[torch.Tensor]] = []
                    for block in self.body:
                        p, la, lt = block(
                            blks, p,
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
                # per-block (CUDA ckpt) | per-block reentrant (XLA ckpt) |
                # un-checkpointed (no ckpt) — same loop, different runner.
                if per_block_ckpt:
                    runner = _ckpt_block
                elif use_xla_ckpt:
                    runner = _xla_ckpt_block
                else:
                    runner = _call_block
                for block in self.body:
                    partial, layer_aux, layer_topk = runner(
                        block, blocks, partial, loop_idx,
                    )
                    aux_loss = aux_loss + layer_aux
                    topk_indices_list.append(layer_topk)
            blocks = blocks + [partial]
            partial = zero_partial()

        for layer in self.exit:
            partial, layer_aux, layer_topk = _call_block(layer, blocks, partial, 0)
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
