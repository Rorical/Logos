"""CPU, network-free tests for Fix 2 ("exclude norms, biases, and flagged
params from weight decay").

Builds a tiny ``LogosTransformer`` and drives the real
``split_param_groups`` / ``build_optimizer_and_scheduler`` path, then
asserts:

  (a) every RMSNorm weight, every bias, every BlockAttentionResidual
      ``proj``, every attention ``sink_logit``, and KDA's flagged
      ``A_log`` / ``dt_bias`` lands in an AdamW group with
      ``weight_decay == 0.0``;
  (b) 2D hidden weights still go to Muon (so the no-decay routing did
      not steal matrices from the Muon bucket);
  (c) the embed group keeps its lr and wd (status quo), and the router
      group keeps its lr and wd (router.linear.weight is 2D and decayed
      today — the fix only makes that explicit);
  (d) the no_decay group's ``weight_decay`` survives a
      ``MuonHyperparamScheduler`` step (the scheduler must iterate only
      the Muon optimizer's groups, never AdamW's).
"""

import unittest
from types import SimpleNamespace

import torch

import scripts.train as train
from models.logos import LogosConfig, LogosTransformer


def _tiny_model():
    # ``swa`` so an attention sink_logit param actually exists; ``kda``
    # blocks then also contribute A_log/dt_bias (flagged) and 3D conv
    # kernels (decayed). One entry/exit + a looped body covers all
    # parameter kinds the audit cares about.
    cfg = LogosConfig(
        vocab_size=32,
        d_model=16,
        num_heads=2,
        head_dim=8,
        d_ff=32,
        num_entry_layers=1,
        num_body_layers=1,
        num_exit_layers=1,
        num_loops=2,
        max_seq_len=16,
        use_moe=True,
        num_shared_experts=1,
        num_sparse_experts=2,
        top_k=1,
        expert_d_ff=16,
        attention_sink=True,
        attn_pattern="swa,kda",
    )
    return LogosTransformer(cfg)


def _opt_args():
    # Only the fields build_optimizer_and_scheduler / the Muon scheduler
    # actually read. Values mirror the Colab recipe's non-zero wd.
    return SimpleNamespace(
        lr=1e-3,
        embed_lr=2e-3,
        router_lr=5e-4,
        weight_decay=0.1,
        muon=True,
        muon_lr=2e-2,
        muon_wd_start=0.2,
        muon_wd_end=0.0,
        muon_momentum_start=0.85,
        muon_momentum_mid=0.90,
        muon_momentum_end=0.95,
        muon_momentum_warmup_1=2,
        muon_momentum_warmup_2=4,
        muon_schedule_hyperparams=True,
        scheduler="wsd",
        warmup_steps=2,
        router_warmup_steps=None,
        decay_steps=2,
        decay_frac=0.1,
        total_steps=10,
    )


class WeightDecayGroupTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = _tiny_model()
        self.args = _opt_args()
        (
            self.muon_params,
            self.embed_params,
            self.no_decay_params,
            self.default_params,
            self.router_params,
        ) = train.split_param_groups(self.model)
        self.optimizer, self.scheduler, self.muon_hp = (
            train.build_optimizer_and_scheduler(
                self.args,
                total_steps=self.args.total_steps,
                fused_adamw=False,
                muon_params=self.muon_params,
                embed_params=self.embed_params,
                default_params=self.default_params,
                router_params=self.router_params,
                no_decay_params=self.no_decay_params,
            )
        )
        self.adamw = self._adamw_optimizer()

    def _adamw_optimizer(self):
        for opt in self.optimizer.optimizers:
            if isinstance(opt, torch.optim.AdamW):
                return opt
        self.fail("no AdamW optimizer was built")

    def _no_decay_ids(self):
        return {id(p) for p in self.no_decay_params}

    def _muon_ids(self):
        return {id(p) for p in self.muon_params}

    # ----- (a) every gain/bias/flag is in a wd=0 group -------------------

    def test_norms_biases_proj_sinks_flags_have_zero_decay(self):
        nd_ids = self._no_decay_ids()
        # Map each AdamW group's param ids to that group's weight_decay so
        # we can look up the real wd the optimizer will apply.
        wd_by_id = {}
        for g in self.adamw.param_groups:
            for p in g["params"]:
                wd_by_id[id(p)] = g["weight_decay"]

        expect_zero = []
        for name, p in self.model.named_parameters():
            is_norm = name.endswith("norm.weight") or name.endswith("_norm.weight")
            is_bias = name.endswith(".bias")
            is_bar = name.endswith("_res.proj") or name.endswith("final_res.proj")
            is_sink = name.endswith("sink_logit")
            is_flagged = getattr(p, "_no_weight_decay", False)
            if is_norm or is_bias or is_bar or is_sink or is_flagged:
                expect_zero.append((name, p))

        # Sanity: the model really does contain each kind we claim to test.
        kinds = {
            "norm": any(n.endswith("norm.weight") for n, _ in expect_zero),
            "bias": any(n.endswith(".bias") for n, _ in expect_zero),
            "bar": any(n.endswith("_res.proj") for n, _ in expect_zero),
            "sink": any(n.endswith("sink_logit") for n, _ in expect_zero),
            "A_log": any(n.endswith("A_log") for n, _ in expect_zero),
            "dt_bias": any(n.endswith("dt_bias") for n, _ in expect_zero),
        }
        for kind, present in kinds.items():
            self.assertTrue(present, f"tiny model produced no {kind} param to test")

        for name, p in expect_zero:
            self.assertIn(
                id(p), nd_ids,
                f"{name} should be routed to the no_decay group",
            )
            self.assertIn(id(p), wd_by_id, f"{name} missing from AdamW groups")
            self.assertEqual(
                wd_by_id[id(p)], 0.0,
                f"{name} must run with weight_decay=0, got {wd_by_id[id(p)]}",
            )

    def test_flagged_kda_params_never_decayed(self):
        nd_ids = self._no_decay_ids()
        flagged = [
            (n, p) for n, p in self.model.named_parameters()
            if getattr(p, "_no_weight_decay", False)
        ]
        self.assertTrue(flagged, "expected KDA A_log/dt_bias to be flagged")
        for name, p in flagged:
            self.assertIn(id(p), nd_ids, f"flagged {name} escaped no_decay")
            self.assertNotIn(
                id(p), self._muon_ids(),
                f"flagged {name} must not be handed to Muon",
            )

    # ----- (b) 2D hidden weights still go to Muon ------------------------

    def test_two_d_hidden_weights_go_to_muon(self):
        muon_ids = self._muon_ids()
        # q/k/v/o projections and MoE expert matrices are 2D and must
        # stay on Muon, untouched by the no_decay routing.
        sampled = 0
        for name, p in self.model.named_parameters():
            two_d_hidden = (
                p.ndim == 2
                and not name.endswith("router.linear.weight")
                and "token_emb" not in name
                and "lm_head" not in name
            )
            if two_d_hidden:
                sampled += 1
                self.assertIn(
                    id(p), muon_ids,
                    f"2D hidden weight {name} should be on Muon",
                )
        self.assertGreater(sampled, 0, "no 2D hidden weights found")

    # ----- (c) embed + router groups keep lr and wd ----------------------

    def test_embed_and_router_groups_keep_lr_and_wd(self):
        embed_ids = {id(p) for p in self.embed_params}
        router_ids = {id(p) for p in self.router_params}
        self.assertTrue(embed_ids, "expected a tied embedding param")
        self.assertTrue(router_ids, "expected MoE router params")

        # LambdaLR has already scaled each group's live ``lr`` by the
        # warmup multiplier (≈0 at step 0). The configured base lr is
        # preserved in ``initial_lr`` — that's what we assert against.
        saw_embed = saw_router = False
        for g in self.adamw.param_groups:
            gids = {id(p) for p in g["params"]}
            if gids & embed_ids:
                saw_embed = True
                self.assertEqual(g["initial_lr"], self.args.embed_lr)
                self.assertEqual(g["weight_decay"], self.args.weight_decay)
            if gids & router_ids:
                saw_router = True
                self.assertEqual(g["initial_lr"], self.args.router_lr)
                self.assertEqual(g["weight_decay"], self.args.weight_decay)
        self.assertTrue(saw_embed, "embed group missing from AdamW")
        self.assertTrue(saw_router, "router group missing from AdamW")

    # ----- (d) Muon scheduler must not touch the AdamW no_decay group ----

    def test_muon_schedule_leaves_adamw_no_decay_at_zero(self):
        self.assertIsNotNone(self.muon_hp, "Muon HP scheduler expected")

        def adamw_no_decay_wds():
            wds = []
            for g in self.adamw.param_groups:
                if any(id(p) in self._no_decay_ids() for p in g["params"]):
                    wds.append(g["weight_decay"])
            return wds

        before = adamw_no_decay_wds()
        self.assertTrue(before, "no AdamW no_decay group found")
        self.assertTrue(all(wd == 0.0 for wd in before))

        # Several scheduler steps mutate Muon's wd toward muon_wd_end but
        # must never reach into AdamW's param groups.
        for _ in range(5):
            self.muon_hp.step()

        after = adamw_no_decay_wds()
        self.assertTrue(
            all(wd == 0.0 for wd in after),
            f"Muon HP schedule leaked into AdamW no_decay wd: {after}",
        )

        # And the Muon optimizer's own wd did move (proves the scheduler
        # is live and was actually iterating *something*).
        muon_opt = next(
            o for o in self.optimizer.optimizers
            if isinstance(o, torch.optim.Muon)
        )
        muon_wds = {g["weight_decay"] for g in muon_opt.param_groups}
        self.assertTrue(
            any(wd != self.args.muon_wd_start for wd in muon_wds),
            "Muon HP schedule never updated the Muon group wd",
        )


if __name__ == "__main__":
    unittest.main()
