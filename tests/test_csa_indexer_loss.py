import unittest

import torch
import torch.nn.functional as F

from models.hybrid import CompressedGlobalAttention, HybridConfig
from models.logos import LogosConfig, LogosTransformer


def _csa_config(**overrides):
    cfg = dict(
        vocab_size=32,
        d_model=16,
        num_heads=2,
        head_dim=8,
        d_ff=32,
        num_entry_layers=1,
        num_body_layers=1,
        num_exit_layers=1,
        num_loops=1,
        max_seq_len=32,
        use_moe=False,
        csa_compression=4,
        csa_top_k=2,
        csa_indexer_heads=2,
        csa_indexer_dim=8,
        # Force a CSA layer somewhere in the schedule so the indexer runs.
        attn_pattern="csa",
    )
    cfg.update(overrides)
    return LogosConfig(**cfg)


def _hybrid_csa_module(mode="csa", **overrides):
    cfg = dict(
        vocab_size=32,
        d_model=16,
        num_heads=2,
        head_dim=8,
        d_ff=32,
        num_layers=1,
        max_seq_len=32,
        use_moe=False,
        csa_compression=4,
        csa_top_k=2,
        csa_indexer_heads=2,
        csa_indexer_dim=8,
    )
    cfg.update(overrides)
    return CompressedGlobalAttention(HybridConfig(**cfg), mode=mode)


class CSAIndexerLossTest(unittest.TestCase):
    def test_indexer_gets_gradient_trunk_isolated(self):
        torch.manual_seed(0)
        model = _csa_config(csa_indexer_loss_weight=1.0)
        net = LogosTransformer(model)
        net.train()

        input_ids = torch.randint(0, 32, (2, 24))
        labels = input_ids.clone()

        # Collect the indexer params and a representative trunk param so we can
        # check (a) indexer gets gradient and (b) trunk grads are identical to a
        # run where the indexer loss is switched off (detach isolation).
        def run(weight):
            net.zero_grad(set_to_none=True)
            net.config.csa_indexer_loss_weight = weight
            out = net(input_ids, labels=labels)
            self.assertIsNotNone(out["indexer_loss"])
            out["loss"].backward()
            grads = {}
            for name, p in net.named_parameters():
                grads[name] = None if p.grad is None else p.grad.detach().clone()
            return out, grads

        out_on, grads_on = run(1.0)
        out_off, grads_off = run(0.0)

        # Indexer loss is a real, positive, finite quantity in training.
        self.assertGreater(out_on["indexer_loss"].detach().item(), 0.0)
        self.assertTrue(torch.isfinite(out_on["indexer_loss"]))

        # (a) Indexer params receive a nonzero gradient when the loss is on.
        idx_names = [
            n for n in grads_on
            if any(k in n for k in (
                "indexer_q_down", "indexer_q_up", "indexer_k_proj", "indexer_w",
            ))
        ]
        self.assertTrue(idx_names, "no indexer params found in the model")
        for n in idx_names:
            self.assertIsNotNone(grads_on[n], f"{n} grad is None")
            self.assertGreater(grads_on[n].abs().sum().item(), 0.0, f"{n} grad is zero")

        # With weight 0 the indexer params receive no gradient at all (their
        # only training signal is the KL loss).
        for n in idx_names:
            g = grads_off[n]
            if g is not None:
                self.assertEqual(g.abs().sum().item(), 0.0, f"{n} grad nonzero at weight 0")

        # (b) Trunk grads (everything that is NOT an indexer param) are identical
        # whether or not the indexer KL loss is active — proving the detached
        # indexer pathway cannot perturb the main model.
        for n, g_on in grads_on.items():
            if n in idx_names:
                continue
            g_off = grads_off[n]
            if g_on is None or g_off is None:
                self.assertEqual(g_on is None, g_off is None, f"{n} grad presence differs")
                continue
            torch.testing.assert_close(g_on, g_off, msg=f"trunk grad changed for {n}")

    def test_kl_zero_when_indexer_matches_attention(self):
        # Build the dense per-head scores, then hand the indexer-loss helper an
        # idx_scores equal to log of the head-aggregated target distribution.
        # KL(teacher || student) is then ~0 by construction.
        torch.manual_seed(1)
        mod = _hybrid_csa_module()
        B, H, T, G = 2, 2, 6, 4
        scores = torch.randn(B, H, T, G)
        invalid = torch.zeros(B, T, G, dtype=torch.bool)

        # Teacher: per-head softmax over G, summed over heads, renormalized.
        per_head = F.softmax(scores.float(), dim=-1)
        target = per_head.sum(dim=1)
        target = target / target.sum(dim=-1, keepdim=True)

        idx_scores = target.clamp_min(1e-9).log()
        loss = mod._indexer_kl_loss(scores, idx_scores, invalid)
        self.assertLess(float(loss), 1e-5)

    def test_kl_excludes_fully_invalid_rows(self):
        torch.manual_seed(2)
        mod = _hybrid_csa_module()
        B, H, T, G = 1, 2, 3, 4
        scores = torch.randn(B, H, T, G)
        invalid = torch.zeros(B, T, G, dtype=torch.bool)
        # Make the last query position fully invalid across all groups.
        scores[:, :, -1, :] = float("-inf")
        invalid[:, -1, :] = True
        idx_scores = torch.randn(B, T, G)
        idx_scores[:, -1, :] = float("-inf")
        loss = mod._indexer_kl_loss(scores, idx_scores, invalid)
        self.assertTrue(torch.isfinite(loss))

    def test_hca_returns_zero_indexer_loss(self):
        torch.manual_seed(3)
        mod = _hybrid_csa_module(mode="hca", hca_compression=4)
        mod.train()
        hidden = torch.randn(2, 16, 16)
        out, index_loss = mod(hidden, is_causal=True)
        self.assertEqual(out.shape, (2, 16, 16))
        self.assertEqual(float(index_loss), 0.0)

    def test_eval_mode_returns_zero_indexer_loss(self):
        torch.manual_seed(4)
        mod = _hybrid_csa_module(mode="csa")
        mod.eval()
        hidden = torch.randn(2, 16, 16)
        out, index_loss = mod(hidden, is_causal=True)
        self.assertEqual(float(index_loss), 0.0)

    def test_model_eval_forward_has_no_indexer_grad_path(self):
        # In eval the surfaced indexer_loss is a detached zero and the combined
        # loss does not include it.
        torch.manual_seed(5)
        net = LogosTransformer(_csa_config())
        net.eval()
        input_ids = torch.randint(0, 32, (2, 24))
        out = net(input_ids, labels=input_ids.clone())
        self.assertEqual(float(out["indexer_loss"]), 0.0)


if __name__ == "__main__":
    unittest.main()
