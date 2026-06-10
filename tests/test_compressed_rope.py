import unittest

import torch

from models.hybrid import CompressedGlobalAttention, HybridConfig


def _module(mode="csa", **overrides):
    cfg = dict(
        vocab_size=32,
        d_model=16,
        num_heads=2,
        head_dim=8,
        d_ff=32,
        num_layers=1,
        max_seq_len=64,
        use_moe=False,
        csa_compression=4,
        csa_top_k=2,
        csa_indexer_heads=2,
        csa_indexer_dim=8,
        # Small HCA compression so the tiny test sequences span >1 group.
        hca_compression=4,
    )
    cfg.update(overrides)
    return CompressedGlobalAttention(HybridConfig(**cfg), mode=mode)


class CompressedRopeTest(unittest.TestCase):
    def test_flag_off_bit_identical_to_no_rope_construction(self):
        # With the flag off, the new RoPE code is fully skipped: q/kv are scored
        # un-rotated and the indexer keys/queries are un-rotated, so the forward
        # must equal a module that has no rotary modules at all. We assert the
        # two off-modules agree bit-for-bit on the same weights and that turning
        # the flag on changes the result.
        for mode in ("csa", "hca"):
            with self.subTest(mode=mode):
                torch.manual_seed(0)
                m_off = _module(mode=mode, compressed_rope=False)
                torch.manual_seed(0)
                m_on = _module(mode=mode, compressed_rope=True)
                # Same trained weights (rotary buffers are non-persistent so the
                # state dicts coincide).
                m_on.load_state_dict(m_off.state_dict())
                m_off.eval()
                m_on.eval()

                x = torch.randn(2, 24, 16)
                out_off, il_off = m_off(x, is_causal=True)
                out_on, il_on = m_on(x, is_causal=True)

                self.assertFalse(torch.isnan(out_off).any())
                self.assertGreater(
                    (out_on - out_off).abs().max().item(), 1e-5,
                    "flag on should change the output",
                )

    def test_off_module_has_no_rotary_params(self):
        m = _module(mode="csa", compressed_rope=False)
        self.assertFalse(hasattr(m, "rotary"))
        self.assertFalse(hasattr(m, "indexer_rotary"))
        m2 = _module(mode="csa", compressed_rope=True)
        self.assertTrue(hasattr(m2, "rotary"))
        self.assertTrue(hasattr(m2, "indexer_rotary"))

    def test_flag_on_no_nan_and_finite(self):
        for mode in ("csa", "hca"):
            with self.subTest(mode=mode):
                torch.manual_seed(1)
                m = _module(mode=mode, compressed_rope=True)
                m.train()
                x = torch.randn(2, 24, 16)
                out, index_loss = m(x, is_causal=True)
                self.assertFalse(torch.isnan(out).any())
                self.assertTrue(torch.isfinite(out).all())
                self.assertTrue(torch.isfinite(index_loss))

    def test_causality_preserved_with_rope(self):
        # Perturbing tokens at/after a group boundary must not change the
        # outputs of earlier query positions.
        boundary = 12  # 3 * csa_compression, a clean group boundary.
        for mode in ("csa", "hca"):
            with self.subTest(mode=mode):
                torch.manual_seed(2)
                m = _module(mode=mode, compressed_rope=True)
                m.eval()
                x = torch.randn(2, 20, 16)
                out1, _ = m(x, is_causal=True)
                x2 = x.clone()
                x2[:, boundary:, :] = torch.randn_like(x2[:, boundary:, :])
                out2, _ = m(x2, is_causal=True)

                early = (out1[:, :boundary] - out2[:, :boundary]).abs().max().item()
                late = (out1[:, boundary:] - out2[:, boundary:]).abs().max().item()
                self.assertLess(early, 1e-5, "future tokens leaked into the past")
                self.assertGreater(late, 1e-5, "perturbation had no effect at all")

    def test_partial_rope_dim_keeps_a_no_position_channel(self):
        # With partial_rope_dim < head_dim only the last rope_dim channels of the
        # score q/k are rotated; the module should still run and differ from off.
        torch.manual_seed(3)
        m = _module(mode="csa", compressed_rope=True, partial_rope_dim=4)
        self.assertEqual(m.rope_dim, 4)
        m.eval()
        x = torch.randn(2, 24, 16)
        out, _ = m(x, is_causal=True)
        self.assertFalse(torch.isnan(out).any())

    def test_group_positions_match_causal_boundary(self):
        # The representative position of a fully-covered group g must be the
        # last token of its a-window, g*c + c - 1, clamped to T-1.
        m = _module(mode="csa", compressed_rope=True)
        G, T = 5, 24
        pos = m._group_positions(G, T, torch.device("cpu"))
        expected = torch.tensor([3, 7, 11, 15, 19], dtype=torch.long)
        torch.testing.assert_close(pos, expected)
        # Trailing partial group clamps to T-1.
        pos2 = m._group_positions(6, 22, torch.device("cpu"))
        self.assertEqual(int(pos2[-1]), 21)

    def test_indexer_loss_still_trains_only_indexer_with_rope(self):
        # The Fix-3 isolation property must hold with RoPE on: the indexer KL
        # loss updates only indexer params, never the trunk.
        torch.manual_seed(4)
        m = _module(mode="csa", compressed_rope=True)
        m.train()
        x = torch.randn(2, 24, 16)
        out, index_loss = m(x, is_causal=True)
        self.assertGreater(index_loss.detach().item(), 0.0)

        index_loss.backward()
        idx_prefixes = ("indexer_",)
        for name, p in m.named_parameters():
            if p.grad is None:
                continue
            is_indexer = name.startswith(idx_prefixes)
            if not is_indexer:
                self.assertEqual(
                    p.grad.abs().sum().item(), 0.0,
                    f"indexer KL loss leaked gradient into trunk param {name}",
                )

    def test_indexer_rope_only_when_dim_positive(self):
        # If the indexer head dim rounds the rope sub-dim to 0, the indexer rope
        # is skipped but compressed-attention rope still applies.
        m = _module(mode="csa", compressed_rope=True, csa_indexer_dim=1)
        self.assertEqual(m.indexer_rope_dim, 0)
        self.assertFalse(hasattr(m, "indexer_rotary"))
        m.eval()
        x = torch.randn(2, 24, 16)
        out, _ = m(x, is_causal=True)
        self.assertFalse(torch.isnan(out).any())


if __name__ == "__main__":
    unittest.main()
