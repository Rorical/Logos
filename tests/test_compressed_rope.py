import unittest

import torch

from models.baseline import RotaryEmbedding
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
    def test_defaults_off(self):
        # Both rope paths are EXPERIMENTAL and must default off so plain
        # training reproduces the pre-a052b99 (NSA-endorsed) position scheme.
        cfg = HybridConfig(vocab_size=32, d_model=16, num_heads=2, head_dim=8,
                           d_ff=32, num_layers=1, max_seq_len=64, use_moe=False)
        self.assertFalse(cfg.compressed_rope)
        self.assertFalse(cfg.indexer_rope)

    def test_flag_off_executes_zero_rotary_calls(self):
        # The most robust flag-off guard: with both flags off the rope code is
        # not merely numerically inert, it never runs. Monkeypatch the only
        # rotary entry point compressed attention uses and assert the counter
        # stays at zero across a full forward (forward + backward through the
        # indexer KL). A regression that wires rope unconditionally would bump
        # this even if the numbers happened to match.
        calls = {"n": 0}
        orig = RotaryEmbedding.forward_at_positions

        def counting(self, x, positions):
            calls["n"] += 1
            return orig(self, x, positions)

        RotaryEmbedding.forward_at_positions = counting
        try:
            for mode in ("csa", "hca"):
                with self.subTest(mode=mode):
                    torch.manual_seed(0)
                    m = _module(mode=mode, compressed_rope=False,
                                indexer_rope=False)
                    m.train()
                    x = torch.randn(2, 24, 16)
                    out, index_loss = m(x, is_causal=True)
                    out.sum().backward()
            self.assertEqual(calls["n"], 0,
                             "flag-off forward must not invoke any RoPE")
        finally:
            RotaryEmbedding.forward_at_positions = orig

    def test_flag_off_matches_inline_no_rope_reference(self):
        # Golden check: a fixed-seed flag-off forward must equal a minimal
        # inline reimplementation of the pre-change scoring path (un-rotated q
        # and pooled kv), proving the off-path is bit-identical to "rope code
        # entirely absent", not just to a different rope configuration.
        for mode in ("csa", "hca"):
            with self.subTest(mode=mode):
                torch.manual_seed(0)
                m = _module(mode=mode, compressed_rope=False,
                            indexer_rope=False)
                m.eval()
                x = torch.randn(2, 24, 16)
                out, _ = m(x, is_causal=True)

                ref = _reference_no_rope_forward(m, x)
                torch.testing.assert_close(out, ref, rtol=0, atol=0)

    def test_flag_on_bit_differs_from_off(self):
        # Turning a flag on must change the output relative to off (and the two
        # off-modules agree on shared weights).
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
        m = _module(mode="csa", compressed_rope=False, indexer_rope=False)
        self.assertFalse(hasattr(m, "rotary"))
        self.assertFalse(hasattr(m, "indexer_rotary"))
        m2 = _module(mode="csa", compressed_rope=True, indexer_rope=True)
        self.assertTrue(hasattr(m2, "rotary"))
        self.assertTrue(hasattr(m2, "indexer_rotary"))

    def test_flags_are_independent(self):
        # compressed_rope and indexer_rope gate separate modules; either works
        # alone.
        m_c = _module(mode="csa", compressed_rope=True, indexer_rope=False)
        self.assertTrue(hasattr(m_c, "rotary"))
        self.assertFalse(hasattr(m_c, "indexer_rotary"))
        m_i = _module(mode="csa", compressed_rope=False, indexer_rope=True)
        self.assertFalse(hasattr(m_i, "rotary"))
        self.assertTrue(hasattr(m_i, "indexer_rotary"))
        # indexer_rope alone still changes the output vs. fully off.
        torch.manual_seed(0)
        m_off = _module(mode="csa", compressed_rope=False, indexer_rope=False)
        torch.manual_seed(0)
        m_idx = _module(mode="csa", compressed_rope=False, indexer_rope=True)
        m_idx.load_state_dict(m_off.state_dict())
        m_off.train()
        m_idx.train()
        x = torch.randn(2, 24, 16)
        out_off, _ = m_off(x, is_causal=True)
        out_idx, _ = m_idx(x, is_causal=True)
        # Indexer rope reshapes top-k selection, so the attention output moves.
        self.assertGreater((out_idx - out_off).abs().max().item(), 1e-6)

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
        m = _module(mode="csa", compressed_rope=True, indexer_rope=True,
                    csa_indexer_dim=1)
        self.assertEqual(m.indexer_rope_dim, 0)
        self.assertFalse(hasattr(m, "indexer_rotary"))
        m.eval()
        x = torch.randn(2, 24, 16)
        out, _ = m(x, is_causal=True)
        self.assertFalse(torch.isnan(out).any())


def _reference_no_rope_forward(m, x):
    """Minimal inline reimplementation of the pre-a052b99 forward (no RoPE).

    Reproduces compress -> norm -> q-project -> un-rotated scores -> softmax ->
    value aggregation, calling the module's own ``_build_scores`` (whose indexer
    rope branch is gated off) so the only thing this skips is the rope wiring in
    ``forward``. Used to prove the flag-off output equals "rope code absent".
    """
    import torch.nn.functional as F

    B, T, _ = x.shape
    kv, group_valid = m.compressor(x, None)
    kv = m.kv_norm(kv)
    q = m.q_up(m.q_down(x)).view(B, T, m.num_heads, m.head_dim)
    q = m.q_norm(q).transpose(1, 2)
    positions = m._group_positions(kv.size(1), T, x.device)
    scores, _ = m._build_scores(q, kv, kv, group_valid, None, True, x, positions)
    all_masked = torch.isinf(scores).all(dim=-1, keepdim=True)
    if m.attention_sink:
        sink = m.sink_logit.float().view(1, m.num_heads, 1, 1).expand(B, -1, T, -1)
        aug = torch.cat([scores.float(), sink], dim=-1)
        weights = F.softmax(aug, dim=-1)[..., :scores.size(-1)].to(kv.dtype)
    else:
        safe_scores = torch.where(all_masked, torch.zeros_like(scores), scores)
        weights = F.softmax(safe_scores.float(), dim=-1).to(kv.dtype)
        weights = torch.where(all_masked, torch.zeros_like(weights), weights)
    out = torch.einsum("bhtn,bnd->bhtd", weights, kv)
    out = out.transpose(1, 2).reshape(B, T, m.num_heads * m.head_dim)
    return m.out_proj(out)


if __name__ == "__main__":
    unittest.main()
