"""CPU, network-free tests for streaming-data shuffle + resumable position.

Covers Fix 1 ("shuffle streaming data and checkpoint stream position"):
  (a) shuffle reorders docs but preserves the multiset, and the val
      hold-out (first val_docs) stays fixed and unshuffled;
  (b) PackedStream emits the correct GLOBAL source_idx with base_offset,
      under a single worker and under 2 simulated workers;
  (c) resume math: consuming N docs then rebuilding skip(N) with the same
      seed continues with (nearly) disjoint documents;
  (d) checkpoint round-trip: stream_docs_consumed survives save/load.
"""

import argparse
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import datasets
import torch

import scripts.train as train
import scripts.train_xla as train_xla


class _CharTokenizer:
    """Deterministic byte-level tokenizer good enough for packing tests."""

    eos_token_id = 256

    def encode(self, text):
        return [b for b in text.encode("utf-8")]


def _make_args(**overrides):
    base = dict(
        dataset="dummy",
        dataset_config=None,
        text_column="text",
        max_length=8,
        batch_size=2,
        num_workers=0,
        prefetch_factor=2,
        seed=42,
        val_docs=2,
        stream_shuffle_buffer=100,
        stream_docs_consumed=0,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _docs(n, prefix="doc"):
    # Each doc is long enough to span at least one block after tokenisation.
    return [f"{prefix}{i:04d}-" + "x" * 16 for i in range(n)]


def _iterable(texts):
    return datasets.Dataset.from_dict({"text": texts}).to_iterable_dataset()


class ShuffleTest(unittest.TestCase):
    def test_shuffle_changes_order_same_multiset(self):
        texts = _docs(50)
        stream = _iterable(texts)
        args = _make_args(stream_shuffle_buffer=50)

        shuffled = [s["text"] for s in train.shuffle_train_stream(args, stream)]
        unshuffled = [s["text"] for s in _iterable(texts)]

        self.assertNotEqual(shuffled, unshuffled, "shuffle should reorder")
        self.assertEqual(sorted(shuffled), sorted(unshuffled),
                         "shuffle must preserve the document multiset")

    def test_shuffle_buffer_zero_is_identity(self):
        texts = _docs(20)
        args = _make_args(stream_shuffle_buffer=0)
        out = [s["text"] for s in train.shuffle_train_stream(args, _iterable(texts))]
        self.assertEqual(out, texts, "buffer=0 must disable shuffle (ablation)")

    def test_val_holdout_fixed_and_unshuffled(self):
        # The val split is built by .take(val_docs) on the un-skipped stream
        # and must stay the first val_docs docs in original order regardless
        # of shuffle settings, so val PPL stays comparable across the change.
        texts = _docs(30)
        args = _make_args(val_docs=4, stream_shuffle_buffer=30)
        val = list(_iterable(texts).take(args.val_docs))
        self.assertEqual([d["text"] for d in val], texts[:4])
        # The training stream skips those same docs before shuffling, so no
        # shuffled train doc is ever drawn from the val hold-out.
        train_stream = train.shuffle_train_stream(
            args, _iterable(texts).skip(args.val_docs),
        )
        train_texts = {s["text"] for s in train_stream}
        self.assertTrue(train_texts.isdisjoint(set(texts[:4])))


class PackedStreamSourceIdxTest(unittest.TestCase):
    def test_source_idx_global_with_base_offset_single_worker(self):
        texts = _docs(40)
        base_offset = 7
        ps = train.PackedStream(
            _iterable(texts), _CharTokenizer(), block_size=8,
            base_offset=base_offset, emit_source_idx=True,
        )
        seen = list(ps)
        self.assertTrue(seen, "expected at least one packed block")
        for chunk in seen:
            self.assertIn("source_idx", chunk)
            self.assertEqual(chunk["source_idx"].dtype, torch.long)
        # source_idx is monotonic non-decreasing and starts at base_offset.
        idxs = [int(c["source_idx"]) for c in seen]
        self.assertEqual(idxs, sorted(idxs))
        self.assertGreaterEqual(idxs[0], base_offset)
        # The last block was fed by at most the final doc (base_offset + N-1).
        self.assertLessEqual(idxs[-1], base_offset + len(texts) - 1)

    def test_no_source_idx_when_flag_off(self):
        ps = train.PackedStream(
            _iterable(_docs(10)), _CharTokenizer(), block_size=8,
        )
        for chunk in ps:
            self.assertNotIn("source_idx", chunk)
            self.assertEqual(set(chunk), {"input_ids", "attention_mask", "labels"})

    def test_two_workers_partition_global_indices(self):
        # With 2 workers, each worker enumerates the FULL source and strides by
        # idx % num_workers, so worker 0 sees even global indices and worker 1
        # odd ones. base_offset shifts every reported source_idx uniformly.
        texts = _docs(40)
        base_offset = 5

        def run_worker(worker_id, num_workers):
            ps = train.PackedStream(
                _iterable(texts), _CharTokenizer(), block_size=8,
                base_offset=base_offset, emit_source_idx=True,
            )
            info = SimpleNamespace(id=worker_id, num_workers=num_workers)
            with mock.patch.object(
                torch.utils.data, "get_worker_info", return_value=info,
            ):
                return [int(c["source_idx"]) for c in ps]

        w0 = run_worker(0, 2)
        w1 = run_worker(1, 2)
        # Reported indices include base_offset; the underlying global doc index
        # is source_idx - base_offset and must match the worker's stride class.
        self.assertTrue(all((i - base_offset) % 2 == 0 for i in w0))
        self.assertTrue(all((i - base_offset) % 2 == 1 for i in w1))


class ResumeMathTest(unittest.TestCase):
    def test_skip_before_shuffle_drops_exactly_n(self):
        # The fix deviates from a naive shuffle-then-skip: on a single-shard
        # iterable, HF's .skip(n) AFTER .shuffle drops only one shard-block, so
        # resume would replay the prefix. .skip(n) BEFORE .shuffle drops
        # exactly n docs and shuffles the remaining tail.
        texts = _docs(60)
        args = _make_args(stream_shuffle_buffer=60)
        n = 20

        resumed = [
            s["text"]
            for s in train.shuffle_train_stream(args, _iterable(texts).skip(n))
        ]
        self.assertEqual(len(resumed), len(texts) - n,
                         "skip-before-shuffle must drop exactly n docs")
        self.assertEqual(set(resumed), set(texts[n:]),
                         "resumed multiset must be exactly the unread tail")
        self.assertTrue(set(resumed).isdisjoint(set(texts[:n])),
                        "resumed docs must not replay the consumed prefix")

    def test_shuffle_train_stream_applies_resume_skip_then_shuffle(self):
        # _shuffle_train_stream is the single-process train pipeline applied
        # after .skip(val_docs): it composes .skip(consumed) then .shuffle.
        texts = _docs(40)
        args0 = _make_args(stream_shuffle_buffer=40, stream_docs_consumed=0)
        args_resumed = _make_args(stream_shuffle_buffer=40, stream_docs_consumed=15)

        fresh = [s["text"] for s in train._shuffle_train_stream(args0, _iterable(texts))]
        resumed = [
            s["text"]
            for s in train._shuffle_train_stream(args_resumed, _iterable(texts))
        ]
        # A fresh run shuffles all docs; a resumed run skips the first 15
        # CORPUS docs (the consumed count tracked via source_idx) and shuffles
        # the rest, so resumed is exactly the corpus tail and never re-reads a
        # consumed corpus doc.
        self.assertEqual(set(fresh), set(texts))
        self.assertEqual(set(resumed), set(texts[15:]))
        self.assertTrue(set(resumed).isdisjoint(set(texts[:15])))

    def test_packed_stream_source_idx_round_trips_to_consumed(self):
        # End-to-end resume math through PackedStream: base_offset=consumed,
        # source_idx = consumed + local_idx, so the max source_idx + 1 is the
        # cumulative post-val docs consumed that the NEXT run skips.
        texts = _docs(30)
        consumed = 8
        ps = train.PackedStream(
            _iterable(texts[consumed:]), _CharTokenizer(), block_size=8,
            base_offset=consumed, emit_source_idx=True,
        )
        seen = list(ps)
        max_idx = max(int(c["source_idx"]) for c in seen)
        # All docs were short+uniform, so the final block is fed by the last
        # doc; consumed-count = max_idx + 1 must land within the corpus.
        self.assertGreaterEqual(max_idx + 1, consumed)
        self.assertLessEqual(max_idx + 1, len(texts))


class CheckpointRoundTripTest(unittest.TestCase):
    def test_stream_docs_consumed_survives_save_load(self):
        # Use the real save_checkpoint / read_stream_docs_consumed with a tiny
        # dummy model + optimizer so the on-disk dict plumbing is exercised.
        model = torch.nn.Linear(4, 4)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        cpu = torch.device("cpu")

        with tempfile.TemporaryDirectory() as d:
            save_dir = Path(d)
            # Pre-create config.json so save_checkpoint doesn't try to
            # serialise inner.config (the dummy Linear has none).
            save_dir.mkdir(parents=True, exist_ok=True)
            (save_dir / "config.json").write_text("{}")
            path = train.save_checkpoint(
                model, optimizer, scheduler=None,
                epoch=123, metrics={"loss": 1.0}, save_dir=save_dir,
                stream_docs_consumed=4567,
            )
            self.assertEqual(train.read_stream_docs_consumed(path, cpu), 4567)

            # load_resume_checkpoint still returns the step and restores state.
            step = train.load_resume_checkpoint(
                path, model, optimizer, None, None, None, cpu,
            )
            self.assertEqual(step, 123)

    def test_read_stream_docs_consumed_defaults_zero(self):
        # Old checkpoints predate the field -> resume from the stream head (0).
        model = torch.nn.Linear(2, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        cpu = torch.device("cpu")
        with tempfile.TemporaryDirectory() as d:
            save_dir = Path(d)
            save_dir.mkdir(parents=True, exist_ok=True)
            (save_dir / "config.json").write_text("{}")
            path = train.save_checkpoint(
                model, optimizer, scheduler=None,
                epoch=1, metrics={"loss": 1.0}, save_dir=save_dir,
                stream_docs_consumed=None,
            )
            self.assertEqual(train.read_stream_docs_consumed(path, cpu), 0)


class BuildStreamingLoadersIntegrationTest(unittest.TestCase):
    def _build(self, texts, **arg_overrides):
        args = _make_args(max_length=8, batch_size=2, num_workers=0,
                          val_docs=4, **arg_overrides)
        # load_dataset is called once for val (full stream) and once for the
        # train stream; both must return a FRESH iterable.
        with mock.patch.object(
            train, "load_dataset", side_effect=lambda **kw: _iterable(texts),
        ):
            return train.build_streaming_loaders(args, _CharTokenizer())

    def test_train_emits_source_idx_val_does_not(self):
        texts = _docs(40)
        train_loader, val_loader = self._build(texts)

        train_batch = next(iter(train_loader))
        self.assertIn("source_idx", train_batch)
        self.assertEqual(train_batch["source_idx"].shape[0],
                         train_batch["input_ids"].shape[0])

        val_batch = next(iter(val_loader))
        self.assertNotIn("source_idx", val_batch,
                         "eval path must never see source_idx")

    def test_loop_pop_tracks_consumed_and_resume_is_disjoint(self):
        # Mirror the run_step_training pop: read a few train batches, pop
        # source_idx, track docs_consumed = max(source_idx)+1. Then rebuild
        # with stream_docs_consumed = that count and confirm the resumed train
        # stream never re-reads a consumed corpus doc.
        texts = _docs(40)
        train_loader, _ = self._build(texts)

        docs_consumed = 0
        seen_texts = set()
        it = iter(train_loader)
        tok = _CharTokenizer()
        for _ in range(3):
            batch = next(it)
            src = batch.pop("source_idx")
            docs_consumed = max(docs_consumed, int(src.max()) + 1)
            # Recover which corpus docs the packed ids came from is awkward;
            # instead assert the bookkeeping count is a sane corpus offset.
        self.assertGreater(docs_consumed, 0)
        self.assertLessEqual(docs_consumed, len(texts))

        # Resume: skip the consumed corpus docs (past the val hold-out).
        resumed_loader, _ = self._build(texts, stream_docs_consumed=docs_consumed)
        # The resumed train stream's first source_idx must be >= docs_consumed,
        # i.e. it starts past everything the first run consumed.
        rbatch = next(iter(resumed_loader))
        self.assertGreaterEqual(int(rbatch["source_idx"].min()), docs_consumed)


class _FakeXm:
    """Minimal xm stub: rank 0 of a single-replica world (CPU, no torch_xla)."""

    def get_ordinal(self):
        return 0

    def xrt_world_size(self):
        return 1


class XlaStreamingLoaderTest(unittest.TestCase):
    def test_xla_train_emits_source_idx_val_does_not(self):
        # The XLA streaming path (separate builder) must get the same fix:
        # shuffle + resumable skip + train-only source_idx.
        texts = _docs(40)
        args = _make_args(val_docs=4, max_length=8, batch_size=2,
                          num_workers=0, stream_shuffle_buffer=40,
                          stream_docs_consumed=0,
                          token_superposition_bag_size=1,
                          token_superposition_ratio=0.0)
        with mock.patch.object(
            train_xla, "load_dataset", side_effect=lambda **kw: _iterable(texts),
        ):
            train_loader, val_loader, phase1 = train_xla._build_streaming_loaders_xla(
                args, _CharTokenizer(), _FakeXm(), rank=0, world_size=1,
            )
        self.assertIn("source_idx", next(iter(train_loader)))
        self.assertNotIn("source_idx", next(iter(val_loader)))
        self.assertIsNone(phase1, "TST disabled => no phase-1 loader")

    def test_xla_save_checkpoint_persists_and_reads_back(self):
        # save_checkpoint_xla writes via xm.save; on CPU xm.save == torch.save.
        model = torch.nn.Linear(4, 4)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        cpu = torch.device("cpu")

        class _SaveXm(_FakeXm):
            def save(self, obj, path, **kw):
                torch.save(obj, path)

            def rendezvous(self, *a, **k):
                pass

        with tempfile.TemporaryDirectory() as d:
            save_dir = Path(d)
            save_dir.mkdir(parents=True, exist_ok=True)
            (save_dir / "config.json").write_text("{}")
            path = train_xla.save_checkpoint_xla(
                _SaveXm(), model, optimizer, None,
                epoch=7, metrics={"loss": 1.0}, save_dir=save_dir,
                stream_docs_consumed=9999,
            )
            self.assertEqual(train.read_stream_docs_consumed(path, cpu), 9999)


class ResolveResumePathTest(unittest.TestCase):
    def test_resolve_resume_path_variants(self):
        with tempfile.TemporaryDirectory() as d:
            save_dir = Path(d)
            self.assertIsNone(train.resolve_resume_path("none", save_dir))
            # auto with no checkpoints -> None
            self.assertIsNone(train.resolve_resume_path("auto", save_dir))
            with self.assertRaises(FileNotFoundError):
                train.resolve_resume_path(str(save_dir / "missing.pt"), save_dir)
            # auto picks the highest checkpoint_epoch_{N}.pt
            (save_dir / "checkpoint_epoch_001.pt").write_bytes(b"x")
            (save_dir / "checkpoint_epoch_005.pt").write_bytes(b"x")
            picked = train.resolve_resume_path("auto", save_dir)
            self.assertEqual(picked.name, "checkpoint_epoch_005.pt")


if __name__ == "__main__":
    unittest.main()
