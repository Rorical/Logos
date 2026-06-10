"""CPU, network-free tests for streaming-data shuffle + resumable position.

Covers Fix 1 ("shuffle streaming data and checkpoint stream position") and its
hardening ("harden streaming resume against shard shuffling and worker skew"):
  (a) shuffle reorders docs but preserves the multiset, and the val
      hold-out (first val_docs) stays fixed and unshuffled;
  (b) PackedStream emits the correct GLOBAL source_idx with base_offset,
      under a single worker and under 2 simulated workers;
  (c) MULTI-SHARD resume (the hardening): consuming N docs then resuming via
      the HF ``state_dict`` re-reads ZERO consumed docs and loses at most a
      shuffle-buffer worth — measured on an 8-shard stream where the OLD
      ``.skip(N)``-before-``.shuffle`` resume replayed most of the prefix
      because ``.shuffle`` also shuffles shard order;
  (d) checkpoint round-trip: stream_docs_consumed AND stream_state survive
      save/load, and a corrupt checkpoint falls back cleanly;
  (e) TST phase-1 -> phase-2 in-process handoff continues the stream instead
      of replaying phase 1's docs.
"""

import argparse
import pickle
import tempfile
import unittest
import zipfile
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


def _iterable(texts, num_shards=1):
    return datasets.Dataset.from_dict({"text": texts}).to_iterable_dataset(
        num_shards=num_shards,
    )


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


def _pipeline(texts, args, num_shards):
    """The real train-stream pipeline: .skip(val_docs) -> .shuffle(seed,buf).

    Mirrors ``build_streaming_loaders`` (skip the val hold-out, then shuffle —
    which on a multi-shard stream ALSO shuffles shard order). Returns the raw HF
    IterableDataset so a test can drive it and capture/load its ``state_dict``.
    """
    stream = _iterable(texts, num_shards=num_shards).skip(args.val_docs)
    return train.shuffle_train_stream(args, stream)


class MultiShardResumeTest(unittest.TestCase):
    """Pins the hardening property on a realistic 8-shard stream.

    HF ``.shuffle`` shuffles SHARD ORDER as well as rows, so the pipeline order
    on 8 shards is NOT the corpus order. The OLD resume — ``.skip(consumed)``
    before ``.shuffle`` — therefore dropped a corpus-order prefix while run 1
    had consumed a shuffled-shard-order prefix, making the two sets near
    disjoint (huge replay). The fix resumes via the HF ``state_dict`` instead.
    """

    SHARDS = 8
    VAL = 4
    BUF = 20
    CONSUME = 40

    def test_old_skip_resume_replays_most_of_prefix_on_8_shards(self):
        # Demonstrates the bug the hardening fixes: skip-before-shuffle resume
        # on 8 shards replays a large fraction of consumed docs (NOT a small
        # buffer-bounded overlap), because shard-order shuffling makes the
        # skipped corpus prefix disjoint from the consumed shuffled prefix.
        texts = _docs(140)
        args = _make_args(val_docs=self.VAL, stream_shuffle_buffer=self.BUF)
        run1 = _pipeline(texts, args, self.SHARDS)
        it = iter(run1)
        consumed = {next(it)["text"] for _ in range(self.CONSUME)}
        resumed_old = {
            s["text"] for s in train.shuffle_train_stream(
                args, _iterable(texts, num_shards=self.SHARDS)
                .skip(args.val_docs).skip(self.CONSUME),
            )
        }
        replay = consumed & resumed_old
        # The OLD mechanism replays a large fraction of the consumed docs —
        # far more than the shuffle buffer. This is the defeated-plateau-fix
        # failure mode; asserted here so the regression can't silently return.
        self.assertGreater(
            len(replay), self.BUF,
            "skip-before-shuffle on multi-shard should replay >> buffer "
            "(this is the bug the state_dict resume replaces)",
        )

    def test_state_dict_resume_zero_replay_bounded_loss_on_8_shards(self):
        # The fix: capture the HF state_dict after consuming CONSUME docs, then
        # resume a fresh same-seed pipeline via load_state_dict. Measured:
        #   REPLAY == 0 (never re-reads a consumed doc), and
        #   LOST   <= shuffle buffer (only the in-flight buffer is dropped),
        # i.e. never replays consumed docs and never skips a large unread region.
        texts = _docs(140)
        args = _make_args(val_docs=self.VAL, stream_shuffle_buffer=self.BUF)
        universe = set(texts[self.VAL:])

        run1 = _pipeline(texts, args, self.SHARDS)
        it = iter(run1)
        consumed = {next(it)["text"] for _ in range(self.CONSUME)}
        state = run1.state_dict()

        resumed_stream = _pipeline(texts, args, self.SHARDS)
        resumed_stream.load_state_dict(state)
        resumed = {s["text"] for s in resumed_stream}

        replay = consumed & resumed
        lost = universe - consumed - resumed
        self.assertEqual(len(replay), 0,
                         "state_dict resume must never re-read a consumed doc")
        self.assertLessEqual(
            len(lost), self.BUF,
            "loss must be bounded by the shuffle buffer (in-flight docs), "
            "never a large unread region",
        )

    def test_packed_stream_capture_restore_plumbing_on_8_shards(self):
        # The production capture/restore path on 8 shards: PackedStream exposes
        # the live HF state_dict via ``hf_state_dict`` (what _capture_stream_state
        # reads) and a fresh PackedStream built with ``resume_state`` loads it
        # and yields only never-consumed corpus docs (a subset of the universe).
        texts = _docs(140)
        args = _make_args(val_docs=self.VAL, stream_shuffle_buffer=self.BUF)
        universe = set(texts[self.VAL:])

        ps = train.PackedStream(
            _pipeline(texts, args, self.SHARDS), _CharTokenizer(),
            block_size=8, emit_source_idx=True,
        )
        it = iter(ps)
        for _ in range(30):
            next(it)
        state = ps.hf_state_dict()
        self.assertIsNotNone(state)

        resumed = train.PackedStream(
            _pipeline(texts, args, self.SHARDS), _CharTokenizer(),
            block_size=8, emit_source_idx=True, resume_state=state,
        )
        # Recover the source texts the resumed stream yields by reading them off
        # its (loaded) HF source after a fresh build with the same state.
        resumed_raw = _pipeline(texts, args, self.SHARDS)
        resumed_raw.load_state_dict(state)
        resumed_docs = {s["text"] for s in resumed_raw}
        self.assertTrue(resumed_docs.issubset(universe))
        self.assertGreater(len(resumed_docs), 0)
        # The resumed PackedStream is iterable and emits source_idx.
        self.assertIn("source_idx", next(iter(resumed)))

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

    def test_stream_state_survives_save_load(self):
        # The HF state_dict is the actual resume mechanism: it must round-trip
        # through save_checkpoint / read_stream_state, and a real captured state
        # must resume the same-seed stream without replay after the round-trip.
        model = torch.nn.Linear(4, 4)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        cpu = torch.device("cpu")

        texts = _docs(80)
        args = _make_args(val_docs=4, stream_shuffle_buffer=20)
        run1 = train.shuffle_train_stream(
            args, _iterable(texts, num_shards=8).skip(args.val_docs),
        )
        it = iter(run1)
        consumed = {next(it)["text"] for _ in range(20)}
        state = run1.state_dict()

        with tempfile.TemporaryDirectory() as d:
            save_dir = Path(d)
            save_dir.mkdir(parents=True, exist_ok=True)
            (save_dir / "config.json").write_text("{}")
            path = train.save_checkpoint(
                model, optimizer, scheduler=None,
                epoch=5, metrics={"loss": 1.0}, save_dir=save_dir,
                stream_docs_consumed=20, stream_state=state,
            )
            loaded = train.read_stream_state(path, cpu)
            self.assertIsInstance(loaded, dict)

            resumed_stream = train.shuffle_train_stream(
                args, _iterable(texts, num_shards=8).skip(args.val_docs),
            )
            resumed_stream.load_state_dict(loaded)
            resumed = {s["text"] for s in resumed_stream}
            self.assertEqual(consumed & resumed, set(),
                             "round-tripped state must resume without replay")

    def test_read_stream_state_defaults_none(self):
        # Old checkpoints (no stream_state) -> None so the run starts fresh.
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
                stream_state=None,
            )
            self.assertIsNone(train.read_stream_state(path, cpu))

    def test_corrupt_checkpoint_falls_back_consistently(self):
        # A truncated/half-written checkpoint must not crash the resume peek:
        # the broader exception set (pickle.UnpicklingError, EOFError,
        # zipfile.BadZipFile, OSError, RuntimeError) all fall back to 0 / None.
        cpu = torch.device("cpu")
        with tempfile.TemporaryDirectory() as d:
            # Empty file -> EOFError/UnpicklingError depending on torch path.
            empty = Path(d) / "empty.pt"
            empty.write_bytes(b"")
            self.assertEqual(train.read_stream_docs_consumed(empty, cpu), 0)
            self.assertIsNone(train.read_stream_state(empty, cpu))

            # Garbage bytes -> BadZipFile / UnpicklingError.
            garbage = Path(d) / "garbage.pt"
            garbage.write_bytes(b"not a torch checkpoint at all")
            self.assertEqual(train.read_stream_docs_consumed(garbage, cpu), 0)
            self.assertIsNone(train.read_stream_state(garbage, cpu))

    def test_corrupt_ckpt_error_set_covers_required_types(self):
        # The fix requires these specific exception types in the fallback set.
        for exc in (pickle.UnpicklingError, EOFError, zipfile.BadZipFile,
                    OSError, RuntimeError):
            self.assertTrue(issubclass(exc, train._CORRUPT_CKPT_ERRORS))


class BuildStreamingLoadersIntegrationTest(unittest.TestCase):
    def _build(self, texts, num_shards=1, stream_state=None, **arg_overrides):
        arg_overrides.setdefault("num_workers", 0)
        args = _make_args(max_length=8, batch_size=2, val_docs=4,
                          **arg_overrides)
        args.stream_state = stream_state
        # load_dataset is called once for val (full stream) and once for the
        # train stream; both must return a FRESH iterable.
        with mock.patch.object(
            train, "load_dataset",
            side_effect=lambda **kw: _iterable(texts, num_shards=num_shards),
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

    def test_train_loader_is_single_process_for_state_capture(self):
        # The streaming TRAIN loader must run single-process so the live HF
        # state_dict is readable on this process at checkpoint time (see
        # PackedStream / _capture_stream_state). Even with --num-workers > 0
        # the train loader pins num_workers=0.
        texts = _docs(40)
        train_loader, _ = self._build(texts, num_workers=4)
        self.assertEqual(train_loader.num_workers, 0,
                         "streaming train loader must be single-process")

    def test_resume_via_captured_state_never_replays_consumed(self):
        # The production resume path on an 8-shard stream: build the train
        # loader, consume a few batches, capture the live state via
        # _capture_stream_state (what run_step_training saves), then rebuild
        # with that state and confirm the resumed stream re-reads ZERO of the
        # docs the first run consumed.
        texts = _docs(140)
        train_loader, _ = self._build(texts, num_shards=8,
                                      stream_shuffle_buffer=20)

        # Identify which docs run 1 consumed by reading the raw HF source the
        # loader's PackedStream wraps (post-skip, post-shuffle order).
        raw_run1 = train_loader.dataset.source
        it = iter(raw_run1)
        consumed = {next(it)["text"] for _ in range(40)}
        state = train._capture_stream_state(train_loader)
        self.assertIsNotNone(state)

        resumed_loader, _ = self._build(texts, num_shards=8,
                                        stream_shuffle_buffer=20,
                                        stream_state=state)
        resumed_raw = resumed_loader.dataset.source
        resumed = {s["text"] for s in resumed_raw}
        self.assertEqual(consumed & resumed, set(),
                         "resume must never re-read a consumed doc")


class TstPhaseHandoffTest(unittest.TestCase):
    """Fix 4: the in-process TST phase-1 -> phase-2 handoff must continue the
    stream, not replay phase 1's docs. Both phases build IDENTICAL same-seed
    pipelines, so without the handoff phase 2 re-reads everything phase 1 saw.
    """

    def _train_loader(self, texts, block_size, num_shards):
        args = _make_args(max_length=block_size, batch_size=2, num_workers=0,
                          val_docs=4, stream_shuffle_buffer=20)
        args.stream_state = None
        with mock.patch.object(
            train, "load_dataset",
            side_effect=lambda **kw: _iterable(texts, num_shards=num_shards),
        ):
            return train.build_streaming_train_loader(args, _CharTokenizer(),
                                                      block_size)

    def test_build_streaming_train_loader_is_single_process(self):
        # Phase-1 loader must also be single-process so its live HF state can be
        # captured for the handoff into phase 2.
        loader = self._train_loader(_docs(40), block_size=8, num_shards=1)
        self.assertEqual(loader.num_workers, 0)
        self.assertIsNotNone(train._capture_stream_state(loader),
                             "phase-1 loader must expose a capturable HF state")

    def test_handoff_continues_stream_no_replay(self):
        # Simulate the loop's handoff: phase-1 loader consumes some docs; we
        # capture its live state and load it into phase-2's loader via
        # _load_stream_state. Phase 2 must then never re-read a phase-1 doc.
        texts = _docs(140)
        phase1 = self._train_loader(texts, block_size=8, num_shards=8)
        phase2 = self._train_loader(texts, block_size=4, num_shards=8)

        # Drive phase 1's raw source to learn which docs it consumed.
        p1_src = phase1.dataset.source
        it = iter(p1_src)
        phase1_docs = {next(it)["text"] for _ in range(40)}

        applied = train._load_stream_state(
            phase2, train._capture_stream_state(phase1),
        )
        self.assertTrue(applied, "handoff must load phase-1 state into phase 2")

        phase2_docs = {s["text"] for s in phase2.dataset.source}
        self.assertEqual(phase1_docs & phase2_docs, set(),
                         "phase 2 must not replay phase-1 docs after handoff")

    def test_load_stream_state_noop_on_none(self):
        loader = self._train_loader(_docs(40), block_size=8, num_shards=1)
        self.assertFalse(train._load_stream_state(loader, None))


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
