"""CPU, network-free tests for W&B init mode resolution.

Covers the review fix: the notebooks' offline fallback sets
``os.environ['WANDB_MODE']='offline'`` when the Colab WANDB_API_KEY secret is
missing, but the run config still carries ``--wandb-mode online``. wandb gives
an explicit ``init(mode=...)`` kwarg precedence over the env var, so
``init_wandb`` must read ``WANDB_MODE`` from the environment (falling back to
``args.wandb_mode``) and pass THAT, otherwise the offline fallback is silently
ignored and a keyless online run blocks on an interactive login.
"""

import os
import sys
import types
import unittest
from types import SimpleNamespace
from unittest import mock

import scripts.train as train


def _wandb_args(wandb_mode="online"):
    return SimpleNamespace(
        wandb=True,
        wandb_project="p",
        wandb_entity=None,
        wandb_run_name=None,
        wandb_tags=None,
        wandb_mode=wandb_mode,
        model="logos",
    )


class _FakeWandb(types.ModuleType):
    def __init__(self):
        super().__init__("wandb")
        self.captured_mode = None

    def init(self, **kwargs):
        self.captured_mode = kwargs.get("mode")
        return SimpleNamespace(**kwargs)


class InitWandbModeTest(unittest.TestCase):
    def _run(self, args, env):
        fake = _FakeWandb()
        with mock.patch.dict(sys.modules, {"wandb": fake}), \
                mock.patch.dict(os.environ, env, clear=False):
            os.environ.pop("WANDB_MODE", None)
            if "WANDB_MODE" in env:
                os.environ["WANDB_MODE"] = env["WANDB_MODE"]
            train.init_wandb(args)
        return fake.captured_mode

    def test_env_offline_overrides_online_arg(self):
        # The fallback path: arg says online, env says offline -> offline wins.
        mode = self._run(_wandb_args("online"), {"WANDB_MODE": "offline"})
        self.assertEqual(mode, "offline")

    def test_arg_used_when_env_unset(self):
        # No env override: the CLI default is honoured.
        mode = self._run(_wandb_args("online"), {})
        self.assertEqual(mode, "online")


if __name__ == "__main__":
    unittest.main()
