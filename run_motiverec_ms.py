"""Multi-thought MotiveRec (model/MotiveRecMS.py) via run_collabrec.py's loop, unchanged.

Extra flag: --thought_support split|shared|first (see the model docstring), stored in the saved config and checked on
reload. All other flags are run_collabrec.py's: --reason_step S = number of thoughts, --support_k K = motives.
"""
import argparse
import sys

import run_collabrec
from model.MotiveRecMS import MotiveRecMS, THOUGHT_SUPPORT

pre = argparse.ArgumentParser(add_help=False)
pre.add_argument("--thought_support", choices=THOUGHT_SUPPORT, default="split")
pre.add_argument("--prl_weight", type=float, default=0.0)  # progressive step loss weight (0 = off)
pre.add_argument("--prl_alpha", type=float, default=1.5)   # step k temperature = alpha ** (S - k)
known, rest = pre.parse_known_args()
sys.argv = [sys.argv[0]] + rest
MODE = known.thought_support


class _Loader:
    @staticmethod
    def from_pretrained(path, config=None, **kwargs):
        saved = getattr(config, "thought_support", None)
        assert saved in (None, MODE), f"checkpoint thought_support={saved}, flag --thought_support={MODE}"
        config.thought_support = MODE
        if "--train_or_test" not in rest or rest[rest.index("--train_or_test") + 1] == "train":
            config.prl_weight, config.prl_alpha = known.prl_weight, known.prl_alpha
        print(f"MotiveRecMS: prl_weight={getattr(config, 'prl_weight', 0.0)} prl_alpha={getattr(config, 'prl_alpha', 1.5)}", flush=True)
        print(f"MotiveRecMS: thought_support={MODE}", flush=True)
        return MotiveRecMS.from_pretrained(path, config=config, **kwargs)


run_collabrec.CollabRecF2 = _Loader
run_collabrec.MODEL_NAME = "MotiveRecMS"

if __name__ == "__main__":
    run_collabrec.run()
