"""Multi-step IntuRec runner: run_inturec_f2.py unchanged, with model.IntuRecMS in place of IntuRecF2.

Extra flag: --step_mode copy|ratt (how thoughts 2..S are built; see model/IntuRecMS.py). All other flags
are run_inturec_f2.py's (--reason_step S sets the number of thought tokens). step_mode is stored in the
saved config; in test mode the saved value must match the flag.
"""
import argparse
import sys

from transformers import AutoConfig

import run_inturec_f2
from model.IntuRecMS import IntuRecMS, STEP_MODES

pre = argparse.ArgumentParser(add_help=False)
pre.add_argument("--step_mode", choices=STEP_MODES, default="copy")
known, rest = pre.parse_known_args()
sys.argv = [sys.argv[0]] + rest
STEP_MODE = known.step_mode


class _IntuRecMSLoader:
    @staticmethod
    def from_pretrained(path, **kwargs):
        config = AutoConfig.from_pretrained(path, trust_remote_code=True)
        saved = getattr(config, "step_mode", None)
        assert saved in (None, STEP_MODE), f"checkpoint step_mode={saved}, flag --step_mode={STEP_MODE}"
        config.step_mode = STEP_MODE
        model = IntuRecMS.from_pretrained(path, config=config, **kwargs)
        print(f"IntuRecMS: step_mode={STEP_MODE}")
        return model


run_inturec_f2.IntuRecF2 = _IntuRecMSLoader
run_inturec_f2.MODEL_NAME = "IntuRecMS"

if __name__ == "__main__":
    run_inturec_f2.run()
