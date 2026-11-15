"""model/IntuRecMS2.py (multi-step IntuRec, thoughts 2..S from the normalised RATT, model/RATTNorm.py) via
run_inturec_f2.py's loop. Extra flag: --ratt_residual 0|1 (default 1), stored in the saved config."""
import argparse
import sys

from transformers import AutoConfig

import run_inturec_f2
from model.IntuRecMS2 import IntuRecMS2

pre = argparse.ArgumentParser(add_help=False)
pre.add_argument("--ratt_residual", type=int, choices=(0, 1), default=1)
known, rest = pre.parse_known_args()
sys.argv = [sys.argv[0]] + rest
RES = known.ratt_residual


class _Loader:
    @staticmethod
    def from_pretrained(path, **kwargs):
        config = AutoConfig.from_pretrained(path, trust_remote_code=True)
        saved = getattr(config, "ratt_residual", None)
        assert saved in (None, RES), f"checkpoint ratt_residual={saved}, flag --ratt_residual={RES}"
        config.ratt_residual = RES
        model = IntuRecMS2.from_pretrained(path, config=config, **kwargs)
        print(f"IntuRecMS2: ratt_residual={RES}", flush=True)
        return model


run_inturec_f2.IntuRecF2 = _Loader
run_inturec_f2.MODEL_NAME = "IntuRecMS2"

if __name__ == "__main__":
    run_inturec_f2.run()
