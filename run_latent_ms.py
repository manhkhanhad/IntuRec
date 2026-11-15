"""Multi-step naive latent recommender (model/LatentMS.py) via run_inturec_f2.py's loop, unchanged.

The intuition inputs IntuRec's dataset provides are ignored by the model (--intuition_num and --alpha are
unused). Extra flags (stored in the saved config):
  --first_thought last|ratt   how thought 1 is built (default last)
  --prefix_weight  lambda on the prefix NTP losses L_1..L_{S-1} (default 1.0)
  --mono_weight    mu on the monotonicity penalty (default 1.0; 0 = off)
  --mono_margin    m in relu(L_t - sg(L_{t-1}) + m) (default 0.02 nats/token)
"""
import argparse
import sys

from transformers import AutoConfig

import run_inturec_f2
from model.LatentMS import LatentMS, FIRST_THOUGHT

pre = argparse.ArgumentParser(add_help=False)
pre.add_argument("--first_thought", choices=FIRST_THOUGHT, default="last")
pre.add_argument("--prefix_weight", type=float, default=1.0)
pre.add_argument("--mono_weight", type=float, default=1.0)
pre.add_argument("--mono_margin", type=float, default=0.02)
known, rest = pre.parse_known_args()
sys.argv = [sys.argv[0]] + rest
EXTRA = vars(known)


class _LatentMSLoader:
    @staticmethod
    def from_pretrained(path, **kwargs):
        config = AutoConfig.from_pretrained(path, trust_remote_code=True)
        for k, v in EXTRA.items():
            saved = getattr(config, k, None)
            assert saved in (None, v), f"checkpoint {k}={saved}, flag --{k}={v}"
            setattr(config, k, v)
        model = LatentMS.from_pretrained(path, config=config, **kwargs)
        print(f"LatentMS: {EXTRA}", flush=True)
        return model


run_inturec_f2.IntuRecF2 = _LatentMSLoader
run_inturec_f2.MODEL_NAME = "LatentMS"

if __name__ == "__main__":
    run_inturec_f2.run()
