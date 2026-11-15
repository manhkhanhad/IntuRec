"""model/LatentMS2.py (LatentMS with the normalised RATT, model/RATTNorm.py) via run_inturec_f2.py's loop.

Flags: those of run_latent_ms.py (--prefix_weight, --mono_weight, --mono_margin; --first_thought must stay
last) plus --ratt_residual 0|1 (default 1). All are stored in the saved config and checked on reload.
"""
import argparse
import sys

pre = argparse.ArgumentParser(add_help=False)
pre.add_argument("--ratt_residual", type=int, choices=(0, 1), default=1)
known, rest = pre.parse_known_args()
sys.argv = [sys.argv[0]] + rest

import run_inturec_f2  # noqa: E402
import run_latent_ms  # noqa: E402  (parses its own flags, installs its loader)
from model.LatentMS2 import LatentMS2  # noqa: E402

run_latent_ms.EXTRA["ratt_residual"] = known.ratt_residual
run_latent_ms.LatentMS = LatentMS2
run_inturec_f2.MODEL_NAME = "LatentMS2"

if __name__ == "__main__":
    run_inturec_f2.run()
