#!/usr/bin/env python3
"""Save the baseline motive picks for the Toys test split (collab/select_picks/):
retriever top-1 and the oracle best of the retriever's top-10 (by stored utility)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from grpo_motive_select import JUDGES, load_utilities  # noqa: E402

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "select_picks")
os.makedirs(out, exist_ok=True)
cand, u, _ = load_utilities(JUDGES["Toys_and_Games"]["test"], 6738)
np.save(os.path.join(out, "toys_test_retriever_top1.npy"), cand[:, 0])
np.save(os.path.join(out, "toys_test_oracle_best10.npy"), cand[np.arange(len(cand)), np.nanargmax(u, 1)])
g = os.path.join(out, "toys_test_grpo_b0_s0.npy")
if os.path.exists(g):
    print("GRPO picks equal to retriever top-1:", float((np.load(g) == cand[:, 0]).mean()))
print("saved to", out)
