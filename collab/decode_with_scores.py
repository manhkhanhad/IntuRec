#!/usr/bin/env python3
"""Decode a saved CollabRec checkpoint on the test split with a chosen retriever's top-k motives.

The supports are the top --support_k prototypes of motive_corpus/retriever/<scores>/scores_test.npy
(motive_source "predicted"; the soft readout also gets their log-probabilities). The readout (inject /
gated / soft) comes from the checkpoint's config. Output: <run>/result/standard/retest_<scores>_top<k>/
meta.json, plus Recall / NDCG @1,5,10 in the log; existing results are never overwritten.

Usage: python collab/decode_with_scores.py --run_dir <run folder> --scores mlp_cand --support_k 10
"""
import argparse
import glob
import os
import sys
from types import SimpleNamespace

import torch
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.CollabRecDataset import CollabRecDataset  # noqa: E402
from model.CollabRecF2 import CollabRecF2  # noqa: E402
from run_collabrec import evaluate  # noqa: E402
from utils.util import calRecallAndNdcgFromFile  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run_dir", required=True)
    p.add_argument("--scores", required=True, help="retriever score set, e.g. cls or mlp_cand")
    p.add_argument("--support_k", type=int, default=10)
    p.add_argument("--dataset", default="Toys_and_Games")
    p.add_argument("--data_path", default="./data")
    p.add_argument("--eval_batch_size", type=int, default=8)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--sample", type=int, default=-1, help="first N test rows only (smoke tests)")
    a = p.parse_args()

    out_dir = os.path.join(a.run_dir, "result", "standard", f"retest_{a.scores}_top{a.support_k}")
    if os.path.exists(os.path.join(out_dir, "meta.json")) and a.sample < 0:
        print(f"exists, skipping: {out_dir}/meta.json", flush=True)
        return
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device(a.device if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(a.run_dir)
    model = CollabRecF2.from_pretrained(a.run_dir, torch_dtype=torch.bfloat16).to(device).eval()
    model.reason_step, model.beta, model.gamma, model.sel_weight = 1, 0.0, 0.0, 0.0
    print(f"run {a.run_dir}\nreadout {getattr(model.config, 'readout', 'inject')}, scores {a.scores}, "
          f"top-{a.support_k}", flush=True)

    ds = CollabRecDataset(a.data_path, a.dataset, tokenizer, support_dir="", support_k=a.support_k,
                          dataset_type="test", reason_step=1, seed=0, support_mode="text",
                          motive_source="predicted", motive_scores=a.scores)
    if a.sample > 0:
        ds.data, ds.data_output, ds.row_ids = ds.data[:a.sample], ds.data_output[:a.sample], ds.row_ids[:a.sample]
    args = SimpleNamespace(topk=10, length_penalty=1.0, temperature=1.0, eval_batch_size=a.eval_batch_size,
                           data_result_meta_dir_path=out_dir,
                           partial_every=0 if a.sample > 0 else 25)  # resumable after preemption
    info = glob.glob(os.path.join(a.data_path, a.dataset, "info", "*.txt"))
    assert len(info) == 1
    path = evaluate(model, tokenizer, ds, info[0], args)
    calRecallAndNdcgFromFile(path, info[0], ans_col_name="tgt_item")


if __name__ == "__main__":
    main()
