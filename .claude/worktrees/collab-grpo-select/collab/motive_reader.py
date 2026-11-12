#!/usr/bin/env python3
"""Run a frozen motive-reading CollabRec model (e.g. O2 + InfoNCE, run 351001) with chosen motives.

Uses the main checkout's CollabRec code (model/CollabRecF2.py, data/CollabRecDataset.py,
run_collabrec.py), whose text-mode supports are support_dir/support_texts.json entries indexed by
support_dir/support_rows_<split>.npy. Motive texts are "<motive>: <reason>" per prototype plus
"No specific motive." (id = n_prototypes), and a single motive fills both slots as [m, m], which
is the input format of the utility judge (collab/motive_utility.py).

  write_supports(dir, split, ids)   ids [n] (one motive per row; n_prototypes = no motive) or [n, 2]
  python collab/motive_reader.py verify   CPU check: reproduce stored utilities for test row 0
  python collab/motive_reader.py decode --picks picks.npy --out_dir DIR   constrained top-10 decode
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

MAIN = "/common/home/users/m/mk.ngohuu.2024/research/IntuRec"
sys.path.insert(0, MAIN)
os.chdir(MAIN)  # the CollabRec code resolves ./data relative to the checkout
from data.CollabRecDataset import CollabRecDataset  # noqa: E402
from model.CollabRecF2 import CollabRecF2  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

CK = "/common/scratch/users/m/mk.ngohuu.2024/IntuRec/checkpoints"
READER = f"{CK}/Toys_and_Games/CollabRec_oracle/text-micro2-oracle_own_motive-cq0.1_user+target+motive_k2_beta0_rs1_lr0.0003_job351001"


def motive_texts(dataset):
    with open(f"{MAIN}/data/{dataset}/collab/motive_corpus/prototypes.json", encoding="utf-8") as f:
        return [f"{p['motive']}: {p['reason']}" for p in json.load(f)] + ["No specific motive."]


def write_supports(out_dir, dataset, split, ids):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "support_texts.json"), "w", encoding="utf-8") as f:
        json.dump(motive_texts(dataset), f)
    ids = np.asarray(ids, np.int64)
    if ids.ndim == 1:
        ids = np.stack([ids, ids], 1)
    np.save(os.path.join(out_dir, f"support_rows_{split}.npy"), ids)


def load_reader(path, device):
    tok = AutoTokenizer.from_pretrained(path)
    model = CollabRecF2.from_pretrained(path, torch_dtype=torch.bfloat16).to(device).eval()
    model.reason_step, model.beta, model.gamma = 1, 0.0, 0.0
    return model, tok


def dataset(tok, dataset_name, split, support_dir):
    return CollabRecDataset("./data", dataset_name, tok, support_dir=support_dir, support_k=2, dataset_type=split,
                            reason_step=1, seed=0, support_mode="text")


@torch.no_grad()
def target_ll(model, ds, i):
    """Teacher-forced log p(title + <|EndPredict|> | prompt, supports) of row i."""
    row = ds.data[i]
    target = ds.tokenizer.encode(row["tgt_item"], bos=False, eos=False) + ds.end_token
    s = ds[i]
    prompt = [t for t, m in zip(s["input_ids"].tolist(), s["attention_mask"].tolist()) if m]
    ids = torch.tensor([prompt + target], device=model.device)
    labels = ids.clone()
    labels[0, :len(prompt)] = -100
    out = model(input_ids=ids, attention_mask=torch.ones_like(ids),
                support_tokens=torch.as_tensor(np.asarray(s["support_tokens"]))[None].to(model.device), labels=labels)
    return -out.ntp_loss.item() * len(target)


def verify():
    tmp = os.path.join(os.environ.get("CLAUDE_JOB_DIR", "/tmp"), "tmp", "reader_verify")
    model, tok = load_reader(READER, "cpu")
    z = np.load(f"{READER}/motive_utility/utility_test.npz")
    sel = z["row"] == 0
    cases = [(int(m), float(ll), int(k), int(r)) for m, ll, k, r in zip(z["motive"][sel], z["ll"][sel], z["kind"][sel], z["rank"][sel])
             if k == 0 or (k == 3 and r in (0, 5))]
    for m, ll_ref, k, r in cases:
        ids = np.full(6738, 881)
        ids[0] = m
        write_supports(tmp, "Toys_and_Games", "test", ids)
        ds = dataset(tok, "Toys_and_Games", "test", tmp)
        print(f"kind {k} rank {r} motive {m}: ll {target_ll(model, ds, 0):.3f} (stored {ll_ref:.3f})", flush=True)


def decode(picks, out_dir, dataset_name, eval_batch_size, device, reader=READER):
    from run_collabrec import evaluate
    from utils.util import calRecallAndNdcgFromFile
    os.makedirs(out_dir, exist_ok=True)
    write_supports(os.path.join(out_dir, "supports"), dataset_name, "test", picks)
    model, tok = load_reader(reader, device)
    ds = dataset(tok, dataset_name, "test", os.path.join(out_dir, "supports"))
    info = glob.glob(f"./data/{dataset_name}/info/*.txt")[0]
    args = argparse.Namespace(topk=10, length_penalty=1.0, temperature=1.0, eval_batch_size=eval_batch_size,
                              data_result_meta_dir_path=out_dir)
    path = evaluate(model, tok, ds, info, args)
    calRecallAndNdcgFromFile(path, info, ans_col_name="tgt_item")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("cmd", choices=["verify", "decode"])
    p.add_argument("--picks", default="")
    p.add_argument("--out_dir", default="")
    p.add_argument("--reader", default=READER)
    p.add_argument("--dataset", default="Toys_and_Games")
    p.add_argument("--eval_batch_size", type=int, default=8)
    p.add_argument("--device", default="cuda:0")
    a = p.parse_args()
    if a.cmd == "verify":
        verify()
    else:
        decode(np.load(a.picks), a.out_dir, a.dataset, a.eval_batch_size,
               a.device if torch.cuda.is_available() else "cpu", a.reader)
