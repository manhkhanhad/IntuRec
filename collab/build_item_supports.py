#!/usr/bin/env python3
"""Item-grounded supports: each retrieved peer support states WHAT it bought and WHY.

Reuses the support rows of an existing stage-2 directory (default collab/stage2:
cross-fitted, target-balanced latent-hybrid retrieval over train rows) and writes a
support_texts.json over all train rows:  "<purchased item title>": <rationale>.
run_collabrec.py --support_mode text --support_dir <out> then pools these texts with
InjectNet. The purchased titles are other users' train purchases; target-matching
supports were already thinned to the test rate when the rows were built.
"""

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="CDs_and_Vinyl")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--rationale-tag", default="v5")
    parser.add_argument("--rows-dir", default="stage2", help="collab/ sub-folder holding support_rows_<split>.npy")
    parser.add_argument("--out-dir", default="stage2_item")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    collab = args.data_root / args.dataset / "collab"
    out = collab / args.out_dir
    out.mkdir(exist_ok=True)
    train = pd.read_csv(next((args.data_root / args.dataset / "train").glob(f"{args.dataset}*.csv")))
    reasons = [None] * len(train)
    with (collab / f"rationales_facet_train_{args.rationale_tag}.jsonl").open(encoding="utf-8") as fin:
        for line in fin:
            rec = json.loads(line)
            reasons[rec["row"]] = rec["reason"]
    assert all(r is not None for r in reasons), "rationales must cover every train row"
    texts = [f"\"{title}\": {reason}" for title, reason in zip(train.item_title, reasons)]
    (out / "support_texts.json").write_text(json.dumps(texts))
    for split in ("train", "valid", "test"):
        shutil.copy(collab / args.rows_dir / f"support_rows_{split}.npy", out / f"support_rows_{split}.npy")
    print(f"wrote {len(texts)} support texts and copied support rows to {out}")
    print("example:", texts[0])


if __name__ == "__main__":
    main()
