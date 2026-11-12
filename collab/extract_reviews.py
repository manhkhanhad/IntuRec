#!/usr/bin/env python3
"""Attach each interaction's own Amazon review to the split CSV rows.

Writes ``<data_root>/<dataset>/collab/reviews_<split>.json``: a list aligned
with the CSV row order, holding the review summary, text and an
``informative`` flag (long enough and not a bare star-rating phrase,
or naming a concrete motive keyword such as gift or soundtrack).
"""

import argparse
import gzip
import json
import re
from pathlib import Path

import pandas as pd

SPLITS = ("train", "valid", "test")
GENERIC_SUMMARY = re.compile(r"^\s*(one|two|three|four|five)\s+stars?\s*$", re.I)
# Short reviews that still name a concrete purchase motive count as informative.
MOTIVE_KEYWORDS = re.compile(
    r"\b(gifts?|present|birthday|christmas|holiday|son|daughter|wife|husband|grand(son|daughter|kids?)|"
    r"soundtrack|movie|film|musical|broadway|vinyl|lp|pressing|remaster(ed)?|mono|stereo|audio|"
    r"packaging|live|concert|price|cheap|deal|bargain|replace(ment|d)?|lost|rare|out of print|cover art)\b",
    re.I)
# Extra concrete-motive keywords for Toys_and_Games (--domain toys).
TOY_KEYWORDS = re.compile(
    r"\b(years? old|toddler|baby|kids?|nephew|niece|stocking|party|educational|learn(ing|s)?|stem|"
    r"collection|collect(or|ing)?|set|series|missing|broke(n)?|expansion|family|game night)\b",
    re.I)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="CDs_and_Vinyl")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--review-file", type=Path, default=None,
                        help="Default: <data_root>/<dataset>/<dataset>.json.gz, else <dataset>.json (plain JSON lines)")
    parser.add_argument("--domain", choices=["music", "toys"], default="music")
    parser.add_argument("--min-words", type=int, default=12,
                        help="Minimum review-text words for informative=True (default: 12).")
    return parser.parse_args()


def find_csv(data_root: Path, dataset: str, split: str) -> Path:
    matches = sorted((data_root / dataset / split).glob(f"{dataset}*.csv"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one {dataset}*.csv in {data_root / dataset / split}")
    return matches[0]


def is_informative(summary: str, text: str, min_words: int, domain: str = "music") -> bool:
    long_enough = len(text.split()) >= min_words and not (
        GENERIC_SUMMARY.match(summary) and len(text.split()) < 2 * min_words
    )
    keyword = MOTIVE_KEYWORDS.search(f"{summary} {text}") or (
        domain == "toys" and TOY_KEYWORDS.search(f"{summary} {text}"))
    return long_enough or bool(keyword)


def main() -> None:
    args = parse_args()
    review_file = args.review_file or args.data_root / args.dataset / f"{args.dataset}.json.gz"
    if args.review_file is None and not review_file.exists():
        review_file = review_file.with_suffix("")
    frames = {s: pd.read_csv(find_csv(args.data_root, args.dataset, s)) for s in SPLITS}
    needed = {(u, a) for f in frames.values() for u, a in zip(f.user_id, f.item_asin)}

    reviews = {}
    opener = gzip.open if review_file.suffix == ".gz" else open
    with opener(review_file, "rt", encoding="utf-8") as fin:
        for line in fin:
            row = json.loads(line)
            key = (row.get("reviewerID"), row.get("asin"))
            if key in needed:
                reviews[key] = (row.get("summary") or "", row.get("reviewText") or "")
    print(f"Matched {len(reviews)}/{len(needed)} interactions")

    out_dir = args.data_root / args.dataset / "collab"
    out_dir.mkdir(parents=True, exist_ok=True)
    for split, frame in frames.items():
        records = []
        for user, asin in zip(frame.user_id, frame.item_asin):
            summary, text = reviews.get((user, asin), ("", ""))
            records.append({
                "summary": summary.strip(),
                "review": text.strip(),
                "informative": is_informative(summary, text, args.min_words, args.domain),
            })
        path = out_dir / f"reviews_{split}.json"
        with path.open("w", encoding="utf-8") as fout:
            json.dump(records, fout)
        rate = sum(r["informative"] for r in records) / len(records)
        print(f"{split}: {len(records)} rows, informative={rate:.3f} -> {path}")


if __name__ == "__main__":
    main()
