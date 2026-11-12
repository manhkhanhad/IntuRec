#!/usr/bin/env python3
"""Check how well a facet list covers free-form purchase motives.

Takes an ``open``-mode and a ``facet``-mode rationale file generated on the
same rows. An LLM judge maps every free-form motive to its closest facet (or
"none"), giving:
  taxonomy coverage   share of motives the facet list can express at all
  assignment recall   share of motives whose facet was also assigned to that row
plus the "other" rate, per-facet share and evidence mix.
"""

import argparse
import collections
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

from gen_rationales import DEFAULT_FACETS, FACET_DEFS

BANNED = re.compile(r"familiar|comfort|preference|the review|history shows|history suggests|the buyer", re.I)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--open-file", type=Path, required=True)
    parser.add_argument("--facet-file", type=Path, required=True)
    parser.add_argument("--facets", default=",".join(DEFAULT_FACETS))
    parser.add_argument("--server", default="http://lexicon:8000")
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--embedder", default="sentence-transformers/all-mpnet-base-v2")
    return parser.parse_args()


def load(path: Path) -> dict:
    records = {}
    with path.open(encoding="utf-8") as fin:
        for line in fin:
            rec = json.loads(line)
            if "error" not in rec:
                records[rec["row"]] = rec
    return records


def judge(session: requests.Session, args, motive: str, facets: list) -> str:
    schema = {"type": "json_schema", "json_schema": {"name": "m", "schema": {
        "type": "object", "properties": {"facet": {"type": "string", "enum": facets + ["none"]}},
        "required": ["facet"], "additionalProperties": False}}}
    listing = "\n".join(f"- {f}: {FACET_DEFS.get(f, '')}".rstrip(": ") for f in facets)
    prompt = (f"Purchase motive: \"{motive}\"\nFacets:\n{listing}\n"
              "Which facet expresses this motive? Answer \"none\" if no facet fits it well.")
    resp = session.post(f"{args.server}/v1/chat/completions", timeout=120, json={
        "model": args.model, "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0, "max_tokens": 30, "response_format": schema,
        "chat_template_kwargs": {"enable_thinking": False}})
    resp.raise_for_status()
    return json.loads(resp.json()["choices"][0]["message"]["content"])["facet"]


def main() -> None:
    args = parse_args()
    facets = [f.strip() for f in args.facets.split(",") if f.strip()]
    open_recs, facet_recs = load(args.open_file), load(args.facet_file)
    rows = sorted(set(open_recs) & set(facet_recs))
    print(f"{len(rows)} rows in both files")

    session = requests.Session()
    sources = [("separate open run", open_recs)]
    if all("motives" in facet_recs[r] for r in rows):
        sources.append(("same call", facet_recs))
    for name, motive_recs in sources:
        pairs = [(r, m) for r in rows for m in motive_recs[r]["motives"]]
        with ThreadPoolExecutor(args.workers) as pool:
            mapped = list(pool.map(lambda p: judge(session, args, p[1], facets), pairs))
        covered = sum(f != "none" for f in mapped)
        recalled = sum(f in facet_recs[r]["facets"] for (r, _), f in zip(pairs, mapped))
        print(f"[motives from {name}] taxonomy coverage: {covered / len(pairs):.3f}  "
              f"assignment recall: {recalled / len(pairs):.3f}")
        uncovered = collections.Counter(m.lower() for (_, m), f in zip(pairs, mapped) if f == "none")
        print("  uncovered motives:", uncovered.most_common(15))

    assigned = collections.Counter(f for r in rows for f in facet_recs[r]["facets"])
    print(f"rows using 'other': {sum('other' in facet_recs[r]['facets'] for r in rows) / len(rows):.3f}")
    print("other_label:", collections.Counter(facet_recs[r]["other_label"].lower() for r in rows
                                              if "other" in facet_recs[r]["facets"]).most_common(15))
    print("facet share (rows containing facet):")
    for facet, n in assigned.most_common():
        print(f"  {facet:24s} {n / len(rows):.3f}")
    for informative in (True, False):
        sub = [facet_recs[r] for r in rows if facet_recs[r]["informative"] == informative]
        mix = collections.Counter(rec["evidence"] for rec in sub)
        print(f"evidence (informative={informative}, n={len(sub)}):",
              {k: round(v / len(sub), 3) for k, v in mix.items()})

    # Generic reasons embed close together and would fake collaborative similarity.
    from sentence_transformers import SentenceTransformer
    reasons = [facet_recs[r]["reason"] for r in rows]
    emb = SentenceTransformer(args.embedder).encode(reasons, normalize_embeddings=True)
    sim = emb @ emb.T
    off_diag = (sim.sum() - len(rows)) / (len(rows) * (len(rows) - 1))
    banned = sum(bool(BANNED.search(text)) for text in reasons) / len(rows)
    print(f"reason mean pairwise cosine: {off_diag:.3f}  reasons with banned phrases: {banned:.3f}")


if __name__ == "__main__":
    main()
