#!/usr/bin/env python3
"""Generate review-grounded purchase rationales with an OpenAI-compatible vLLM server.

Modes:
  open   free-form motive phrases, no facet list shown (facet-discovery pilot)
  facet  facets constrained to --facets (+ "other") via guided JSON decoding
  predict  HyDE query: from the history ONLY (no target, no review), predict the
           motivation behind the next purchase, in the same format as facet mode

Output is a resumable JSONL keyed by CSV row index:
``<data_root>/<dataset>/collab/rationales_<mode>_<split>[_<tag>].jsonl``.
"""

import argparse
import json
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

# v5: revised from the open-vocabulary pilot (rationales_open_train_pilot.jsonl)
# and the held-out coverage checks.
FACET_DEFS = {
    "genre_style_affinity": "wants more of a specific genre or musical style they already buy",
    "artist_loyalty": "is a loyal fan of a specific artist or band",
    "nostalgia_era": "connects to their own past or youth, or to a past era of music",
    "emotional_mood": "seeks emotional connection or resonance, or a specific mood (uplift, energy, relaxation, solace)",
    "artistry_musicianship": "admires the vocals, songwriting, playing skill or artistic vision",
    "compilation_best_of": "wants a greatest-hits, anthology or curated compilation rather than a regular album",
    "collection_completion": "fills a gap in a set or discography, or seeks a rare or collectible edition; NOT merely adding another album",
    "format_sound_quality": "cares about the vinyl/CD format, pressing, remaster, audio fidelity or packaging",
    "discovery_novelty": "is curious to explore an artist, style or work that is new to them",
    "live_performance": "wants a live or concert recording",
    "value_price": "driven by price or the amount of content for the money",
    "occasion_gift_seasonal": "buys for a holiday, season or event, or as a gift for someone else",
    "soundtrack_media_tiein": "connected to a film, TV show, stage musical or game",
}
DEFAULT_FACETS = list(FACET_DEFS)

# Toys_and_Games: same design, facets written for toy / game purchases.
TOY_FACET_DEFS = {
    "gift_occasion": "buys for a birthday, holiday, party or other occasion, or as a gift for someone else",
    "child_development_learning": "wants to teach a skill or support a child's learning or development (educational, STEM, motor skills)",
    "age_stage_fit": "picks something suited to a child's age, size or developmental stage, or a child outgrowing earlier toys",
    "interest_theme_affinity": "follows a theme, character type or play interest the buyer or child already likes",
    "franchise_brand_loyalty": "is loyal to a specific franchise, character universe or toy line",
    "collection_completion": "fills a gap in a set or series, or seeks a rare or collectible item; NOT merely buying another similar toy",
    "creativity_building": "wants creative, building, craft or open-ended imaginative play",
    "social_family_play": "wants a game or activity to play together with family, friends or a group",
    "hobby_enthusiast": "pursues an adult hobby (tabletop and strategy games, puzzles, models, RC, collecting)",
    "active_outdoor_play": "wants physical, active, outdoor, water or ride-on play",
    "replacement_add_on": "replaces a lost, broken or worn item, or adds an accessory, expansion or part to something owned",
    "quality_safety": "cares about durability, materials, safety or build quality",
    "value_price": "driven by price, a deal or the amount of content for the money",
}
DOMAINS = {
    "music": {
        "facets": FACET_DEFS,
        "system": "You analyse why music buyers purchase albums. Answer only with the requested JSON.",
        "item": "album",
        "concrete": "gift, soundtrack, format or pressing, cover art, live performance, price, replacing a lost copy...",
        "names": "album, song or artist name",
        "generic": "enjoying music, liking quality, or a familiar or comfortable sound",
    },
    "toys": {
        "facets": TOY_FACET_DEFS,
        "system": "You analyse why shoppers buy toys and games. Answer only with the requested JSON.",
        "item": "toy or game",
        "concrete": "gift, birthday or holiday, a child's age, learning a skill, playing as a family, completing a set, "
                    "replacing a lost or broken piece, price...",
        "names": "product, brand, franchise or character name",
        "generic": "liking toys, having fun, liking quality, or a familiar or comfortable choice",
    },
}
EVIDENCE = ["review", "history", "weak"]

PROMPT = """A buyer's purchase history (oldest to newest):
{history}

They then purchased: "{target}"

The buyer's own review of that purchase (may be uninformative):
Summary: {summary}
Review: {review}

{task}
Rules: if the review names a concrete motive ({concrete}), it takes priority over the purchase history. Otherwise infer from the history and the purchase. Never mention any {names}; describe tastes and motives abstractly."""

PREDICT_PROMPT = """A buyer's purchase history (oldest to newest):
{history}

The buyer is about to buy their next {item}. You do not know which one.

{task}
Rules: infer from the purchase history only. Never mention any {names}; describe tastes and motives abstractly."""

OPEN_TASK = """Return JSON:
{"motives": [1-3 short phrases (2-6 words each) naming the buyer's motivations],
 "reason": "<=40 words explaining the buyer's motivation"}"""

FACET_TASK = """Motivation facets:
{facets}
- other: none of the above fits; name it in other_label (2-5 words)

First write the 1-2 most important motive phrases (2-6 words each) in your own words, then label them with the facets that match those motives.
Do not use generic motives such as {generic}.
evidence: "review" if the review explicitly states the motive, "history" if it is inferred from the purchase pattern, "weak" if it is a guess.
Return JSON:
{{"motives": [1-2 phrases], "facets": [1-2 facets matching the motives], "other_label": "<empty unless other is chosen>",
 "evidence": "review|history|weak", "reason": "<reason>"}}
reason: 1-2 sentences (<=40 words) written in the first person as the buyer ("I ..."), giving the concrete trigger for buying THIS item now, not a general taste profile. Include any specific trigger the review names, described without titles. Do not use the words "familiar", "comfort" or "preference"."""

PREDICT_TASK = """Motivation facets:
{facets}
- other: none of the above fits; name it in other_label (2-5 words)

First write the 1-2 most likely motive phrases (2-6 words each) for the next purchase in your own words, then label them with the facets that match those motives.
Do not use generic motives such as {generic}.
Return JSON:
{{"motives": [1-2 phrases], "facets": [1-2 facets matching the motives], "other_label": "<empty unless other is chosen>",
 "reason": "<reason>"}}
reason: 1-2 sentences (<=40 words) written in the first person as the buyer ("I ..."), giving the most likely concrete trigger for the next purchase, not a general taste profile. Do not use the words "familiar", "comfort" or "preference"."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="CDs_and_Vinyl")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--split", default="train")
    parser.add_argument("--mode", choices=["open", "facet", "predict"], required=True)
    parser.add_argument("--domain", choices=list(DOMAINS), default="music",
                        help="Prompt wording and default facet list (music = CDs_and_Vinyl, toys = Toys_and_Games).")
    parser.add_argument("--facets", default=None,
                        help="Comma-separated facet enum for --mode facet (default: all facets of --domain).")
    parser.add_argument("--sample", type=int, default=-1,
                        help="Stratified sample size (by informative review x target popularity); -1 = all rows.")
    parser.add_argument("--exclude-from", type=Path, default=None,
                        help="JSONL whose rows are excluded from sampling (to get a held-out pilot set).")
    parser.add_argument("--rows-from", type=Path, default=None,
                        help="JSONL whose rows are used as-is (overrides --sample/--exclude-from).")
    parser.add_argument("--tag", default="", help="Suffix for the output file name.")
    parser.add_argument("--server", default="http://lexicon:8000")
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--max-review-words", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def find_csv(data_root: Path, dataset: str, split: str) -> Path:
    matches = sorted((data_root / dataset / split).glob(f"{dataset}*.csv"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one {dataset}*.csv in {data_root / dataset / split}")
    return matches[0]


def stratified_sample(frame: pd.DataFrame, reviews: list, n: int, exclude: set, seed: int) -> list:
    popularity = frame.item_id.map(frame.item_id.value_counts())
    tercile = pd.qcut(popularity.rank(method="first"), 3, labels=False)
    strata = {}
    for idx in range(len(frame)):
        if idx not in exclude:
            strata.setdefault((reviews[idx]["informative"], int(tercile[idx])), []).append(idx)
    rng = random.Random(seed)
    picked = []
    for key in sorted(strata):
        pool = strata[key]
        picked += rng.sample(pool, min(len(pool), round(n / len(strata))))
    return sorted(picked)


def response_schema(mode: str, facets: list) -> dict:
    if mode == "open":
        props = {"motives": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 3},
                 "reason": {"type": "string"}}
    else:
        props = {"motives": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 2},
                 "facets": {"type": "array", "items": {"type": "string", "enum": facets + ["other"]},
                            "minItems": 1, "maxItems": 2},
                 "other_label": {"type": "string"},
                 "evidence": {"type": "string", "enum": EVIDENCE},
                 "reason": {"type": "string"}}
        if mode == "predict":
            del props["evidence"]
    return {"type": "json_schema", "json_schema": {"name": "rationale", "schema": {
        "type": "object", "properties": props, "required": list(props), "additionalProperties": False}}}


def build_prompt(row, review: dict, task: str, max_words: int, mode: str, domain: dict) -> str:
    history = "\n".join(f"- {t}" for t in eval(row.history_item_title))
    if mode == "predict":
        return PREDICT_PROMPT.format(history=history, task=task, **domain)
    text = " ".join(review["review"].split()[:max_words]) or "(none)"
    return PROMPT.format(history=history, target=row.item_title,
                         summary=review["summary"] or "(none)", review=text, task=task, **domain)


def call(session: requests.Session, args, prompt: str, schema: dict, system: str) -> dict:
    payload = {
        "model": args.model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 200,
        "chat_template_kwargs": {"enable_thinking": False},
        "response_format": schema,
    }
    for attempt in range(5):
        try:
            resp = session.post(f"{args.server}/v1/chat/completions", json=payload, timeout=120)
            resp.raise_for_status()
            return json.loads(resp.json()["choices"][0]["message"]["content"])
        except (requests.RequestException, json.JSONDecodeError, KeyError) as err:
            if attempt == 4:
                return {"error": str(err)}
            time.sleep(2 ** attempt)


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(find_csv(args.data_root, args.dataset, args.split))
    collab_dir = args.data_root / args.dataset / "collab"
    with (collab_dir / f"reviews_{args.split}.json").open(encoding="utf-8") as fin:
        reviews = json.load(fin)

    exclude = set()
    if args.exclude_from:
        with args.exclude_from.open(encoding="utf-8") as fin:
            exclude = {json.loads(line)["row"] for line in fin}
    if args.rows_from:
        with args.rows_from.open(encoding="utf-8") as fin:
            rows = sorted({json.loads(line)["row"] for line in fin})
    elif args.sample > 0:
        rows = stratified_sample(frame, reviews, args.sample, exclude, args.seed)
    else:
        rows = [i for i in range(len(frame)) if i not in exclude]

    suffix = f"_{args.tag}" if args.tag else ""
    out_path = collab_dir / f"rationales_{args.mode}_{args.split}{suffix}.jsonl"
    done = set()
    if out_path.exists():
        with out_path.open(encoding="utf-8") as fin:
            done = {json.loads(line)["row"] for line in fin if '"error"' not in line}
    todo = [i for i in rows if i not in done]
    print(f"{len(rows)} rows selected, {len(done)} already done, {len(todo)} to generate -> {out_path}")

    domain = DOMAINS[args.domain]
    facet_defs = domain["facets"]
    facets = [f.strip() for f in (args.facets or ",".join(facet_defs)).split(",") if f.strip()]
    facet_listing = "\n".join(f"- {f}: {facet_defs.get(f, '')}".rstrip(": ") for f in facets)
    task = {"open": OPEN_TASK, "facet": FACET_TASK, "predict": PREDICT_TASK}[args.mode]
    if args.mode != "open":
        task = task.format(facets=facet_listing, generic=domain["generic"])
    schema = response_schema(args.mode, facets)
    session = requests.Session()
    lock = threading.Lock()

    def work(idx: int) -> dict:
        prompt = build_prompt(frame.iloc[idx], reviews[idx], task, args.max_review_words, args.mode, domain)
        return {"row": idx, "item_id": int(frame.item_id.iloc[idx]),
                "informative": reviews[idx]["informative"], **call(session, args, prompt, schema, domain["system"])}

    errors = 0
    with out_path.open("a", encoding="utf-8") as fout, ThreadPoolExecutor(args.workers) as pool:
        for fut in tqdm(as_completed([pool.submit(work, i) for i in todo]), total=len(todo)):
            rec = fut.result()
            errors += "error" in rec
            with lock:
                fout.write(json.dumps(rec) + "\n")
                fout.flush()
    print(f"Finished with {errors} errors (rerun the same command to retry them).")


if __name__ == "__main__":
    main()
