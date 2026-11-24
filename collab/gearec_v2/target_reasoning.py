"""
MotiveRec copy of GEARec_R3_toys/v2_target_reasoning/target_reasoning.py (GEARec's own file is untouched), extended
with --domain cds for CDs_and_Vinyl (music wording in the prompt, music stop words) and two asset stages:
  --stage items     -> items.json (titles from the split CSVs, brand/categories/description from Amazon meta)
  --stage item_emb  -> item_emb_mxbai.pt (mxbai over title/brand/categories/first 128 description words)
Toys behaviour (--domain toys) is unchanged.
2026-11-20: --domain steam / movielens (R3 Steam and Movielens splits). Their item_meta.json is {asin: {title,
description: ["Genres: ... Tags: ... [Developer/Publisher/Released]"]}}: --stage items reads it (--meta), genres become
categories, the developer (Steam) becomes the brand; item_block omits an empty brand. Prompt wording per domain.

v2: target-guided reasoning with model-selected support items + validation.

For each sample (history -> target), ONE prompt shows the history AND the target and asks
the model to (1) pick 1-3 supporting history items and (2) write a reasoning paragraph,
phrased as a prediction from the history, that never names the target.  N candidates
per sample are generated, then validated:

  V1 format     : both tags present; 1-3 distinct, valid history numbers.
  V2 leak       : (recorded; filtered only with --filter_leak) reasoning contains the target title, the target brand (when no
                  history item shares it), or any *distinctive* target-title word
                  (a word of the target title that is not in the history text and is rare
                  in the catalog).
  V3 reachable  : reasoning text ALONE (no titles prepended) embedded with mxbai; rank of
                  the target among all items (history masked) <= --rank_max.
                  Best candidate = lowest rank among those passing V1+V2.
  V4 judge      : an LLM judge (default gemma-3-4b-it) scores the best candidate
                  1-5 on grounded / support_relevant / target_blind; pass = all >= --judge_min.

Stages (separate processes so each model gets the whole GPU):
  --stage generate  -> {out}/candidates.jsonl
  --stage validate  -> {out}/validated.jsonl          (V1-V3, picks best candidate)
  --stage judge     -> {out}/judged.jsonl             (V4)
  --stage report    -> summary + {out}/final.jsonl    (samples passing everything)
"""
import argparse, ast, collections, datetime, glob, json, os, random, re, sys, time


GEN_SYSTEM = "You are an expert e-commerce analyst who explains why customers make their purchases."

GEN_TASK = """Task:
1. Select the 1-3 history items that best explain why the customer bought the next item. Refer to them by their numbers.
2. Write one reasoning paragraph (3-5 sentences). Start from the selected history items and explain the customer's underlying needs and preferences (for example: who the toys are for, age, interests, play style, features they value), and how these lead to the kind of item they want next.

Rules:
- Write the reasoning as a prediction made from the purchase history, as if you did not know the next item.
- Never mention the next item's name, brand, character, or product line. Describe what the customer wants next only in general terms (type of toy, features, use) that the history itself supports.
- Every claim about the customer must be supported by the purchase history.
- In the reasoning, refer to history items by their names (a short form of the title is fine), never by their numbers.

Output format (exactly):
<support_items>[comma-separated item numbers]</support_items>
<reasoning>[one paragraph]</reasoning>"""

GEN_TASK_CDS = """Task:
1. Select the 1-3 history items that best explain why the customer bought the next item. Refer to them by their numbers.
2. Write one reasoning paragraph (3-5 sentences). Start from the selected history items and explain the customer's underlying needs and preferences (for example: favourite genres, eras, artists' styles, moods or occasions, preferred formats such as CD, vinyl or box sets), and how these lead to the kind of item they want next.

Rules:
- Write the reasoning as a prediction made from the purchase history, as if you did not know the next item.
- Never mention the next item's title, artist, band, album, song or record label. Describe what the customer wants next only in general terms (genre, era, style, mood, format) that the history itself supports.
- Every claim about the customer must be supported by the purchase history.
- In the reasoning, refer to history items by their names (a short form of the title is fine), never by their numbers.

Output format (exactly):
<support_items>[comma-separated item numbers]</support_items>
<reasoning>[one paragraph]</reasoning>"""

GEN_TASK_STEAM = """Task:
1. Select the 1-3 history items that best explain why the player got the next game. Refer to them by their numbers.
2. Write one reasoning paragraph (3-5 sentences). Start from the selected history items and explain the player's underlying preferences (for example: favourite genres, gameplay styles, single-player or multiplayer, difficulty, setting or art style, indie or big-budget games), and how these lead to the kind of game they want next.

Rules:
- Write the reasoning as a prediction made from the history, as if you did not know the next game.
- Never mention the next game's title, franchise, developer or publisher. Describe what the player wants next only in general terms (genre, gameplay, setting, style) that the history itself supports.
- Every claim about the player must be supported by the history.
- In the reasoning, refer to history items by their names (a short form of the title is fine), never by their numbers.

Output format (exactly):
<support_items>[comma-separated item numbers]</support_items>
<reasoning>[one paragraph]</reasoning>"""

GEN_TASK_MOVIELENS = """Task:
1. Select the 1-3 history movies that best explain why the viewer watched the next movie. Refer to them by their numbers.
2. Write one reasoning paragraph (3-5 sentences). Start from the selected history movies and explain the viewer's underlying tastes (for example: favourite genres, themes, moods, eras, styles of storytelling), and how these lead to the kind of movie they want next.

Rules:
- Write the reasoning as a prediction made from the viewing history, as if you did not know the next movie.
- Never mention the next movie's title, franchise, director, actors or characters. Describe what the viewer wants next only in general terms (genre, theme, mood, era, style) that the history itself supports.
- Every claim about the viewer must be supported by the viewing history.
- In the reasoning, refer to history movies by their titles (a short form is fine), never by their numbers.

Output format (exactly):
<support_items>[comma-separated item numbers]</support_items>
<reasoning>[one paragraph]</reasoning>"""

DOMAIN_TEXT = {  # (system prompt, prompt header) per domain
    "toys": (None, None), "cds": (None, None),
    "steam": ("You are an expert video game analyst who explains why players choose the games they play.",
              "Below is a player's Steam game history (oldest to newest) and the game they got next."),
    "movielens": ("You are an expert film analyst who explains why viewers choose the movies they watch.",
                  "Below is a viewer's movie rating history (oldest to newest) and the movie they watched next."),
}
HEADER = "Below is a customer's purchase history (oldest to newest) and the item they bought next."
LABELS = ("User Purchase History", "Next Purchase")
DOMAIN_LABELS = {"steam": ("Game History", "Next Game"), "movielens": ("Viewing History", "Next Movie")}

JUDGE_SYSTEM = "You are a strict reviewer of explanations for e-commerce recommendations."

JUDGE_TASK = """An analyst saw this history and the next purchase, selected supporting history items, and wrote a reasoning paragraph that should read as a prediction made WITHOUT knowing the next purchase.

Check the reasoning carefully. Be strict: most reasonings have at least one problem.

1. leaked_detail: Does the reasoning mention any specific detail of the next purchase that is NOT already implied by the history items? Examples: its name, brand, character, theme, subject, or a distinguishing feature or function (e.g. history shows science kits and the reasoning says "exploring light or color" when the next item is about rainbows). Quote the exact phrase, or write NONE.
2. unsupported_claim: Quote any claim about the customer that the purchase history does not support, or write NONE.
3. support_matches: Are the selected support items the history items the reasoning actually relies on? Answer "yes" or "no".
4. support_relevant: How strongly do the selected support items motivate the next purchase? 1 (unrelated) to 5 (clearly related).

Answer with JSON only:
{"leaked_detail": "<quote or NONE>", "unsupported_claim": "<quote or NONE>", "support_matches": "<yes/no>", "support_relevant": <1-5>}"""


# ---------------------------------------------------------------- helpers
def load_items(path):
    return {int(k): v for k, v in json.load(open(path)).items()}


def item_block(it, n=None, ts=None):
    head = f"{n}. " if n is not None else ""
    date = (f"Date: {datetime.datetime.utcfromtimestamp(int(ts)).strftime('%Y-%m-%d')}\n"
            if ts is not None else "")
    cats = ", ".join(it["categories"]) or "Unknown Category"
    brand = f"Brand: {it['brand']}\n" if it.get("brand") else ""
    tags = f"Tags: {it['tags']}\n" if it.get("tags") else ""
    return f"{head}{date}Title: {it['title']}\n{brand}Categories: {cats}\n{tags}"


def history_section(items, hist, ts):
    return "\n".join(item_block(items[i], n, t) for n, (i, t) in enumerate(zip(hist, ts), 1))


TASK = GEN_TASK  # set from --domain in __main__


def gen_prompt(items, hist, ts, target):
    return (f"{HEADER}\n\n"
            f"- **{LABELS[0]}:**\n{history_section(items, hist, ts)}\n"
            f"- **{LABELS[1]} (for your analysis only):**\n{item_block(items[target])}\n{TASK}")


def parse(text):
    s = re.search(r"<support_items>(.*?)</support_items>", text, re.S)
    r = re.search(r"<reasoning>(.*?)</reasoning>", text, re.S)
    return (s.group(1).strip() if s else None), (r.group(1).strip() if r else None)


WORD = re.compile(r"[a-z][a-z'\-]+")
STOP = set("""the and for with set toy toys kids kid game games pack piece pieces inch inches
year years ages age small large mini big new style edition series color colors classic deluxe
baby boys girls girl boy play playset figure figures doll dolls""".split())
STOP_CDS = set("""the and for with cd cds vinyl lp album albums edition remastered remaster deluxe live best of hits
greatest collection anthology vol volume disc discs box set original soundtrack music songs version expanded
anniversary import digipak mono stereo""".split())
STOP_STEAM = set("""the and for with game games edition deluxe complete collection pack bundle remastered hd simulator
gold ultimate goty definitive soundtrack dlc vr online""".split())
STOP_MOVIELENS = set("""the and for with movie film part chapter story man men life love night day last first""".split())
STOPS = {"toys": STOP, "cds": STOP_CDS, "steam": STOP_STEAM, "movielens": STOP_MOVIELENS}


def words(t):
    return set(WORD.findall(t.lower()))


def load_samples(a, items):
    df = __import__("pandas").read_csv(glob.glob(os.path.join(a.data_dir, a.split, "*.csv"))[0])
    df["sample_id"] = range(len(df))
    if a.sample_n:
        df = df.sample(n=a.sample_n, random_state=a.seed).sort_values("sample_id")
    if a.only_ids:
        keep = set(json.load(open(a.only_ids)))
        df = df[df["sample_id"].isin(keep)]
    out = []
    for r in df.itertuples():
        h = ast.literal_eval(r.history_item_id)
        ts = ast.literal_eval(r.history_timestamp)
        if a.history_limit:
            h, ts = h[-a.history_limit:], ts[-a.history_limit:]
        out.append({"sample_id": int(r.sample_id), "user_id": r.user_id, "target_id": int(r.item_id),
                    "history_ids": h, "history_ts": ts})
    return out


def _norm(t):
    t = t.lower().replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    return re.sub(r"[^a-z0-9' ]+", " ", t).split()


def verified_quote(flag, reasoning, min_words=2):
    """True if the judge's flag quotes (part of) the reasoning; False for NONE or hallucinated quotes."""
    if flag is None or str(flag).strip().strip('"').upper() in ("NONE", "", "N/A"):
        return False
    flag = str(flag)
    # quote delimiters must not sit inside a word, so apostrophes ("it's", "Schleich's") are not
    # mistaken for closing quotes; the whole flag text is always checked as a fallback
    quoted = re.findall(r"(?<![A-Za-z])['\"‘“](.+?)['\"’”](?![A-Za-z])", flag) + [flag]
    rtoks = " " + " ".join(_norm(reasoning)) + " "
    for q in quoted:
        qt = _norm(q)
        if len(qt) >= min_words and (" " + " ".join(qt) + " ") in rtoks:
            return True
        # judge sometimes paraphrases the frame; accept any 4-word window that occurs verbatim
        for k in range(len(qt) - 3):
            if " " + " ".join(qt[k:k + 4]) + " " in rtoks:
                return True
    return False


def chat(tok, system, user):
    return tok.apply_chat_template([{"role": "system", "content": system},
                                    {"role": "user", "content": user}],
                                   tokenize=False, add_generation_prompt=True)


# ---------------------------------------------------------------- stages
def stage_generate(a, items):
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    S = load_samples(a, items)
    prev = {}
    if a.extend_from:  # append the new candidates to an earlier run's (use a different --seed)
        for l in open(os.path.join(a.extend_from, "candidates.jsonl")):
            x = json.loads(l)
            prev[x["sample_id"]] = x["candidates"]
        assert set(prev) == {s["sample_id"] for s in S}, "extend_from run has different samples"
    tok = AutoTokenizer.from_pretrained(a.gen_model)
    prompts = [chat(tok, DOMAIN_TEXT[a.domain][0] or GEN_SYSTEM, gen_prompt(items, s["history_ids"], s["history_ts"], s["target_id"]))
               for s in S]
    llm = LLM(model=a.gen_model, max_model_len=6144, gpu_memory_utilization=a.gpu_mem,
              dtype="bfloat16", max_num_seqs=256, seed=a.seed, enable_prefix_caching=True)
    t0 = time.time()
    outs = llm.generate(prompts, SamplingParams(temperature=a.temperature, top_p=0.95,
                                                max_tokens=a.max_tokens, n=a.n_samples, seed=a.seed))
    print(f"generated {len(S)} x {a.n_samples} in {time.time() - t0:.0f}s")
    with open(os.path.join(a.out, "candidates.jsonl"), "w") as f:
        for s, p, o in zip(S, prompts, outs):
            s["prompt"] = p
            s["candidates"] = prev.get(s["sample_id"], []) + [x.text.strip() for x in o.outputs]
            f.write(json.dumps(s) + "\n")


def stage_validate(a, items):
    import torch
    from sentence_transformers import SentenceTransformer
    S = [json.loads(l) for l in open(os.path.join(a.out, "candidates.jsonl"))]
    # catalog document frequency of title words -> "distinctive" = rare words
    df_ = collections.Counter(w for it in items.values() for w in words(it["title"]))
    stop = STOPS[a.domain]
    rare = lambda w: df_[w] <= a.rare_df and w not in stop and len(w) >= 4

    emb = SentenceTransformer(a.emb_model, device="cuda")
    ie = torch.load(a.item_emb).cuda()
    ie = ie / ie.norm(dim=-1, keepdim=True).clamp(min=1e-8)

    jobs = []
    for s in S:
        t = items[s["target_id"]]
        hist_txt = " ".join(items[i]["title"] + " " + items[i]["brand"] + " " + " ".join(items[i]["categories"])
                            for i in s["history_ids"])
        hist_w = words(hist_txt)
        brand = t["brand"].strip().lower()
        brand_new = len(brand) >= 3 and brand not in ("unknown brand", "generic") and brand not in hist_txt.lower()
        distinct = sorted(w for w in words(t["title"]) if rare(w) and w not in hist_w)
        s["checks"] = []
        for ci, c in enumerate(s["candidates"]):
            sup, rea = parse(c)
            ck = {"cand": ci, "v1_format": False, "v2_leak": None, "rank": None}
            nums = [int(x) for x in re.findall(r"\d+", sup or "")]
            nums = list(dict.fromkeys(nums))
            # drop out-of-range citations (e.g. "1, 2" with a 1-item history) instead of rejecting the candidate
            nums = [n for n in nums if 1 <= n <= len(s["history_ids"])]
            if rea and nums and 1 <= len(nums) <= 3:
                ck["v1_format"] = True
                ck["support"] = nums
                ck["reasoning"] = rea
                low = rea.lower()
                leaks = []
                if t["title"].lower()[:40] in low:
                    leaks.append("title")
                if brand_new and re.search(r"\b" + re.escape(brand) + r"\b", low):
                    leaks.append("brand:" + brand)
                leaks += ["word:" + w for w in distinct if re.search(r"\b" + re.escape(w) + r"\b", low)]
                ck["v2_leak"] = leaks
                if a.filter_leak and leaks:
                    continue
                jobs.append((s, ck))
            s["checks"].append(ck)

    texts = [ck["reasoning"] for _, ck in jobs]
    q = emb.encode(texts, batch_size=256, convert_to_tensor=True, show_progress_bar=False).float()
    q = q / q.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    for j, (s, ck) in enumerate(jobs):
        sc = q[j] @ ie.T
        excl = [i for i in s["history_ids"] if i != s["target_id"]]
        if excl:
            sc[excl] = float("-inf")
        ck["rank"] = int((sc > sc[s["target_id"]]).sum()) + 1

    with open(os.path.join(a.out, "validated.jsonl"), "w") as f:
        for s in S:
            ok = [ck for ck in s["checks"] if ck["v1_format"] and (not a.filter_leak or not ck["v2_leak"])]
            best = min(ok, key=lambda c: c["rank"]) if ok else None
            s["best"] = best
            s["pass_v1"] = any(ck["v1_format"] for ck in s["checks"])
            s["pass_v2"] = bool(ok)
            s["pass_v3"] = bool(best and best["rank"] <= a.rank_max)
            f.write(json.dumps(s) + "\n")
    print(f"validated {len(S)} samples")


def stage_judge(a, items):
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    S = [json.loads(l) for l in open(os.path.join(a.out, "validated.jsonl"))]
    todo = [s for s in S if s["pass_v3"]]
    tok = AutoTokenizer.from_pretrained(a.judge_model)
    prompts = []
    for s in todo:
        h = s["history_ids"]
        sup = "\n".join(f"{n}. {items[h[n - 1]]['title']}" for n in s["best"]["support"])
        user = (f"- **Purchase History:**\n{history_section(items, h, s['history_ts'])}\n"
                f"- **Next Purchase:**\n{item_block(items[s['target_id']])}\n"
                f"- **Selected support items:**\n{sup}\n\n"
                f"- **Reasoning:**\n{s['best']['reasoning']}\n\n{JUDGE_TASK}")
        prompts.append(chat(tok, JUDGE_SYSTEM, user))
    llm = LLM(model=a.judge_model, max_model_len=8192, gpu_memory_utilization=a.gpu_mem_judge,
              dtype="bfloat16", max_num_seqs=128, seed=a.seed)
    t0 = time.time()
    outs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=200))
    print(f"judged {len(todo)} in {time.time() - t0:.0f}s")
    for s, o in zip(todo, outs):
        txt = o.outputs[0].text
        m = re.search(r"\{.*\}", txt, re.S)
        try:
            j = json.loads(m.group(0)) if m else None
        except json.JSONDecodeError:
            j = None
        s["judge"] = j or {"raw": txt}
        rea = s["best"]["reasoning"]
        s["judge_checks"] = {
            # a flag only counts if the quoted phrase really occurs in the reasoning
            # (small judges sometimes "quote" the target instead of the reasoning)
            "no_leak": bool(j) and not verified_quote(j.get("leaked_detail"), rea),
            "grounded": bool(j) and not verified_quote(j.get("unsupported_claim"), rea),
            "support_matches": bool(j) and str(j.get("support_matches", "")).lower().startswith("y"),
            "support_relevant": bool(j) and isinstance(j.get("support_relevant"), (int, float))
                                and j["support_relevant"] >= a.judge_min,
        }
        s["pass_v4"] = judge_pass(s["judge_checks"], a)
    with open(os.path.join(a.out, "judged.jsonl"), "w") as f:
        for s in S:
            s.setdefault("pass_v4", False)
            f.write(json.dumps(s) + "\n")


def judge_pass(checks, a):
    """no_leak and grounded are recorded only, unless --filter_leak / --filter_grounded."""
    skip = set() if a.filter_leak else {"no_leak"}
    if not a.filter_grounded:
        skip.add("grounded")
    return all(v for k, v in checks.items() if k not in skip)


def stage_report(a, items):
    import statistics as st
    S = [json.loads(l) for l in open(os.path.join(a.out, "judged.jsonl"))]
    n = len(S)
    for s in S:  # re-apply the current pass rule, so changing it needs no judge re-run
        s["pass_v4"] = bool(s.get("judge_checks")) and judge_pass(s["judge_checks"], a)
    rep = sum(s["target_id"] in s["history_ids"] for s in S)
    print(f"samples {n} (target already in history: {rep})")
    for k, lab in [("pass_v1", "V1 format"), ("pass_v2", "V2 leak filter" if a.filter_leak else "V2 (leaks recorded only)"), ("pass_v3", f"V3 rank<={a.rank_max}"),
                   ("pass_v4", f"V4 judge>={a.judge_min}")]:
        print(f"  cumulative {lab:<16} {sum(s[k] for s in S):6d}  ({sum(s[k] for s in S) / n:.1%})")
    cands = [ck for s in S for ck in s["checks"]]
    fm = [ck for ck in cands if ck["v1_format"]]
    print(f"  candidates: {len(cands)}, well-formed {len(fm) / len(cands):.1%}, "
          f"leaky among well-formed {sum(bool(c['v2_leak']) for c in fm) / max(len(fm), 1):.1%}")
    lk = collections.Counter(x.split(":")[0] for c in fm for x in (c["v2_leak"] or []))
    print(f"  leak types: {dict(lk)}")
    br = [s["best"]["rank"] for s in S if s["best"]]
    if br:
        print(f"  best-candidate rank: median {st.median(br):.0f}, top20 {sum(r <= 20 for r in br) / len(br):.1%}, "
              f"top100 {sum(r <= 100 for r in br) / len(br):.1%}")
    JC = [s["judge_checks"] for s in S if s.get("judge_checks")]
    if JC:
        print(f"  judge checks on {len(JC)} judged samples: " +
              ", ".join(f"{k} {sum(c[k] for c in JC) / len(JC):.1%}" for k in JC[0]))
    ns = collections.Counter(len(s["best"]["support"]) for s in S if s["best"])
    print(f"  #support items in best candidate: {dict(sorted(ns.items()))}")
    with open(os.path.join(a.out, "final.jsonl"), "w") as f:
        for s in S:
            if s["pass_v4"]:
                h = s["history_ids"]
                sup_titles = [items[h[k - 1]]["title"] for k in s["best"]["support"]]
                f.write(json.dumps({
                    "sample_id": s["sample_id"], "user_id": s["user_id"],
                    "history_ids": h, "target_id": s["target_id"],
                    "support_ids": [h[k - 1] for k in s["best"]["support"]],
                    "support_titles": sup_titles, "reasoning": s["best"]["reasoning"],
                    "rank": s["best"]["rank"], "judge": s["judge"],
                    "target_in_history": s["target_id"] in h,
                }) + "\n")
    print(f"final (all checks passed): {sum(s['pass_v4'] for s in S)} -> {os.path.join(a.out, 'final.jsonl')}")


# ---------------------------------------------------------------- asset stages (MotiveRec addition)
def stage_items(a, items):
    import gzip
    import pandas as pd
    asin2id, id2title = {}, {}
    for split in ("train", "valid", "test"):
        df = pd.read_csv(glob.glob(os.path.join(a.data_dir, split, "*.csv"))[0])
        for r in df.itertuples():
            asins = ast.literal_eval(r.item_asins) + [r.item_asin]
            ids = ast.literal_eval(r.history_item_id) + [int(r.item_id)]
            titles = ast.literal_eval(r.history_item_title) + [r.item_title]
            for s_, i, t in zip(asins, ids, titles):
                asin2id[s_] = int(i)
                id2title[int(i)] = str(t)
    print(f"{len(asin2id)} asins -> {len(set(asin2id.values()))} item ids", flush=True)
    if a.meta.endswith("item_meta.json"):  # Steam / Movielens (R3): {asin: {title, description: [text]}}
        import re as _re
        meta = json.load(open(a.meta))
        out = {}
        for asin, iid in asin2id.items():
            d = " ".join(meta.get(str(asin), {}).get("description", []))
            g = _re.search(r"Genres: (.*?)\.(?: |$)", d)
            t = _re.search(r"Tags: (.*?)\.(?: |$)", d)
            dev = _re.search(r"Developer: (.*?)\.(?: |$)", d)
            out[str(iid)] = {"asin": str(asin), "title": id2title[iid], "brand": dev.group(1) if dev else "",
                             "categories": [x.strip() for x in g.group(1).split(",")] if g else [],
                             "tags": t.group(1) if t else "", "description": d}
        print(f"items {len(out)}, without metadata: {sum(str(a_) not in meta for a_ in asin2id)}")
        json.dump(dict(sorted(out.items(), key=lambda kv: int(kv[0]))), open(a.items, "w"))
        print(f"saved {a.items}")
        return
    meta = {}
    op = gzip.open if a.meta.endswith(".gz") else open
    with op(a.meta, "rt") as f:
        for line in f:
            k = line.find('"asin": "')
            if k < 0:
                continue
            asin = line[k + 9:line.find('"', k + 9)]
            if asin in asin2id and asin not in meta:
                meta[asin] = json.loads(line)
    out = {}
    for asin, iid in asin2id.items():
        m = meta.get(asin, {})
        desc = m.get("description", [])
        out[str(iid)] = {"asin": asin, "title": id2title[iid], "brand": m.get("brand", "") or "Unknown Brand",
                         "categories": [c for c in m.get("category", []) if c],
                         "description": " ".join(desc) if isinstance(desc, list) else str(desc)}
    print(f"items {len(out)}, without metadata: {sum(a_ not in meta for a_ in asin2id)}")
    json.dump(dict(sorted(out.items(), key=lambda kv: int(kv[0]))), open(a.items, "w"))
    print(f"saved {a.items}")


def stage_item_emb(a, items):
    import torch
    from sentence_transformers import SentenceTransformer
    doc = lambda it: (f"Title: {it['title']}\nBrand: {it['brand']}\nCategory: {', '.join(it['categories'])}\n"
                      f"Description: {' '.join(it['description'].split()[:128])}")
    emb = SentenceTransformer(a.emb_model, device="cuda")
    ids = sorted(items)
    e = emb.encode([doc(items[i]) for i in ids], batch_size=256, convert_to_tensor=True,
                   show_progress_bar=True).float().cpu()
    out = torch.zeros(max(ids) + 1, e.shape[1])
    out[ids] = e
    torch.save(out, a.item_emb)
    print(f"wrote {a.item_emb} {tuple(out.shape)}")


if __name__ == "__main__":
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["items", "item_emb", "generate", "validate", "judge", "report"])
    ap.add_argument("--domain", default="toys", choices=["toys", "cds", "steam", "movielens"])
    ap.add_argument("--meta", default="", help="items stage: Amazon meta_<cat>.json(.gz)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--data_dir", default="/common/home/users/m/mk.ngohuu.2024/research/R3/data/Toys_and_Games")
    ap.add_argument("--items", default=os.path.join(here, "items.json"))
    ap.add_argument("--item_emb", default=os.path.join(here, "outputs", "item_emb_mxbai.pt"))
    ap.add_argument("--split", default="train")
    ap.add_argument("--sample_n", type=int, default=0, help="random subset size (0 = all)")
    ap.add_argument("--history_limit", type=int, default=0, help="keep the last N history items (0 = all)")
    ap.add_argument("--only_ids", default="", help="JSON list of sample_ids to generate for (default: all)")
    ap.add_argument("--gen_model", default="google/gemma-3-4b-it")
    ap.add_argument("--judge_model", default="google/gemma-3-4b-it")
    ap.add_argument("--emb_model", default="mixedbread-ai/mxbai-embed-large-v1")
    ap.add_argument("--n_samples", type=int, default=4)
    ap.add_argument("--extend_from", default="", help="generate: prepend this run dir's candidates (Best-of-N top-up)")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--max_tokens", type=int, default=400)
    ap.add_argument("--rank_max", type=int, default=20)
    ap.add_argument("--judge_min", type=int, default=4)
    ap.add_argument("--filter_leak", action="store_true",
                    help="drop candidates with lexical leaks (V2) and judge-flagged leaks; default: record only")
    ap.add_argument("--filter_grounded", action="store_true",
                    help="require the judge's grounded check to pass; default: record only")
    ap.add_argument("--rare_df", type=int, default=30, help="title word counts as distinctive if in <= this many item titles")
    ap.add_argument("--gpu_mem", type=float, default=0.6)
    ap.add_argument("--gpu_mem_judge", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    TASK = {"toys": GEN_TASK, "cds": GEN_TASK_CDS, "steam": GEN_TASK_STEAM, "movielens": GEN_TASK_MOVIELENS}[a.domain]
    HEADER = DOMAIN_TEXT[a.domain][1] or HEADER
    LABELS = DOMAIN_LABELS.get(a.domain, LABELS)
    items = {} if a.stage == "items" else load_items(a.items)
    globals()["stage_" + a.stage](a, items)
