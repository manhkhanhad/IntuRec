"""Shared data / vocabulary helpers for DLRRec (discrete latent reasoning for recommendation).

Reasoning codes are the motive prototypes of collab/motive_corpus (built in the main
checkout): each code <|m_k|> stands for one motive, with a canonical text
("motive: first-person reason"). A row's gold codes are the motive prototypes of its
own rationale (0-2 per row), ordered by train frequency (most common first).
"""

import collections
import glob
import json
import os
import random

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from data.IntuRecF2Dataset import Tokenizer

# Data and the motive corpus live in the main checkout (untracked there).
MAIN = "/common/home/users/m/mk.ngohuu.2024/research/IntuRec"
DATA_PATH = f"{MAIN}/data"
CORPUS = f"{MAIN}/data/CDs_and_Vinyl/collab/motive_corpus"
RATIONALES = f"{MAIN}/data/CDs_and_Vinyl/collab/rationales_facet_{{split}}_v5.jsonl"

REASON, END_REASON = "<|Reason|>", "<|EndReason|>"
START_PREDICT, END_PREDICT = "<|StartPredict|>", "<|EndPredict|>"
CATEGORY = {"CDs_and_Vinyl": "musics", "Books": "books", "Toys_and_Games": "toys and games",
            "Video_Games": "video games", "Musical_Instruments": "music instruments"}


def code_token(k: int) -> str:
    return f"<|m_{k}|>"


def load_prototypes(corpus: str = CORPUS) -> list:
    with open(os.path.join(corpus, "prototypes.json"), encoding="utf-8") as f:
        protos = json.load(f)
    return [f"{p['motive']}: {p['reason']}" for p in protos]


def load_row_codes(split: str, corpus: str = CORPUS) -> list:
    """Gold codes per row (0-2), ordered by train frequency (most common motive first)."""
    with open(os.path.join(corpus, "row_motives_train.json")) as f:
        freq = collections.Counter(k for ks in json.load(f) for k in ks)
    with open(os.path.join(corpus, f"row_motives_{split}.json")) as f:
        rows = json.load(f)
    return [sorted(ks, key=lambda k: (-freq[k], k)) for ks in rows]


def add_reasoning_tokens(tokenizer, n_codes: int) -> list:
    tokens = [REASON, END_REASON, START_PREDICT, END_PREDICT] + [code_token(k) for k in range(n_codes)]
    new = [t for t in tokens if t not in tokenizer.get_vocab()]
    tokenizer.add_special_tokens({"additional_special_tokens": tokenizer.additional_special_tokens + new})
    tokenizer.padding_side = "left"
    return new


def code_ids(tokenizer, n_codes: int) -> list:
    return [tokenizer.convert_tokens_to_ids(code_token(k)) for k in range(n_codes)]


def read_split(dataset: str, split: str) -> pd.DataFrame:
    files = glob.glob(os.path.join(DATA_PATH, dataset, split, f"{dataset}*.csv"))
    assert len(files) == 1, f"Expected one {dataset}*.csv for {split}"
    return pd.read_csv(files[0])


class DLRRecDataset(Dataset):
    """IntuRec prompt + optional discrete reasoning codes + target title.

    reasoning="codes": target = <|Reason|> codes <|EndReason|> <|StartPredict|> title <|EndPredict|>
    reasoning="none":  target = <|StartPredict|> title <|EndPredict|>   (control)
    Labels cover the whole target; `code_mask` marks the reasoning part so code and
    title losses can be reported separately. Test samples hold only the prompt.
    """

    def __init__(self, dataset, tokenizer, split, reasoning="codes", sample=-1, seed=0):
        frame = read_split(dataset, split)
        self.row_ids = list(range(len(frame)))
        if sample > 0:
            self.row_ids = random.Random(seed).sample(self.row_ids, sample)
        codes = load_row_codes(split) if reasoning == "codes" else [[] for _ in range(len(frame))]
        self.data = [{"his_items": eval(frame.history_item_title.iat[i]),
                      "tgt_item": "\"" + str(frame.item_title.iat[i]) + "\"",
                      "codes": codes[i]} for i in self.row_ids]
        self.tokenizer = Tokenizer(tokenizer)
        self.split, self.reasoning = split, reasoning
        category = CATEGORY[dataset]
        self.instruction = (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request. \n\n### Instruction: \n"
            f"Given a list of {category} the user recetenly enjoy, please write a new {category} that the user may bought\n")
        self.category = category
        enc = lambda t: self.tokenizer.encode(t, bos=False, eos=False)
        self.reason_tok, self.end_reason_tok = enc(REASON), enc(END_REASON)
        self.start_tok, self.end_tok = enc(START_PREDICT), enc(END_PREDICT)
        self.process()

    def prompt_ids(self, row):
        history = ", ".join(f'"{t}"' for t in row["his_items"])
        text = (self.instruction + f"### User Input: \nThe user has palyed the following {self.category}s before: "
                f"{history}.\n" + "### Response: \n")
        return self.tokenizer.encode(text, bos=True, eos=False)

    def reasoning_ids(self, row):
        if self.reasoning != "codes":
            return []
        return self.reason_tok + [self.tokenizer.tokenizer.convert_tokens_to_ids(code_token(k))
                                  for k in row["codes"]] + self.end_reason_tok

    def process(self):
        pad = self.tokenizer.tokenizer.pad_token_id
        parts = []
        for row in self.data:
            prompt = self.prompt_ids(row)
            reason = self.reasoning_ids(row)
            title = self.start_tok + self.tokenizer.encode(row["tgt_item"], bos=False, eos=False) + self.end_tok
            parts.append((prompt, reason, title))
        if self.split == "test":
            max_len = max(len(p) for p, _, _ in parts)
        else:
            max_len = max(len(p) + len(r) + len(t) for p, r, t in parts)
        self.samples = []
        for prompt, reason, title in parts:
            if self.split == "test":
                ids, labels, cmask = prompt, None, None
            else:
                ids = prompt + reason + title
                labels = [-100] * len(prompt) + reason + title
                cmask = [0] * len(prompt) + [1] * len(reason) + [0] * len(title)
            padn = max_len - len(ids)
            s = {"input_ids": np.array([pad] * padn + ids), "attention_mask": np.array([0] * padn + [1] * len(ids))}
            if labels is not None:
                s["labels"] = np.array([-100] * padn + labels)
                s["code_mask"] = np.array([0] * padn + cmask)
            self.samples.append(s)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
