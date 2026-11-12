import glob
import json
import os
import random

import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from tqdm import tqdm

from data.IntuRecF2Dataset import Tokenizer

CATEGORY = {"Office_Products": "office products", "Books": "books", "steam": "games", "CDs_and_Vinyl": "musics",
            "Toys_and_Games": "toys and games", "Video_Games": "video games", "Musical_Instruments": "music instruments",
            "Sports_and_Outdoors": "sports and outdoors", "Pet_Supplies": "pet supplies",
            "Arts_Crafts_and_Sewing": "arts products", "Movies": "movie", "yelp": "resturant"}


def user_fold(user_ids, n):
    """Stable user -> fold in [0, n) (crc32 of the user id), so folds agree across scripts and runs."""
    import zlib
    return np.array([zlib.crc32(str(u).encode()) % n for u in user_ids])


class CollabRecDataset(Dataset):
    """Rows of the split CSV with the IntuRec prompt, plus collaborative reasoning supports.

    support_dir (collab/build_stage2_support.py output) holds, per split,
    support_rows_<split>.npy (train-row indices whose rationales are the supports).
      support_mode="text": returns rationale TEXT tokens (left-padded to max_rationale_len) for the
        supports, and for train/valid the row's own rationale plus aux_negatives random train
        rationales (aux-loss targets, never inputs). Texts come from
        <data_path>/<dataset>/collab/rationales_facet_<split>_<rationale_tag>.jsonl.
      If support_dir also holds support_texts.json (e.g. the motive-prototype corpus), support ids
        index into those texts instead of the train rationales; negatives and the own-rationale
        target still come from the train rationales.
      support_mode="emb": returns rationale_emb_train.npy vectors and own_rationale_<split>.npy.
    """

    def __init__(self, data_path, dataset, tokenizer, support_dir=None, support_k=10, sample=-1,
                 dataset_type="train", reason_step=1, seed=0, support_mode="emb", rationale_tag="v5",
                 max_rationale_len=64, aux_negatives=63, oracle="", sup_anchor="", collab_queue=False,
                 motive_source="", noisy_precision=1.0, motive_scores="cls", train_fold="",
                 utility_files="", utility_margin=0.0):
        csv_files = glob.glob(os.path.join(f"{data_path}/{dataset}/{dataset_type}", f"{dataset}*.csv"))
        assert len(csv_files) == 1, f"Expected one {dataset}*.csv in {data_path}/{dataset}/{dataset_type}"
        frame = pd.read_csv(csv_files[0])

        self.row_ids = list(range(len(frame)))
        if train_fold and dataset_type == "train":
            # "i/n": train on the users NOT in fold i (cross-fitted judges; see user_fold)
            i, n = map(int, train_fold.split("/"))
            folds = user_fold(frame.user_id, n)
            self.row_ids = [r for r in self.row_ids if folds[r] != i]
        if sample > 0:
            self.row_ids = random.Random(seed).sample(self.row_ids, sample)
        self.data = [{"his_items": eval(frame.history_item_title.iat[i]),
                      "tgt_item": "\"" + str(frame.item_title.iat[i]) + "\""} for i in self.row_ids]

        # Collaborative InfoNCE labels (train only; never model inputs): user, target item, row id and
        # the row's motive prototype ids (motive_corpus/row_motives_train.json, -1 padded to 2).
        self.cq = None
        if collab_queue and dataset_type == "train":
            with open(os.path.join(data_path, dataset, "collab", "motive_corpus", "row_motives_train.json"),
                      encoding="utf-8") as f:
                motives = json.load(f)
            self.cq = {"user": pd.factorize(frame.user_id)[0].astype(np.int64),
                       "target": frame.item_id.to_numpy(dtype=np.int64),
                       "motives": np.array([(m + [-1, -1])[:2] for m in motives], dtype=np.int64)}

        self.tokenizer = Tokenizer(tokenizer)
        self.support_mode = support_mode
        self.aux_negatives = aux_negatives
        self.rng = np.random.default_rng(seed)
        self.support_rows = self.rationale_emb = self.own_rationale = None
        self.train_rationale_tokens = self.own_rationale_tokens = self.support_table = None
        if support_dir and support_k > 0:
            self.support_rows = np.load(os.path.join(support_dir, f"support_rows_{dataset_type}.npy"))[:, :support_k]
        if support_dir and support_mode == "text":
            collab_dir = os.path.join(data_path, dataset, "collab")
            self.train_rationale_tokens = self.tokenize_rationales(
                os.path.join(collab_dir, f"rationales_facet_train_{rationale_tag}.jsonl"), max_rationale_len)
            self.support_table = self.train_rationale_tokens
            texts_path = os.path.join(support_dir, "support_texts.json")
            if os.path.exists(texts_path):
                with open(texts_path, encoding="utf-8") as f:
                    self.support_table = self.tokenize_texts(json.load(f), max_rationale_len)
            if dataset_type == "train":
                self.own_rationale_tokens = self.train_rationale_tokens
            elif dataset_type == "valid":
                self.own_rationale_tokens = self.tokenize_rationales(
                    os.path.join(collab_dir, f"rationales_facet_valid_{rationale_tag}.jsonl"), max_rationale_len)
        elif support_dir:
            if self.support_rows is not None:
                self.rationale_emb = np.load(os.path.join(support_dir, "rationale_emb_train.npy"), mmap_mode="r")
            own_path = os.path.join(support_dir, f"own_rationale_{dataset_type}.npy")
            if dataset_type != "test" and os.path.exists(own_path):
                self.own_rationale = np.load(own_path, mmap_mode="r")

        # Oracle analysis only (text mode): the support is each row's OWN ground truth, written with
        # the target known, for every split. "own": its rationale text (k=1); "own_motive": the texts
        # of its 1-2 motive prototypes (k=2; one motive is repeated, none -> "No specific motive.").
        if oracle:
            collab_dir = os.path.join(data_path, dataset, "collab")
            if oracle == "own":
                self.support_table = self.tokenize_rationales(
                    os.path.join(collab_dir, f"rationales_facet_{dataset_type}_{rationale_tag}.jsonl"), max_rationale_len)
                self.support_rows = np.arange(len(self.support_table))[:, None]
            elif oracle == "own_motive":
                texts, rows = self.motive_texts(collab_dir, dataset_type)
                self.support_table = self.tokenize_texts(texts, max_rationale_len)
                self.support_rows = rows
            elif oracle == "own_motive_plus_pred":
                # The row's 2 true motive prototypes + the motive retriever's top (support_k - 2) predicted
                # prototypes that are not true ones (train: out-of-fold scores), shuffled per row.
                texts, rows = self.motive_texts(collab_dir, dataset_type)
                scores = np.load(os.path.join(collab_dir, "motive_corpus", "retriever", "cls",
                                              f"scores_{dataset_type}.npy")).astype(np.float32)
                n_pred = support_k - rows.shape[1]
                combined = []
                for r, true in enumerate(rows):
                    pred = [int(i) for i in np.argsort(-scores[r]) if i not in set(true.tolist())][:n_pred]
                    ids = np.array(list(true) + pred)
                    np.random.default_rng(seed + r).shuffle(ids)
                    combined.append(ids)
                self.support_table = self.tokenize_texts(texts, max_rationale_len)
                self.support_rows = np.array(combined)
            else:
                raise ValueError(f"unknown oracle {oracle!r}")

        # Motive-utilization supports (text mode), support_k slots of motive prototype texts, with
        # per-slot retriever log-probabilities (support_logp) and, for train/valid, whether the slot is
        # one of the row's true motives (support_is_true; selection-loss label, never an input).
        #   "predicted": the retriever's top-k motives (train: out-of-fold scores) -- deployable.
        #   "noisy":     noisy oracle: each true motive is kept with probability noisy_precision, else
        #                replaced by the retriever's highest-ranked non-true motive; remaining slots are
        #                filled with further non-true predictions; slots shuffled. Train rows are
        #                re-drawn on every access (augmentation); valid/test are fixed per row.
        self.motive_source, self.noisy_precision, self.seed = motive_source, noisy_precision, seed
        if motive_source:
            assert support_mode == "text" and support_k > 0 and not oracle
            corpus = os.path.join(data_path, dataset, "collab", "motive_corpus")
            texts, _ = self.motive_texts(os.path.join(data_path, dataset, "collab"), dataset_type)
            self.support_table = self.tokenize_texts(texts, max_rationale_len)
            with open(os.path.join(corpus, f"row_motives_{dataset_type}.json"), encoding="utf-8") as f:
                self.true_motives = json.load(f)
            scores = np.load(os.path.join(corpus, "retriever", motive_scores,
                                          f"scores_{dataset_type}.npy")).astype(np.float32)
            scores -= scores.max(1, keepdims=True)
            self.motive_logp = scores - np.log(np.exp(scores).sum(1, keepdims=True))
            # Enough ranked candidates to fill support_k slots after skipping up to 2 true motives.
            self.pred_order = np.argsort(-self.motive_logp, axis=1)[:, :support_k + 2]
            if motive_source == "predicted":
                self.support_rows = self.pred_order[:, :support_k]
            elif motive_source == "noisy":
                self.support_rows = None  # drawn in __getitem__
            else:
                raise ValueError(f"unknown motive_source {motive_source!r}")
            self.support_k = support_k
            # Gate labels from the utility judge (collab/motive_utility.py) instead of correctness:
            # slot label = u(m) > utility_margin, u from cross-fitted judges (comma-separated npz files,
            # rows = CSV rows of this split). Slots without a utility score keep the correctness label.
            self.utility = None
            if utility_files and dataset_type == "train":
                self.utility = {}
                for path in utility_files.split(","):
                    z = np.load(path)
                    null = {int(r): l for r, k, l in zip(z["row"], z["kind"], z["ll"]) if k == 0}
                    for r, k, m, l in zip(z["row"], z["kind"], z["motive"], z["ll"]):
                        if k in (2, 3):  # single true motive / predicted, both fed as [m, m]
                            self.utility[(int(r), int(m))] = float(l - null[int(r)])
                self.utility_margin = utility_margin
                print(f"utility labels: {len(self.utility)} (row, motive) scores from {utility_files}", flush=True)

        # Contrastive support loss (train/valid only): random supports in the same format, and the
        # anchor the thought should point at: the target title ("title", as IntuRec's tgt_tokens) or
        # the row's ground-truth motive prototype texts ("motive"). Never model inputs.
        self.sup_anchor = sup_anchor if dataset_type != "test" else ""
        self.anchor_table = self.anchor_rows = None
        if self.sup_anchor == "title":
            self.anchor_table = self.tokenize_texts([d["tgt_item"] for d in self.data], max_rationale_len)
        elif self.sup_anchor == "motive":
            texts, self.anchor_rows = self.motive_texts(os.path.join(data_path, dataset, "collab"), dataset_type)
            self.anchor_table = self.tokenize_texts(texts, max_rationale_len)
        elif self.sup_anchor:
            raise ValueError(f"unknown sup_anchor {sup_anchor!r}")

        self.dataset_type = dataset_type
        self.reason_step = reason_step
        category = CATEGORY[dataset]
        self.instruction = (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request. \n\n### Instruction: \n"
            f"Given a list of {category} the user recetenly enjoy, please write a new {category} that the user may bought\n")
        self.category = category
        self.data_process()

    @staticmethod
    def motive_texts(collab_dir, split):
        """Motive prototype texts (+ a 'no motive' entry) and each row's 2 prototype ids
        (one motive is repeated; rows without motives point at the 'no motive' entry)."""
        corpus = os.path.join(collab_dir, "motive_corpus")
        with open(os.path.join(corpus, "prototypes.json"), encoding="utf-8") as f:
            texts = [f"{p['motive']}: {p['reason']}" for p in json.load(f)] + ["No specific motive."]
        with open(os.path.join(corpus, f"row_motives_{split}.json"), encoding="utf-8") as f:
            motives = json.load(f)
        return texts, np.array([(m + m)[:2] if m else [len(texts) - 1] * 2 for m in motives])

    def motive_slots(self, row):
        """Motive prototype ids fed as supports for this row (motive_source != '')."""
        if self.motive_source == "predicted":
            return self.support_rows[row]
        # noisy oracle: train re-draws on every access, valid/test are fixed per row
        rng = self.rng if self.dataset_type == "train" else np.random.default_rng((self.seed, row))
        true = list(dict.fromkeys(self.true_motives[row]))[:self.support_k]
        wrong = [int(m) for m in self.pred_order[row] if m not in set(true)]
        slots = [t if rng.random() < self.noisy_precision else wrong.pop(0) for t in true]
        slots += wrong[:self.support_k - len(slots)]
        return np.array(slots, dtype=np.int64)[rng.permutation(self.support_k)]

    def tokenize_rationales(self, path, max_len):
        """Left-padded token ids [n_rows, max_len] of each row's rationale `reason` (row order)."""
        reasons = {}
        with open(path, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                if "error" not in rec:
                    reasons[rec["row"]] = rec["reason"]
        assert len(reasons) == max(reasons) + 1, f"{path} must cover every row"
        return self.tokenize_texts([reasons[row] for row in range(len(reasons))], max_len)

    def tokenize_texts(self, texts, max_len):
        """Left-padded token ids [len(texts), max_len]."""
        pad = self.tokenizer.tokenizer.pad_token_id
        out = np.full((len(texts), max_len), pad, dtype=np.int64)
        for i, text in enumerate(texts):
            ids = self.tokenizer.encode(text, bos=False, eos=False)[:max_len]
            out[i, max_len - len(ids):] = ids
        return out

    def __len__(self):
        return len(self.data)

    def prompt_parts(self, row):
        history_list = [f'\"{title}\"' for title in row["his_items"]]
        history = (f'### User Input: \n'
                   f'The user has palyed the following {self.category}s before: '
                   f'{", ".join(history_list)}.\n')
        answer_prefix = "### Response: \n"
        part1 = (self.tokenizer.encode(self.instruction, bos=True, eos=False)
                 + self.tokenizer.encode(history, bos=False, eos=False)
                 + self.tokenizer.encode(answer_prefix, bos=False, eos=False)
                 + self.thought_token * self.reason_step + self.start_token)
        part2 = self.tokenizer.encode(row["tgt_item"], bos=False, eos=False) + self.end_token
        text = self.instruction + history + answer_prefix + "<|Thought|>" * self.reason_step + "<|StartPredict|>"
        return part1, part2, text

    def data_process(self):
        self.thought_token = self.tokenizer.encode("<|Thought|>", bos=False, eos=False)
        self.start_token = self.tokenizer.encode("<|StartPredict|>", bos=False, eos=False)
        self.end_token = self.tokenizer.encode("<|EndPredict|>", bos=False, eos=False)
        parts = [self.prompt_parts(row) for row in tqdm(self.data)]
        if self.dataset_type == "test":
            max_len = max(len(p1) for p1, _, _ in parts)
        else:
            max_len = max(len(p1) + len(p2) for p1, p2, _ in parts)
        pad = self.tokenizer.tokenizer.pad_token_id

        self.data_output, self.text_data = [], []
        for row, (part1, part2, text) in zip(self.data, parts):
            if self.dataset_type == "test":
                ids = part1
                labels = None
            else:
                ids = part1 + part2
                labels = [-100] * (max_len - len(ids)) + [-100] * len(part1) + part2
            sample = {
                "input_ids": np.array([pad] * (max_len - len(ids)) + ids),
                "attention_mask": np.array([0] * (max_len - len(ids)) + [1] * len(ids)),
            }
            if labels is not None:
                sample["labels"] = np.array(labels)
            self.data_output.append(sample)
            self.text_data.append({"all_input": text + ("" if labels is None else row["tgt_item"] + "<|EndPredict|>"),
                                   "tgt_item": row["tgt_item"]})

    def __getitem__(self, idx):
        sample = dict(self.data_output[idx])
        row = self.row_ids[idx]
        if self.cq is not None:
            sample.update(cq_user=self.cq["user"][row], cq_target=self.cq["target"][row],
                          cq_sample=np.int64(row), cq_motives=self.cq["motives"][row])
        if self.support_mode == "text":
            if self.motive_source:
                slots = self.motive_slots(row)
                sample["support_tokens"] = self.support_table[slots]
                sample["support_logp"] = self.motive_logp[row, slots].astype(np.float32)
                if self.dataset_type != "test":
                    true = set(self.true_motives[row])
                    label = [float(s in true) for s in slots]
                    if self.utility is not None:
                        label = [float(self.utility[(row, int(s))] > self.utility_margin)
                                 if (row, int(s)) in self.utility else y for s, y in zip(slots, label)]
                    sample["support_is_true"] = np.array(label, dtype=np.float32)
            elif self.support_rows is not None:
                sample["support_tokens"] = self.support_table[self.support_rows[row]]
            if self.own_rationale_tokens is not None:
                sample["own_tokens"] = self.own_rationale_tokens[row]
                negs = self.rng.integers(0, len(self.train_rationale_tokens), self.aux_negatives)
                sample["neg_tokens"] = self.train_rationale_tokens[negs]
            if self.sup_anchor and self.support_rows is not None:
                rand = self.rng.integers(0, len(self.support_table), self.support_rows.shape[1])
                sample["rand_support_tokens"] = self.support_table[rand]
                sample["anchor_tokens"] = (self.anchor_table[idx][None] if self.sup_anchor == "title"
                                           else self.anchor_table[self.anchor_rows[row]])
            return sample
        if self.support_rows is not None:
            sample["support_emb"] = np.asarray(self.rationale_emb[self.support_rows[row]], dtype=np.float32)
        if self.own_rationale is not None:
            sample["own_rationale"] = np.asarray(self.own_rationale[row], dtype=np.float32)
        return sample
