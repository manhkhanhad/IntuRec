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
            "Arts_Crafts_and_Sewing": "arts products", "Movies": "movie", "yelp": "resturant",
            "Steam": "games", "Movielens": "movie"}


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
                 max_rationale_len=64, aux_negatives=63, oracle="", sup_anchor="",
                 motive_corpus="", insert_p=0.0, insert_fill="pred", eval_motives="top", decode_format=False,
                 noisy_precision=-1.0, sel_labels=False):
        csv_files = glob.glob(os.path.join(f"{data_path}/{dataset}/{dataset_type}", f"{dataset}*.csv"))
        assert len(csv_files) == 1, f"Expected one {dataset}*.csv in {data_path}/{dataset}/{dataset_type}"
        frame = pd.read_csv(csv_files[0])

        self.row_ids = list(range(len(frame)))
        if sample > 0:
            self.row_ids = random.Random(seed).sample(self.row_ids, sample)
        self.data = [{"his_items": eval(frame.history_item_title.iat[i]),
                      "tgt_item": "\"" + str(frame.item_title.iat[i]) + "\""} for i in self.row_ids]

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
            texts_path = os.path.join(support_dir, "support_texts.json")
            train_rat = os.path.join(collab_dir, f"rationales_facet_train_{rationale_tag}.jsonl")
            # Steam/Movielens have no IntuRec facet rationales. With motive support texts they are only the
            # L_collab targets (own/neg tokens, used only when beta > 0), so they may be absent.
            has_rationales = os.path.exists(train_rat)
            assert has_rationales or os.path.exists(texts_path), f"need {train_rat} or {texts_path}"
            if has_rationales:
                self.train_rationale_tokens = self.tokenize_rationales(train_rat, max_rationale_len)
                self.support_table = self.train_rationale_tokens
            if os.path.exists(texts_path):
                with open(texts_path, encoding="utf-8") as f:
                    self.support_table = self.tokenize_texts(json.load(f), max_rationale_len)
            if not has_rationales:
                pass
            elif dataset_type == "train":
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

        # Motive insertion + LM-side motive head (reader B recipe, GRPO session 365709), on a motive corpus
        # whose prototype ids index support_texts.json (e.g. motive_corpus_v6/P1000). support_rows = the
        # retriever's top-k (train: cross-fitted). Train only, with prob. insert_p per access: a row whose top-k
        # holds none of its true motives gets its true motives + fillers ("pred": the retriever's next non-true
        # picks; "similar": the prototypes nearest its first true motive, centroid cosine), shuffled.
        # motive_target (train/valid): the row's true motive ids as 2 slots (one repeated; none -> K = "no motive").
        # eval_motives (valid/test decode diagnostics): top | true (true first, then top non-true) | null | random.
        self.motive_corpus, self.insert_p, self.insert_fill = motive_corpus, insert_p, insert_fill
        # Noisy oracle (CollabRec 356575; noisy_precision >= 0, train/valid only): each true motive is kept with
        # prob. noisy_precision, else replaced by the retriever's highest-ranked non-true motive; remaining slots
        # are filled with further non-true predictions; slots shuffled. Train re-drawn per access, valid fixed per
        # row. sel_labels: emit support_is_true (slot is a true motive) for the gated readout's selection BCE.
        self.noisy_precision, self.sel_labels, self.seed = noisy_precision, sel_labels, seed
        self.decode_format = decode_format
        self.true_motives = self.motive_target = None
        if motive_corpus:
            assert support_mode == "text" and self.support_rows is not None and not oracle
            corpus = os.path.join(data_path, dataset, "collab", motive_corpus)
            with open(os.path.join(corpus, f"row_motives_{dataset_type}.json"), encoding="utf-8") as f:
                self.true_motives = json.load(f)
            n_proto = len(self.support_table)
            assert n_proto == len(json.load(open(os.path.join(corpus, "prototypes.json"), encoding="utf-8")))
            self.n_motives = n_proto + 1
            self.motive_target = np.array([(m + m)[:2] if m else [n_proto] * 2 for m in self.true_motives],
                                          dtype=np.int64)
            k = self.support_rows.shape[1]
            full = np.load(os.path.join(support_dir, f"support_rows_{dataset_type}.npy"))  # ranked, >= k + 2
            self.pred_order = full
            if insert_fill == "similar":
                c = np.load(os.path.join(corpus, "centroids.npy")).astype(np.float32)
                c /= np.linalg.norm(c, axis=1, keepdims=True)
                sim = c @ c.T
                np.fill_diagonal(sim, -np.inf)
                self.similar = np.argsort(-sim, axis=1)[:, :k + 2]
            if dataset_type != "train" and eval_motives != "top":
                null_id = n_proto
                self.support_table = np.concatenate(
                    [self.support_table, self.tokenize_texts(["No specific motive."], max_rationale_len)])
                if eval_motives == "true":
                    rows = []
                    for r, t in enumerate(self.true_motives):
                        t = list(dict.fromkeys(t))[:k]
                        rows.append(t + [int(m) for m in full[r] if m not in set(t)][:k - len(t)])
                    self.support_rows = np.array(rows, dtype=np.int64)
                elif eval_motives == "null":
                    self.support_rows = np.full((len(self.true_motives), k), null_id, np.int64)
                elif eval_motives == "random":
                    self.support_rows = np.random.default_rng(seed).integers(0, n_proto, (len(self.true_motives), k))
                else:
                    raise ValueError(f"unknown eval_motives {eval_motives!r}")

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
        test_format = self.dataset_type == "test" or self.decode_format
        if test_format:
            max_len = max(len(p1) for p1, _, _ in parts)
        else:
            max_len = max(len(p1) + len(p2) for p1, p2, _ in parts)
        pad = self.tokenizer.tokenizer.pad_token_id

        self.data_output, self.text_data = [], []
        for row, (part1, part2, text) in zip(self.data, parts):
            if test_format:
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

    def motive_slots(self, row):
        """Support prototype ids for this row; train rows may get their true motives inserted (insert_p)."""
        top = self.support_rows[row]
        if self.true_motives is not None and self.noisy_precision >= 0 and self.dataset_type != "test" \
                and not self.decode_format:
            k = len(top)
            rng = self.rng if self.dataset_type == "train" else np.random.default_rng((self.seed, row))
            true = list(dict.fromkeys(self.true_motives[row]))[:k]
            wrong = [int(m) for m in self.pred_order[row] if m not in set(true)]
            slots = [t if rng.random() < self.noisy_precision else wrong.pop(0) for t in true]
            slots += wrong[:k - len(slots)]
            return np.array(slots, dtype=np.int64)[rng.permutation(k)]
        if self.true_motives is None or self.dataset_type != "train" or self.insert_p <= 0:
            return top
        k = len(top)
        true = list(dict.fromkeys(self.true_motives[row]))[:k]
        if not true or set(true) & set(top.tolist()) or self.rng.random() >= self.insert_p:
            return top
        pool = self.similar[true[0]] if self.insert_fill == "similar" else self.pred_order[row]
        fill = [int(m) for m in pool if m not in set(true)][:k - len(true)]
        return np.array(true + fill, dtype=np.int64)[self.rng.permutation(k)]

    def __getitem__(self, idx):
        sample = dict(self.data_output[idx])
        row = self.row_ids[idx]
        if self.support_mode == "text":
            if self.support_rows is not None:
                slots = self.motive_slots(row)
                sample["support_tokens"] = self.support_table[slots]
                if self.sel_labels and self.true_motives is not None and "labels" in sample:
                    true = set(self.true_motives[row])
                    sample["support_is_true"] = np.array([float(s in true) for s in slots], dtype=np.float32)
            if self.motive_target is not None and "labels" in sample:
                sample["motive_target"] = self.motive_target[row]
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
