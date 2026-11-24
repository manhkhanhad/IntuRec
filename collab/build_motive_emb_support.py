#!/usr/bin/env python3
"""Embedding-mode supports for MotiveRec (run_collabrec.py --support_mode emb), no code change needed.

Text-mode MotiveRec reads motive_corpus/stage2/ (support_texts.json = "<motive>: <reason>" per prototype,
support_rows_<split>.npy = top-k predicted prototype ids). Emb mode expects one table
rationale_emb_train.npy that is BOTH the support table (indexed by support_rows) and the InfoNCE negative
bank, plus own_rationale_<split>.npy (alignment target). To keep the alignment target and negatives exactly
as in the collaborative emb runs (the row's own v5 rationale vs train rationales, all-mpnet-base-v2,
centred on the train-rationale mean and normalised), the table is

    [ train rationales (n_train rows, = collab/stage2/rationale_emb_train.npy) ; prototypes (K rows) ]

and support ids are shifted by n_train. Prototype texts are embedded with the same encoder and centred with
the same mean, so supports and the target share one space. The K prototype rows also enter the negative
bank (K / n_train < 2%).

Writes <data>/<dataset>/collab/motive_corpus/stage2_emb/.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "collab"))
from eval_support import centre  # noqa: E402

ENCODER = "sentence-transformers/all-mpnet-base-v2"

for dataset in (sys.argv[1:] or ["CDs_and_Vinyl", "Toys_and_Games"]):
    collab = ROOT / "data" / dataset / "collab"
    src, base, out = collab / "motive_corpus" / "stage2", collab / "stage2", collab / "motive_corpus" / "stage2_emb"
    raw = np.load(collab / "hyp_cache" / "reason_v5.npy").astype(np.float32)
    mean = raw.mean(0)
    train_emb = np.load(base / "rationale_emb_train.npy")
    assert np.allclose(centre(raw, mean), train_emb, atol=1e-4), "stage2 rationale table is not centre(reason_v5)"
    texts = json.loads((src / "support_texts.json").read_text())
    from sentence_transformers import SentenceTransformer
    proto = SentenceTransformer(ENCODER, device="cpu").encode(texts, batch_size=64, convert_to_numpy=True,
                                                              show_progress_bar=False).astype(np.float32)
    proto = centre(proto, mean)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "rationale_emb_train.npy", np.concatenate([train_emb, proto]).astype(np.float32))
    for split in ("train", "valid", "test"):
        np.save(out / f"support_rows_{split}.npy", np.load(src / f"support_rows_{split}.npy") + len(train_emb))
    for split in ("train", "valid"):
        np.save(out / f"own_rationale_{split}.npy", np.load(base / f"own_rationale_{split}.npy"))
    (out / "support_texts_ref.json").write_text(json.dumps({"n_train": len(train_emb), "n_prototypes": len(texts),
                                                            "encoder": ENCODER, "source": str(src)}))
    # sanity: nearest train rationale of each prototype should be about the same motive
    sim = proto @ train_emb.T
    print(f"{dataset}: table {len(train_emb)}+{len(texts)}, proto-to-nearest-rationale cos "
          f"{sim.max(1).mean():.3f}, proto-to-proto mean cos {(proto @ proto.T).mean():.3f}")
