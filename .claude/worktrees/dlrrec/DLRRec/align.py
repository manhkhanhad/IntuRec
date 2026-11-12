"""DLRRec stage 1: latent-text alignment (DLR, Chang et al. 2026, stage 1).

Adds the reasoning vocabulary (<|Reason|>, <|EndReason|>, <|StartPredict|>, <|EndPredict|>,
one <|m_k|> per motive prototype) to the base LLM, initialises each code embedding as the
mean token embedding of its canonical motive text, then trains ONLY the new embedding rows
(the LLM stays frozen; Qwen2.5 ties input/output embeddings, so this also sets the output
rows) so that the frozen LLM continues a code prefix with the right text:
  prototype targets  <|Reason|> <|m_k|> <|EndReason|> -> "motive: reason" of prototype k
  row targets        <|Reason|> codes(row) <|EndReason|> -> that train row's own rationale
Saves the aligned model + tokenizer to DLRRec/checkpoints/align/.
"""

import argparse
import json
import os
import random
import time

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM, AutoTokenizer

from DLRRec.dlrrec_data import (END_REASON, RATIONALES, REASON, add_reasoning_tokens, code_ids, load_prototypes,
                                load_row_codes)

HERE = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base_model", default="Qwen/Qwen2.5-1.5B")
    p.add_argument("--output_dir", default=os.path.join(HERE, "checkpoints", "align"))
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--proto_repeats", type=int, default=20, help="Times each prototype target appears per epoch.")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--max_text_len", type=int, default=64)
    p.add_argument("--sample", type=int, default=-1, help="Use only this many train rows (smoke tests).")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda"
    protos = load_prototypes()
    tok = AutoTokenizer.from_pretrained(args.base_model)
    new_tokens = add_reasoning_tokens(tok, len(protos))
    new_ids = tok.convert_tokens_to_ids(new_tokens)
    model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.bfloat16).to(device)
    # Qwen's embedding matrix has spare rows beyond its tokenizer, so new ids can land inside the
    # original matrix: always address the new tokens by id, never by "rows past the old size".
    model.resize_token_embeddings(max(len(tok), model.get_input_embeddings().weight.shape[0]))
    model.config.pad_token_id = tok.eos_token_id
    emb = model.get_input_embeddings().weight
    assert model.config.tie_word_embeddings, "expects tied input/output embeddings"

    # Warm start: each code = mean embedding of its canonical text; boundary tokens = mean of all codes.
    codes = code_ids(tok, len(protos))
    with torch.no_grad():
        for k, text in enumerate(protos):
            ids = tok(text, add_special_tokens=False).input_ids
            emb[codes[k]] = emb[ids].mean(0)
        for i in set(new_ids) - set(codes):
            emb[i] = emb[codes].mean(0)

    # Only the new embedding rows learn.
    for p in model.parameters():
        p.requires_grad_(False)
    emb.requires_grad_(True)
    keep = torch.zeros(emb.shape[0], 1, device=device, dtype=emb.dtype)
    keep[new_ids] = 1
    emb.register_hook(lambda g: g * keep)
    opt = torch.optim.AdamW([emb], lr=args.lr, weight_decay=0.0)

    # Alignment examples: (prefix ids, target text ids).
    reason, end_reason = tok.convert_tokens_to_ids(REASON), tok.convert_tokens_to_ids(END_REASON)
    enc = lambda t: tok(t, add_special_tokens=False).input_ids[:args.max_text_len] + [tok.eos_token_id]
    examples = [([reason, codes[k], end_reason], enc(text)) for k, text in enumerate(protos)] * args.proto_repeats
    row_codes = load_row_codes("train")
    reasons = {}
    with open(RATIONALES.format(split="train"), encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            reasons[r["row"]] = r["reason"]
    rows = list(range(len(row_codes)))
    if args.sample > 0:
        rows = random.sample(rows, args.sample)
    examples += [([reason] + [codes[k] for k in row_codes[i]] + [end_reason], enc(reasons[i]))
                 for i in rows if row_codes[i]]
    print(f"alignment examples: {len(examples)} ({len(protos)} prototypes x {args.proto_repeats}, "
          f"{len(examples) - len(protos) * args.proto_repeats} train rows)", flush=True)

    model.train()
    step = 0
    for epoch in range(args.epochs):
        random.shuffle(examples)
        losses, t0 = [], time.time()
        for start in range(0, len(examples), args.batch_size):
            batch = examples[start:start + args.batch_size]
            ids = [torch.tensor(p + t) for p, t in batch]
            labels = [torch.tensor([-100] * len(p) + t) for p, t in batch]
            ids = pad_sequence(ids, batch_first=True, padding_value=tok.eos_token_id).to(device)
            labels = pad_sequence(labels, batch_first=True, padding_value=-100).to(device)
            mask = (labels != -100) | (torch.arange(ids.shape[1], device=device)[None] < torch.tensor(
                [len(p) for p, _ in batch], device=device)[:, None])
            loss = model(input_ids=ids, attention_mask=mask.long(), labels=labels).loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
            step += 1
            if step % 100 == 0:
                print(f"epoch {epoch + 1} step {step}: loss {np.mean(losses[-100:]):.4f} "
                      f"({time.time() - t0:.0f}s)", flush=True)
        print(f"epoch {epoch + 1}: mean loss {np.mean(losses):.4f}", flush=True)

    # Quick check: greedy continuation of a few code prefixes.
    model.eval()
    with torch.no_grad():
        for k in random.sample(range(len(protos)), 3):
            prefix = torch.tensor([[reason, codes[k], end_reason]], device=device)
            out = model.generate(prefix, max_new_tokens=30, do_sample=False, pad_token_id=tok.eos_token_id)
            print(f"code {k}: target '{protos[k][:80]}'\n   generated '{tok.decode(out[0, 3:], skip_special_tokens=True)[:80]}'")

    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tok.save_pretrained(args.output_dir)
    print(f"saved aligned model to {args.output_dir}")


if __name__ == "__main__":
    main()
