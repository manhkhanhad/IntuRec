"""CPU check of run_collabrec.train's checkpoint/resume (tiny random Qwen2 + InfoNCE queue).

A run interrupted after a few optimizer steps (--stop_after_steps) and then resumed must end with the same
weights and queue as an uninterrupted run. Run from the repo root: python tests/test_resume.py
"""
import os
import shutil
import sys
import tempfile
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import Qwen2Config

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import run_collabrec as R  # noqa: E402
from model.CollabRecF2 import CollabRecF2  # noqa: E402
from run_inturec_f2 import initial_optimizer_scheduler  # noqa: E402


class Rows(Dataset):
    def __init__(self, n, seed):
        g = torch.Generator().manual_seed(seed)
        self.rows = []
        for i in range(n):
            ids = torch.randint(1, 60, (12,), generator=g)
            ids[6] = 64 - 3  # <|Thought|>
            labels = ids.clone()
            labels[:7] = -100
            self.rows.append(dict(input_ids=ids, attention_mask=torch.ones_like(ids), labels=labels,
                                  support_tokens=torch.randint(1, 60, (2, 5), generator=g),
                                  cq_user=torch.tensor(i % 3), cq_target=torch.tensor(i % 4),
                                  cq_sample=torch.tensor(i), cq_motives=torch.tensor([i % 5, -1])))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


class Tok:
    def save_pretrained(self, path):
        pass


def make(args):
    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                      num_attention_heads=4, num_key_value_heads=2, pad_token_id=0)
    cfg.support_mode = "text"
    cfg.collab_queue = {"relations": ["user", "target", "motive"], "bank_size": 16, "projection_dim": 16,
                        "temperature": 0.1, "min_token_accuracy": 0.0, "max_positives": 16, "num_negatives": 256}
    model = CollabRecF2(cfg).to(torch.bfloat16)
    model.collab_weights = {r: 0.1 for r in cfg.collab_queue["relations"]}
    model.beta, model.gamma, model.reason_step = 0.0, 0.0, 1
    accum = args.train_batch_size // args.train_micro_batch_size
    train = DataLoader(Rows(24, 1), batch_size=args.train_micro_batch_size, shuffle=True,
                       generator=torch.Generator().manual_seed(args.seed))
    valid = DataLoader(Rows(4, 2), batch_size=2)
    opt, sched = initial_optimizer_scheduler(model, lr=1e-3, wd=0.0, optimizer="adamw", epochs=args.nums_epoch,
                                             warmup_step=2, batchnum_per_epoch=(24 + accum - 1) // accum,
                                             scheduler_type="linear")
    return model, train, valid, opt, sched


def run(root, name, stop=0, resume_every=1):
    args = SimpleNamespace(ckpt_root=root, dataset="d", experiment="e", run_name=name, train_batch_size=4,
                           train_micro_batch_size=2, nums_epoch=2, seed=0, resume_every=resume_every,
                           stop_after_steps=stop, motive_source="", noisy_precision=1.0,
                           noisy_precision_start=-1.0)
    model, train, valid, opt, sched = make(args)
    try:
        R.train(model, Tok(), train, valid, opt, sched, torch.device("cpu"), args)
    except SystemExit:
        return None
    return model


def main():
    root = tempfile.mkdtemp()
    try:
        full = run(root, "full", resume_every=0)
        assert run(root, "resumed", stop=4) is None  # 6 steps per epoch: stops mid-epoch 1
        assert os.path.exists(os.path.join(root, "d", "e", "resumed", "resume", "state.pt"))
        assert run(root, "resumed", stop=5) is None  # second interruption, now in epoch 2
        resumed = run(root, "resumed")
        for (k, a), (_, b) in zip(full.state_dict().items(), resumed.state_dict().items()):
            assert torch.equal(a, b), f"weights differ after resume: {k}"
        for k in R.BANK_BUFFERS:
            assert torch.equal(getattr(full, k), getattr(resumed, k)), f"queue differs after resume: {k}"
        print("ok: interrupted + resumed training ends with identical weights and queue")
    finally:
        shutil.rmtree(root)


if __name__ == "__main__":
    main()
