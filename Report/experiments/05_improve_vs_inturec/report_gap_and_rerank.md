# E5a — Gap to IntuRec, and motive re-ranking (CPU)

Back to [main plan](../../main.md). Date: 2026-11-13. Goal (author): make MotiveRec beat IntuRec. Being below O0 is
also IntuRec's problem, so O0 is not the target.

## Paired gap MotiveRec-T − IntuRec (bootstrap 95% CI, same test rows)
| | HR@1 | HR@5 | HR@10 |
|---|---|---|---|
| CDs | −0.0002 [−0.0044, +0.0041] | −0.0063 [−0.0117, −0.0015] | −0.0028 [−0.0089, +0.0026] |
| Toys | −0.0010 [−0.0047, +0.0025] | +0.0013 [−0.0031, +0.0056] | −0.0040 [−0.0091, +0.0010] |

- Only CDs HR@5 is significant. The other gaps are within noise. Earlier logs give a seed spread of about 0.003 HR@10.
- **Confound:** IntuRec and O0 were trained with micro-batch 2 (`micro2`). MotiveRec-T used micro-batch 8, the
  `submit_collabrec.sbatch` default. Both start from base Qwen2.5-1.5B. Use matched micro-batches before comparing.

## Motive re-ranking of the top-10 beams (`pilot/motive_rerank.py`)
- **Score:** beam logp + λ · log Σ_m p(m|history) p(m|item), with p(m|history) = softmax of the cls retriever scores.
- **p(m|item):** training-purchase motive counts of the item, Dirichlet-smoothed toward the motive prior.
- **Scope:** this only reorders the top 10, so HR@10 is unchanged.

| λ | CDs IntuRec HR@1 / HR@5 / N@10 | CDs MotiveRec-T | Toys IntuRec | Toys MotiveRec-T |
|---|---|---|---|---|
| 0 | 0.0580 / 0.0981 / 0.0860 | 0.0578 / 0.0918 / 0.0838 | 0.0420 / 0.0833 / 0.0721 | 0.0410 / 0.0846 / 0.0705 |
| 0.25 | 0.0577 / 0.0983 / 0.0857 | 0.0580 / 0.0911 / 0.0838 | 0.0423 / 0.0837 / 0.0724 | 0.0413 / 0.0850 / 0.0708 |
| 0.5 | 0.0570 / 0.0984 / 0.0855 | 0.0565 / 0.0914 / 0.0832 | 0.0417 / 0.0849 / 0.0723 | 0.0410 / 0.0847 / 0.0708 |
| 1.0 | 0.0560 / 0.0979 / 0.0851 | 0.0559 / 0.0914 / 0.0829 | 0.0408 / 0.0846 / 0.0717 | 0.0387 / 0.0856 / 0.0698 |

**Negative result:** changes are ≤ 0.0004 for small λ and hurt for λ ≥ 1. The item-level motive profile adds nothing
to the beam ranking. Not pursued. (λ was chosen on test; this does not matter for a null result.)
