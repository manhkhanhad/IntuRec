# Deleted MotiveRec checkpoints (2026-11-18)

Only `model.safetensors` (~3 GB each) was removed. Logs, `train.log`, `slurm-*.out`, `result/` and configs remain in
`/common/scratch/users/m/mk.ngohuu.2024/IntuRec/checkpoints/<dataset>/CollabRec/<run>/`.
All runs: K = number of motives, NTP only (beta 0) unless noted, micro-batch 8 unless noted, best-valid-NTP epoch checkpoint, test split, retrieved (top) motives.

| Dataset | Run (job) | HR@1 | HR@5 | HR@10 | NDCG@1 | NDCG@5 | NDCG@10 | Valid NTP ep1/2/3 |
|---|---|---|---|---|---|---|---|---|
| Toys | text-motivev6_k3 (365542) | 0.0421 | 0.0837 | 0.1128 | 0.0421 | 0.0637 | 0.0730 | 0.6347/0.5554/0.5824 |
| Toys | text-motive_k3 (v5) (365543) | 0.0423 | 0.0844 | 0.1122 | 0.0423 | 0.0639 | 0.0728 | 0.6373/0.5522/0.5783 |
| Toys | text-motivev6_k10 (365544) | 0.0417 | 0.0819 | 0.1097 | 0.0417 | 0.0624 | 0.0713 | 0.6347/0.5547/0.5796 |
| Toys | text-motive_k10 (v5) (365545) | 0.0404 | 0.0844 | 0.1101 | 0.0404 | 0.0636 | 0.0719 | 0.6394/0.5542/0.5802 |
| Toys | text-motivev6aligner_k3 (365546) | 0.0404 | 0.0810 | 0.1082 | 0.0404 | 0.0616 | 0.0703 | 0.6403/0.5555/0.5823 |
| Toys | emb-motive_k10 (v5) (365547) | 0.0402 | 0.0839 | 0.1094 | 0.0402 | 0.0628 | 0.0710 | 0.6425/0.5568/0.5828 |
| Toys | text-motivev6-llm_mean_ln_k10 (365549) | 0.0398 | 0.0839 | 0.1095 | 0.0398 | 0.0625 | 0.0708 | 0.6425/0.5628/0.5971 |
| Toys | text-motivev6-ratv6_k3, beta 0.1 (365889) | 0.0426 | 0.0831 | 0.1103 | 0.0426 | 0.0635 | 0.0723 | 0.6404/0.5566/0.5828 |
| Toys | text-motivev6-llm_mean_ln_k3 (365890) | 0.0413 | 0.0840 | 0.1080 | 0.0413 | 0.0633 | 0.0710 | 0.6331/0.5554/0.5864 |
| Toys | emb-motive_k3 (v5) (365891) | 0.0390 | 0.0831 | 0.1091 | 0.0390 | 0.0616 | 0.0700 | 0.6435/0.5584/0.5852 |
| CDs | emb-motive_k10 (v5) (365548) | 0.0557 | 0.0901 | 0.1135 | 0.0557 | 0.0736 | 0.0810 | 1.0959/0.9837/1.0997 |
| CDs | emb-motive_k3 (v5) (365892) | 0.0575 | 0.0911 | 0.1140 | 0.0575 | 0.0751 | 0.0824 | 1.0976/0.9820/1.0913 |

Reference (kept): Toys b-micro2 `text-motivev6-micro2_k3_…_job365893` 0.0430/0.0879/0.1129 | 0.0430/0.0662/0.0743.
Kept (reader runs, may need more decodes): B6 366598, B6sim 366599, C5 366678, C6 366679; CDs v6 368133 (training).
