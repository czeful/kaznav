# Calibration qwen3vl_4b_devall_vacuum_cmdq.json

backend qwen3vl_4b (qwen3-vl-4b), verify prompt 5728face305a, runs: runs/matrix/dev_calibration/qwen_kk/seed0, runs/matrix/dev_calibration/qwen_en/seed0, runs/matrix/dev_calibration/qwen_kkru/seed0, runs/matrix/dev_demo/qwen_kk/seed0, runs/matrix/dev_demo/qwen_en/seed0, runs/matrix/dev_demo/qwen_kkru/seed0, runs/matrix/dev_confirm/qwen_kk_cmdq/seed0 (dev)

1768 calls, 281 show the goal (base rate 0.159); 279 dev episodes; label: content

| | ECE | Brier |
|---|---|---|
| raw P(yes) | 0.143 | 0.143 |
| platt, fitted and judged on the same calls | 0.020 | 0.085 |
| platt, out of fold (5 folds by episode) | 0.017 | — |

Stop threshold (max_f1): 0.406 — precision 0.641, recall 0.655 on the fitting calls; out of fold: precision 0.601, recall 0.658 (1768 calls, 0 folds skipped).

