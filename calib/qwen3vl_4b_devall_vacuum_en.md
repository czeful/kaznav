# Calibration qwen3vl_4b_devall_vacuum_en.json

backend qwen3vl_4b (qwen3-vl-4b), verify prompt 5728face305a, runs: runs/matrix/dev_calibration/qwen_en/seed0, runs/matrix/dev_demo/qwen_en/seed0, runs/matrix/dev_demo/mt_qwen_kk/seed0, runs/matrix/dev_demo/mt_qwen_kkru/seed0, runs/matrix/dev_confirm/qwen_en/seed0, runs/matrix/dev_confirm/qwen_kk/seed0, runs/matrix/dev_confirm/qwen_kkru/seed0, runs/matrix/dev_confirm/mt_qwen_kk/seed0, runs/matrix/dev_confirm/mt_qwen_kkru/seed0 (dev)

1934 calls, 263 show the goal (base rate 0.136); 279 dev episodes; label: content

| | ECE | Brier |
|---|---|---|
| raw P(yes) | 0.144 | 0.141 |
| platt, fitted and judged on the same calls | 0.017 | 0.079 |
| platt, out of fold (5 folds by episode) | 0.014 | — |

Stop threshold (max_f1): 0.272 — precision 0.541, recall 0.658 on the fitting calls; out of fold: precision 0.548, recall 0.612 (1934 calls, 0 folds skipped).

