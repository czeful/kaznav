# Calibration qwen3vl_4b_dev.json

backend qwen3vl_4b (qwen3-vl-4b), verify prompt 5728face305a, runs: runs/matrix/dev_calibration/qwen_kk/seed0, runs/matrix/dev_calibration/qwen_en/seed0, runs/matrix/dev_calibration/qwen_kkru/seed0 (dev)

1142 calls, 176 show the goal (base rate 0.154); 150 dev episodes; label: content

| | ECE | Brier |
|---|---|---|
| raw P(yes) | 0.137 | 0.139 |
| platt, fitted and judged on the same calls | 0.014 | 0.084 |
| platt, out of fold (5 folds by episode) | 0.017 | — |

Stop threshold (max_f1): 0.411 — precision 0.622, recall 0.636 on the fitting calls; out of fold: precision 0.546, recall 0.636 (1142 calls, 0 folds skipped).

