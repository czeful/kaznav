# Calibration qwen3vl_4b_dev18_en.json

backend qwen3vl_4b (qwen3-vl-4b), verify prompt 5728face305a, runs: runs\matrix\dev_calibration\qwen_en\seed0, runs\matrix\dev_demo\qwen_en\seed0, runs\matrix\dev_demo\mt_qwen_kk\seed0, runs\matrix\dev_demo\mt_qwen_kkru\seed0 (dev)

804 calls, 179 show the goal (base rate 0.223); 207 dev episodes; label: content

| | ECE | Brier |
|---|---|---|
| raw P(yes) | 0.143 | 0.145 |
| platt, fitted and judged on the same calls | 0.031 | 0.102 |
| platt, out of fold (5 folds by episode) | 0.026 | — |

Stop threshold (max_f1): 0.293 — precision 0.609, recall 0.782 on the fitting calls; out of fold: precision 0.629, recall 0.721 (804 calls, 0 folds skipped).

