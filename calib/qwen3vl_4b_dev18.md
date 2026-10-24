# Calibration qwen3vl_4b_dev18.json

backend qwen3vl_4b (qwen3-vl-4b), verify prompt 5728face305a, runs: runs\matrix\dev_calibration\qwen_kk\seed0, runs\matrix\dev_calibration\qwen_en\seed0, runs\matrix\dev_calibration\qwen_kkru\seed0, runs\matrix\dev_demo\qwen_kk\seed0, runs\matrix\dev_demo\qwen_en\seed0, runs\matrix\dev_demo\qwen_kkru\seed0 (dev)

1510 calls, 268 show the goal (base rate 0.177); 207 dev episodes; label: content

| | ECE | Brier |
|---|---|---|
| raw P(yes) | 0.138 | 0.139 |
| platt, fitted and judged on the same calls | 0.016 | 0.089 |
| platt, out of fold (5 folds by episode) | 0.018 | — |

Stop threshold (max_f1): 0.460 — precision 0.692, recall 0.638 on the fitting calls; out of fold: precision 0.644, recall 0.642 (1510 calls, 0 folds skipped).

