# Calibration qwen3vl_4b_dev23_en.json

backend qwen3vl_4b (qwen3-vl-4b), verify prompt 5728face305a, runs: runs\matrix\dev_demo\qwen_en\seed0, runs\matrix\dev_demo\mt_qwen_kk\seed0, runs\matrix\dev_demo\mt_qwen_kkru\seed0, runs\matrix\dev_confirm\qwen_en\seed0, runs\matrix\dev_confirm\qwen_kk\seed0, runs\matrix\dev_confirm\qwen_kkru\seed0, runs\matrix\dev_confirm\mt_qwen_kk\seed0, runs\matrix\dev_confirm\mt_qwen_kkru\seed0 (dev)

1529 calls, 201 show the goal (base rate 0.131); 129 dev episodes; label: content

| | ECE | Brier |
|---|---|---|
| raw P(yes) | 0.150 | 0.147 |
| platt, fitted and judged on the same calls | 0.022 | 0.080 |
| platt, out of fold (5 folds by episode) | 0.021 | — |

Stop threshold (max_f1): 0.315 — precision 0.565, recall 0.582 on the fitting calls; out of fold: precision 0.534, recall 0.587 (1529 calls, 0 folds skipped).

