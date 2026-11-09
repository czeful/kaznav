# Calibration qwen3vl_4b_dev_agent_en.json

backend qwen3vl_4b (qwen3-vl-4b), verify prompt 5728face305a, runs: runs/matrix/dev_agent_final/qwen_en/seed0 (dev)

561 calls, 90 show the goal (base rate 0.160); 150 dev episodes; label: content

| | ECE | Brier |
|---|---|---|
| raw P(yes) | 0.112 | 0.107 |
| platt, fitted and judged on the same calls | 0.020 | 0.059 |
| platt, out of fold (5 folds by episode) | 0.025 | — |

Stop threshold (max_f1): 0.354 — precision 0.704, recall 0.844 on the fitting calls; out of fold: precision 0.710, recall 0.789 (561 calls, 0 folds skipped).

