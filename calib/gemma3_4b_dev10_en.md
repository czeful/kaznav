# Calibration gemma3_4b_dev10_en.json

backend gemma3_4b (gemma-3-4b), verify prompt 5728face305a, runs: runs/matrix/dev_gemma_calib/gemma_kk/seed0, runs/matrix/dev_gemma_calib/gemma_en/seed0 (dev)

784 calls, 125 show the goal (base rate 0.159); 150 dev episodes; label: content

| | ECE | Brier |
|---|---|---|
| raw P(yes) | 0.229 | 0.227 |
| platt, fitted and judged on the same calls | 0.030 | 0.103 |
| platt, out of fold (5 folds by episode) | 0.027 | — |

Stop threshold (max_f1): 0.429 — precision 0.580, recall 0.552 on the fitting calls; out of fold: precision 0.516, recall 0.504 (784 calls, 0 folds skipped).

