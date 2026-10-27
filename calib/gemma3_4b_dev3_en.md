# Calibration gemma3_4b_dev3_en.json

backend gemma3_4b (gemma-3-4b), verify prompt 5728face305a, runs: runs/matrix/dev_gemma_confirm/gemma_kk/seed0, runs/matrix/dev_gemma_confirm/gemma_en/seed0 (dev)

497 calls, 40 show the goal (base rate 0.080); 72 dev episodes; label: content

| | ECE | Brier |
|---|---|---|
| raw P(yes) | 0.263 | 0.259 |
| platt, fitted and judged on the same calls | 0.002 | 0.061 |
| platt, out of fold (5 folds by episode) | 0.007 | — |

Stop threshold (max_f1): 0.285 — precision 0.403, recall 0.625 on the fitting calls; out of fold: precision 0.400, recall 0.600 (497 calls, 0 folds skipped).

