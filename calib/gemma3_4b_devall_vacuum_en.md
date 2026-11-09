# Calibration gemma3_4b_devall_vacuum_en.json

backend gemma3_4b (gemma-3-4b), verify prompt 5728face305a, runs: runs/matrix/dev_gemma_calib/gemma_kk/seed0, runs/matrix/dev_gemma_calib/gemma_en/seed0, runs/matrix/dev_gemma_confirm/gemma_kk/seed0, runs/matrix/dev_gemma_confirm/gemma_en/seed0 (dev)

1281 calls, 165 show the goal (base rate 0.129); 222 dev episodes; label: content

| | ECE | Brier |
|---|---|---|
| raw P(yes) | 0.242 | 0.239 |
| platt, fitted and judged on the same calls | 0.040 | 0.090 |
| platt, out of fold (5 folds by episode) | 0.017 | — |

Stop threshold (max_f1): 0.409 — precision 0.571, recall 0.485 on the fitting calls; out of fold: precision 0.494, recall 0.509 (1281 calls, 0 folds skipped).

