# Calibration qolda_devall_vacuum_en.json

backend qolda (qolda-4b), verify prompt 5728face305a, runs: runs/matrix/dev_calibration/qolda_en/seed0, runs/matrix/dev_demo/qolda_en/seed0, runs/matrix/dev_demo/qolda_en_fix/seed0, runs/matrix/dev_confirm/qolda_en/seed0, runs/matrix/dev_confirm/qolda_kk/seed0, runs/matrix/dev_confirm/qolda_kkru/seed0 (dev)

1313 calls, 232 show the goal (base rate 0.177); 279 dev episodes; label: content

| | ECE | Brier |
|---|---|---|
| raw P(yes) | 0.146 | 0.152 |
| platt, fitted and judged on the same calls | 0.043 | 0.118 |
| platt, out of fold (5 folds by episode) | 0.045 | — |

Stop threshold (max_f1): 0.217 — precision 0.444, recall 0.647 on the fitting calls; out of fold: precision 0.440, recall 0.599 (1313 calls, 0 folds skipped).

