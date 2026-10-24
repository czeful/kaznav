# Calibration qolda_dev.json

backend qolda (qolda-4b), verify prompt 5728face305a, runs: runs/matrix/dev_calibration/qolda_kk/seed0, runs/matrix/dev_calibration/qolda_en/seed0, runs/matrix/dev_calibration/qolda_kkru/seed0 (dev)

1707 calls, 351 show the goal (base rate 0.206); 150 dev episodes; label: content

| | ECE | Brier |
|---|---|---|
| raw P(yes) | 0.071 | 0.134 |
| platt, fitted and judged on the same calls | 0.036 | 0.129 |
| platt, out of fold (5 folds by episode) | 0.033 | — |

Stop threshold (max_f1): 0.302 — precision 0.528, recall 0.621 on the fitting calls; out of fold: precision 0.521, recall 0.601 (1707 calls, 0 folds skipped).

