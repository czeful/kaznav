# Calibration qolda_dev18_en.json

backend qolda (qolda-4b), verify prompt 5728face305a, runs: runs\matrix\dev_calibration\qolda_en\seed0, runs\matrix\dev_demo\qolda_en\seed0, runs\matrix\dev_demo\qolda_en_fix\seed0 (dev)

735 calls, 167 show the goal (base rate 0.227); 207 dev episodes; label: content

| | ECE | Brier |
|---|---|---|
| raw P(yes) | 0.108 | 0.149 |
| platt, fitted and judged on the same calls | 0.050 | 0.136 |
| platt, out of fold (5 folds by episode) | 0.049 | — |

Stop threshold (max_f1): 0.285 — precision 0.586, recall 0.635 on the fitting calls; out of fold: precision 0.581, recall 0.623 (735 calls, 0 folds skipped).

