# Calibration qolda_dev18.json

backend qolda (qolda-4b), verify prompt 5728face305a, runs: runs\matrix\dev_calibration\qolda_kk\seed0, runs\matrix\dev_calibration\qolda_en\seed0, runs\matrix\dev_calibration\qolda_kkru\seed0, runs\matrix\dev_demo\qolda_kk\seed0, runs\matrix\dev_demo\qolda_kk_fix\seed0, runs\matrix\dev_demo\qolda_en\seed0, runs\matrix\dev_demo\qolda_en_fix\seed0, runs\matrix\dev_demo\qolda_kkru\seed0, runs\matrix\dev_demo\qolda_kkru_fix\seed0 (dev)

2489 calls, 601 show the goal (base rate 0.241); 207 dev episodes; label: content

| | ECE | Brier |
|---|---|---|
| raw P(yes) | 0.070 | 0.151 |
| platt, fitted and judged on the same calls | 0.063 | 0.150 |
| platt, out of fold (5 folds by episode) | 0.060 | — |

Stop threshold (max_f1): 0.348 — precision 0.566, recall 0.566 on the fitting calls; out of fold: precision 0.542, recall 0.551 (2489 calls, 0 folds skipped).

