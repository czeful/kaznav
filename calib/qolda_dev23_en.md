# Calibration qolda_dev23_en.json

backend qolda (qolda-4b), verify prompt 5728face305a, runs: runs\matrix\dev_demo\qolda_en\seed0, runs\matrix\dev_demo\qolda_en_fix\seed0, runs\matrix\dev_confirm\qolda_en\seed0, runs\matrix\dev_confirm\qolda_kk\seed0, runs\matrix\dev_confirm\qolda_kkru\seed0 (dev)

811 calls, 131 show the goal (base rate 0.162); 129 dev episodes; label: content

| | ECE | Brier |
|---|---|---|
| raw P(yes) | 0.158 | 0.155 |
| platt, fitted and judged on the same calls | 0.044 | 0.114 |
| platt, out of fold (5 folds by episode) | 0.049 | — |

Stop threshold (max_f1): 0.328 — precision 0.628, recall 0.450 on the fitting calls; out of fold: precision 0.492, recall 0.489 (811 calls, 0 folds skipped).

