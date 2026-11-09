# Calibration qolda_devall_vacuum_cmdq.json

backend qolda (qolda-4b), verify prompt 5728face305a, runs: runs/matrix/dev_calibration/qolda_kk/seed0, runs/matrix/dev_calibration/qolda_en/seed0, runs/matrix/dev_calibration/qolda_kkru/seed0, runs/matrix/dev_demo/qolda_kk/seed0, runs/matrix/dev_demo/qolda_kk_fix/seed0, runs/matrix/dev_demo/qolda_en/seed0, runs/matrix/dev_demo/qolda_en_fix/seed0, runs/matrix/dev_demo/qolda_kkru/seed0, runs/matrix/dev_demo/qolda_kkru_fix/seed0, runs/matrix/dev_confirm/qolda_kk_cmdq/seed0 (dev)

2715 calls, 616 show the goal (base rate 0.227); 279 dev episodes; label: content

| | ECE | Brier |
|---|---|---|
| raw P(yes) | 0.071 | 0.147 |
| platt, fitted and judged on the same calls | 0.054 | 0.144 |
| platt, out of fold (5 folds by episode) | 0.051 | — |

Stop threshold (max_f1): 0.330 — precision 0.541, recall 0.570 on the fitting calls; out of fold: precision 0.521, recall 0.549 (2715 calls, 0 folds skipped).

