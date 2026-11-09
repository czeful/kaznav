# Calibration qolda_dev_agent_en.json

backend qolda (qolda-4b), verify prompt 5728face305a, runs: runs/matrix/dev_agent_final/qolda_kk/seed0 (dev)

607 calls, 105 show the goal (base rate 0.173); 150 dev episodes; label: content

| | ECE | Brier |
|---|---|---|
| raw P(yes) | 0.113 | 0.105 |
| platt, fitted and judged on the same calls | 0.030 | 0.079 |
| platt, out of fold (5 folds by episode) | 0.038 | — |

Stop threshold (max_f1): 0.369 — precision 0.758, recall 0.657 on the fitting calls; out of fold: precision 0.673, recall 0.648 (607 calls, 0 folds skipped).

