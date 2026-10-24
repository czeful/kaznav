# Calibration transfer between scene groups: Qolda

A: runs/matrix/dev_calibration/qolda_kk/seed0, runs/matrix/dev_calibration/qolda_en/seed0, runs/matrix/dev_calibration/qolda_kkru/seed0
B: runs/matrix/dev_demo/qolda_kk/seed0, runs/matrix/dev_demo/qolda_kk_fix/seed0, runs/matrix/dev_demo/qolda_en/seed0, runs/matrix/dev_demo/qolda_en_fix/seed0, runs/matrix/dev_demo/qolda_kkru/seed0, runs/matrix/dev_demo/qolda_kkru_fix/seed0

| group | calls | goal crops | base rate | AUROC raw | ECE raw |
|---|---|---|---|---|---|
| A (150 episodes) | 1707 | 351 | 0.206 | 0.795 | 0.071 |
| B (57 episodes) | 782 | 250 | 0.320 | 0.713 | 0.111 |

| rule fitted on | judged on | a | b | threshold | ECE | precision | recall | F1 |
|---|---|---|---|---|---|---|---|---|
| A | A (in-sample) | 0.508 | -0.584 | 0.302 | 0.036 | 0.528 | 0.621 | 0.571 |
| A | B (**transfer**) | 0.508 | -0.584 | 0.302 | 0.121 | 0.656 | 0.472 | 0.549 |
| B | A (**transfer**) | 0.328 | -0.120 | 0.419 | 0.117 | 0.490 | 0.632 | 0.552 |
| B | B (in-sample) | 0.328 | -0.120 | 0.419 | 0.097 | 0.663 | 0.528 | 0.588 |
