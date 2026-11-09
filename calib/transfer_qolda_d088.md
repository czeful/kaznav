# Calibration transfer between scene groups: Qolda (D-088)

A: runs/matrix/dev_calibration/qolda_en/seed0
B: runs/matrix/dev_demo/qolda_en/seed0, runs/matrix/dev_demo/qolda_en_fix/seed0, runs/matrix/dev_confirm/qolda_en/seed0, runs/matrix/dev_confirm/qolda_kk/seed0, runs/matrix/dev_confirm/qolda_kkru/seed0

| group | calls | goal crops | base rate | AUROC raw | ECE raw |
|---|---|---|---|---|---|
| A (150 episodes) | 502 | 101 | 0.201 | 0.795 | 0.126 |
| B (129 episodes) | 811 | 131 | 0.162 | 0.773 | 0.158 |

| rule fitted on | judged on | a | b | threshold | ECE | precision | recall | F1 |
|---|---|---|---|---|---|---|---|---|
| A | A (in-sample) | 0.604 | -1.034 | 0.221 | 0.039 | 0.479 | 0.693 | 0.567 |
| A | B (**transfer**) | 0.604 | -1.034 | 0.221 | 0.063 | 0.372 | 0.679 | 0.481 |
| B | A (**transfer**) | 0.492 | -1.331 | 0.328 | 0.039 | 0.655 | 0.376 | 0.478 |
| B | B (in-sample) | 0.492 | -1.331 | 0.328 | 0.044 | 0.628 | 0.450 | 0.524 |
