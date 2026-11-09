# Calibration transfer between scene groups: Qwen (D-088)

A: runs/matrix/dev_calibration/qwen_en/seed0
B: runs/matrix/dev_demo/qwen_en/seed0, runs/matrix/dev_demo/mt_qwen_kk/seed0, runs/matrix/dev_demo/mt_qwen_kkru/seed0, runs/matrix/dev_confirm/qwen_en/seed0, runs/matrix/dev_confirm/qwen_kk/seed0, runs/matrix/dev_confirm/qwen_kkru/seed0, runs/matrix/dev_confirm/mt_qwen_kk/seed0, runs/matrix/dev_confirm/mt_qwen_kkru/seed0

| group | calls | goal crops | base rate | AUROC raw | ECE raw |
|---|---|---|---|---|---|
| A (150 episodes) | 405 | 62 | 0.153 | 0.932 | 0.122 |
| B (129 episodes) | 1529 | 201 | 0.131 | 0.843 | 0.150 |

| rule fitted on | judged on | a | b | threshold | ECE | precision | recall | F1 |
|---|---|---|---|---|---|---|---|---|
| A | A (in-sample) | 0.233 | -1.510 | 0.272 | 0.034 | 0.595 | 0.806 | 0.685 |
| A | B (**transfer**) | 0.233 | -1.510 | 0.272 | 0.043 | 0.467 | 0.642 | 0.541 |
| B | A (**transfer**) | 0.155 | -1.511 | 0.315 | 0.051 | 0.641 | 0.661 | 0.651 |
| B | B (in-sample) | 0.155 | -1.511 | 0.315 | 0.022 | 0.565 | 0.582 | 0.574 |
