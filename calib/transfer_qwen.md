# Calibration transfer between scene groups: Qwen3-VL-4B

A: runs/matrix/dev_calibration/qwen_kk/seed0, runs/matrix/dev_calibration/qwen_en/seed0, runs/matrix/dev_calibration/qwen_kkru/seed0
B: runs/matrix/dev_demo/qwen_kk/seed0, runs/matrix/dev_demo/qwen_en/seed0, runs/matrix/dev_demo/qwen_kkru/seed0

| group | calls | goal crops | base rate | AUROC raw | ECE raw |
|---|---|---|---|---|---|
| A (150 episodes) | 1142 | 176 | 0.154 | 0.880 | 0.137 |
| B (57 episodes) | 368 | 92 | 0.250 | 0.862 | 0.145 |

| rule fitted on | judged on | a | b | threshold | ECE | precision | recall | F1 |
|---|---|---|---|---|---|---|---|---|
| A | A (in-sample) | 0.162 | -1.415 | 0.411 | 0.014 | 0.622 | 0.636 | 0.629 |
| A | B (**transfer**) | 0.162 | -1.415 | 0.411 | 0.088 | 0.881 | 0.641 | 0.742 |
| B | A (**transfer**) | 0.166 | -0.735 | 0.462 | 0.063 | 0.548 | 0.676 | 0.606 |
| B | B (in-sample) | 0.166 | -0.735 | 0.462 | 0.068 | 0.776 | 0.717 | 0.746 |
