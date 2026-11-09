# D-088: agent calls, old vs new threshold

| model | artifact | calls | goal crops | goal crops rejected | other crops accepted | threshold |
|---|---|---|---|---|---|---|
| Qolda | `calib/qolda_dev23_en.json` | 607 | 105 | 0.371 | 0.042 | 0.328 |
| Qolda | `calib/qolda_dev_agent_en.json` | 607 | 105 | 0.343 | 0.044 | 0.369 |
| Qwen | `calib/qwen3vl_4b_dev23_en.json` | 561 | 90 | 0.156 | 0.070 | 0.315 |
| Qwen | `calib/qwen3vl_4b_dev_agent_en.json` | 561 | 90 | 0.156 | 0.068 | 0.354 |
