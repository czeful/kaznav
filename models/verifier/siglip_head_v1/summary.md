# Light verifier: siglip_head_v1

data runs/verifier_data/v1: 16065 crops, 40 scenes (8 held out); evaluation: 10225 verifier crops of the finished runs

| crops                                        |     n |   goal | score   |   auroc |   tpr@fpr0.1 |   fpr@tpr0.7 |
|:---------------------------------------------|------:|-------:|:--------|--------:|-------------:|-------------:|
| held-out data scenes                         |  3173 |   1378 | head    |   0.902 |        0.687 |        0.104 |
| held-out data scenes                         |  3173 |   1378 | sig     |   0.837 |        0.547 |        0.169 |
| held-out data scenes                         |  3173 |   1378 | soft    |   0.851 |        0.565 |        0.158 |
| all verifier crops                           | 10225 |   1899 | VLM     |   0.809 |        0.498 |        0.203 |
| all verifier crops                           | 10225 |   1899 | head    |   0.834 |        0.533 |        0.198 |
| all verifier crops                           | 10225 |   1899 | sig     |   0.864 |        0.631 |        0.133 |
| all verifier crops                           | 10225 |   1899 | soft    |   0.881 |        0.684 |        0.106 |
| qolda en wheeled_vacuum_kinematic dev10      |  2123 |    406 | VLM     |   0.791 |        0.51  |        0.236 |
| qolda en wheeled_vacuum_kinematic dev10      |  2123 |    406 | head    |   0.84  |        0.564 |        0.207 |
| qolda en wheeled_vacuum_kinematic dev10      |  2123 |    406 | sig     |   0.881 |        0.67  |        0.11  |
| qolda en wheeled_vacuum_kinematic dev10      |  2123 |    406 | soft    |   0.905 |        0.722 |        0.093 |
| qolda en wheeled_vacuum_kinematic dev2       |   233 |     66 | VLM     |   0.763 |        0.636 |        0.251 |
| qolda en wheeled_vacuum_kinematic dev2       |   233 |     66 | head    |   0.789 |        0.485 |        0.257 |
| qolda en wheeled_vacuum_kinematic dev2       |   233 |     66 | sig     |   0.856 |        0.636 |        0.162 |
| qolda en wheeled_vacuum_kinematic dev2       |   233 |     66 | soft    |   0.858 |        0.667 |        0.12  |
| qolda en wheeled_vacuum_kinematic dev3       |   574 |     65 | VLM     |   0.803 |        0.538 |        0.234 |
| qolda en wheeled_vacuum_kinematic dev3       |   574 |     65 | head    |   0.855 |        0.6   |        0.145 |
| qolda en wheeled_vacuum_kinematic dev3       |   574 |     65 | sig     |   0.929 |        0.723 |        0.059 |
| qolda en wheeled_vacuum_kinematic dev3       |   574 |     65 | soft    |   0.937 |        0.769 |        0.043 |
| qolda kk wheeled_vacuum_kinematic dev10      |  2619 |    528 | VLM     |   0.801 |        0.504 |        0.225 |
| qolda kk wheeled_vacuum_kinematic dev10      |  2619 |    528 | head    |   0.825 |        0.492 |        0.215 |
| qolda kk wheeled_vacuum_kinematic dev10      |  2619 |    528 | sig     |   0.869 |        0.616 |        0.124 |
| qolda kk wheeled_vacuum_kinematic dev10      |  2619 |    528 | soft    |   0.89  |        0.686 |        0.108 |
| qolda kk wheeled_vacuum_kinematic dev2       |   549 |    184 | VLM     |   0.713 |        0.424 |        0.37  |
| qolda kk wheeled_vacuum_kinematic dev2       |   549 |    184 | head    |   0.83  |        0.451 |        0.195 |
| qolda kk wheeled_vacuum_kinematic dev2       |   549 |    184 | sig     |   0.832 |        0.576 |        0.2   |
| qolda kk wheeled_vacuum_kinematic dev2       |   549 |    184 | soft    |   0.834 |        0.598 |        0.225 |
| qolda kk wheeled_vacuum_kinematic dev3       |   444 |     45 | VLM     |   0.808 |        0.489 |        0.173 |
| qolda kk wheeled_vacuum_kinematic dev3       |   444 |     45 | head    |   0.87  |        0.644 |        0.128 |
| qolda kk wheeled_vacuum_kinematic dev3       |   444 |     45 | sig     |   0.893 |        0.622 |        0.105 |
| qolda kk wheeled_vacuum_kinematic dev3       |   444 |     45 | soft    |   0.858 |        0.689 |        0.1   |
| qwen3vl_4b en habitat_agent dev10            |   695 |    143 | VLM     |   0.943 |        0.867 |        0.053 |
| qwen3vl_4b en habitat_agent dev10            |   695 |    143 | head    |   0.874 |        0.657 |        0.129 |
| qwen3vl_4b en habitat_agent dev10            |   695 |    143 | sig     |   0.929 |        0.727 |        0.092 |
| qwen3vl_4b en habitat_agent dev10            |   695 |    143 | soft    |   0.936 |        0.79  |        0.058 |
| qwen3vl_4b en wheeled_vacuum_kinematic dev10 |   400 |     62 | VLM     |   0.931 |        0.806 |        0.089 |
| qwen3vl_4b en wheeled_vacuum_kinematic dev10 |   400 |     62 | head    |   0.854 |        0.581 |        0.172 |
| qwen3vl_4b en wheeled_vacuum_kinematic dev10 |   400 |     62 | sig     |   0.889 |        0.597 |        0.13  |
| qwen3vl_4b en wheeled_vacuum_kinematic dev10 |   400 |     62 | soft    |   0.91  |        0.694 |        0.104 |
| qwen3vl_4b en wheeled_vacuum_kinematic dev2  |   375 |    112 | VLM     |   0.87  |        0.643 |        0.137 |
| qwen3vl_4b en wheeled_vacuum_kinematic dev2  |   375 |    112 | head    |   0.811 |        0.58  |        0.316 |
| qwen3vl_4b en wheeled_vacuum_kinematic dev2  |   375 |    112 | sig     |   0.81  |        0.616 |        0.163 |
| qwen3vl_4b en wheeled_vacuum_kinematic dev2  |   375 |    112 | soft    |   0.83  |        0.688 |        0.106 |
| qwen3vl_4b en wheeled_vacuum_kinematic dev3  |  1292 |    112 | VLM     |   0.84  |        0.625 |        0.14  |
| qwen3vl_4b en wheeled_vacuum_kinematic dev3  |  1292 |    112 | head    |   0.868 |        0.607 |        0.145 |
| qwen3vl_4b en wheeled_vacuum_kinematic dev3  |  1292 |    112 | sig     |   0.856 |        0.625 |        0.114 |
| qwen3vl_4b en wheeled_vacuum_kinematic dev3  |  1292 |    112 | soft    |   0.863 |        0.688 |        0.108 |
| qwen3vl_4b kk wheeled_vacuum_kinematic dev10 |   521 |    108 | VLM     |   0.855 |        0.63  |        0.16  |
| qwen3vl_4b kk wheeled_vacuum_kinematic dev10 |   521 |    108 | head    |   0.88  |        0.62  |        0.136 |
| qwen3vl_4b kk wheeled_vacuum_kinematic dev10 |   521 |    108 | sig     |   0.881 |        0.593 |        0.148 |
| qwen3vl_4b kk wheeled_vacuum_kinematic dev10 |   521 |    108 | soft    |   0.905 |        0.713 |        0.09  |
| qwen3vl_4b kk wheeled_vacuum_kinematic dev2  |   197 |     55 | VLM     |   0.819 |        0.691 |        0.155 |
| qwen3vl_4b kk wheeled_vacuum_kinematic dev2  |   197 |     55 | head    |   0.791 |        0.545 |        0.289 |
| qwen3vl_4b kk wheeled_vacuum_kinematic dev2  |   197 |     55 | sig     |   0.827 |        0.582 |        0.148 |
| qwen3vl_4b kk wheeled_vacuum_kinematic dev2  |   197 |     55 | soft    |   0.811 |        0.545 |        0.183 |
| qwen3vl_4b kk wheeled_vacuum_kinematic dev3  |   203 |     13 | VLM     |   0.982 |        1     |        0.021 |
| qwen3vl_4b kk wheeled_vacuum_kinematic dev3  |   203 |     13 | head    |   0.95  |        0.769 |        0.053 |
| qwen3vl_4b kk wheeled_vacuum_kinematic dev3  |   203 |     13 | sig     |   0.976 |        1     |        0.037 |
| qwen3vl_4b kk wheeled_vacuum_kinematic dev3  |   203 |     13 | soft    |   0.977 |        1     |        0.037 |

D-075 rule: head AUROC on the verifier crops 0.834, zero-shot soft 0.881 -> **zero-shot (sig)**
