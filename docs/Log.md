#### 20xx/mm/dd

- join!

**Submission**

- PL:
- Ranking:

#### 2026/02/01

- join!

**nb/000_eda.ipynb**:

- Age: Presence group is higher (mean 56 vs 52)
- Max HR: Presence group is lower (143 vs 160)
- ST depression: Presence group is significantly higher (1.17 vs 0.35) + higher zero rate
- Number of vessels fluro: Presence group is larger (0.84 vs 0.14)
- Thallium: Presence group larger (5.93 vs 3.55)
- BP shows almost no difference (based on current data)

- Performance: ROC-AUC ≈ 0.953
- Stability: std ≈ 0.0004

**Submission**

- PL:
- Ranking:

#### 2026/02/02

- Working on catboost

**nb/004_catboost.ipynb**

- CV AUC: mean 0.955501
    - std 0.000448
- best_iteration: [1183, 972, 1566, 1187, 1424]
    - median = 1187
    - mean = 1266.4

👉 final_iterations = int(1187 \* 1.15) = 1365 (rounded down) = 1400

| fold | roc_auc  | best_iteration |
| ---- | -------- | -------------- |
| 1    | 0.955856 | 1183.0         |
| 2    | 0.954810 | 972.0          |
| 3    | 0.955629 | 1566.0         |
| 4    | 0.955181 | 1187.0         |
| 5    | 0.956030 | 1424.0         |
| mean | 0.955501 | 1266.4         |
| std  | 0.000448 | NaN            |

**006_catboost_features**

- Incorporate sweep's "winning parameters" into production:
    - depth (4/5/6)
    - l2_leaf_reg (around 3/5/10)
    - rsm (0.7-0.9)
    - bootstrap_type + (subsample or bagging_temperature)
    - random_strength
- Highly robust "multi-seed average" ensemble: submit by averaging the test_pred of 5 models with random_seed = 42, 202, 777, 1337, 31415.

**007_catboost_features**

- Early stopping is not performed on the Submit side (a huge waste)
- Seed ensemble:
    - Vary 5-10 random seeds
    - For each seed, run 5-fold cross-validation and average the test predictions
    - Finally, average the seeds
- Mix "fast sweep top performers" in the model difference ensemble

**Submission**

- PL:
- Ranking:
