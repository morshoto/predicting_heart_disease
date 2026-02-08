#### 20xx/mm/dd

- join!

**Submission**

- PL:
- Ranking:

---

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

---

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

👉 This took above 9 hours of execution due to GPU usage failure

**Submission**

- PL:
- Ranking:

---

#### 2026/02/03

- UTC time is 9:00 on GMT+9

**Submission**

- PL:
- Ranking:

---

#### 2026/02/04

**007_catboost_features**

- 5-fold base CV: mean AUC ≈ 0.95550
- Final Ensemble OOF: 0.955425
- Kaggle Public: 0.95351

This difference (approx. 0.0019) is commonly observed in this type of small-scale tabular data. There are two main reasons:

OOF optimization leans too heavily on the CV split (strong 'luck of the split').
By fixing StratifiedKFold to random_state=42, settings that "perform well" on that specific fold are chosen.

Model differences are almost identical (highly correlated), so ensembling doesn't improve the Public score.
This is exemplified by the nearly identical fold AUCs for 'base' and 'rand_strength2'.

**Submission**

- PL:
- Ranking:

---

#### 2026/02/07

**009_nb**:

- **Age_X_Cholesterol Feature (-0.00010 to -0.00015)**

    ```python
    train["Age_X_Cholesterol"] = (train["Age"] * train["Cholesterol"]).astype(float)
    ```

    - **Problem:** This creates a NUMERIC feature (values 7,000-20,000+)
    - **Not added** to `cat_cols_model` → CatBoost treats as continuous
    - **Result:** Just noise, no clear pattern, hurts performance

- **Aggressive Hyperparameters (-0.00005 to -0.00010)**
  You changed 5 parameters at once:
    - `bootstrap_type="Bayesian"` + `bagging_temperature=1.0` → Too aggressive for 270k dataset
    - `random_strength=1.0` → May not help
    - `od_wait=120` (was 150) → Stopping too early
    - `l2_leaf_reg=5` → Too much regularization

**010_remove_age_cholesterol**:

- PL:0.95342

- **TOO MANY FEATURES (27 total!)**
    - You added 12 new features at once
    - Many are redundant: Age_bucket + Age_X_Sex, 3× MaxHR features, etc.
    - Model is **memorizing training patterns** that don't generalize

- **WRONG HYPERPARAMETERS**
    - Your sweep found **depth=4** works best (0.956203)
    - But you used **depth=6** (not even in top 10!)
    - Used **l2_leaf_reg=3** (sweep says 5-10 is better)
    - Result: Too deep + too little regularization = overfitting

- **Complexity Paradox**
    - V3: Simple (15 features) → LB 0.95366 ✓
    - V5: Complex (27 features) → LB 0.95342 ✗
    - **More features made it WORSE!**

**Submission**

- PL:
- Ranking:

---
