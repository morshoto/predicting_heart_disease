---
link: https://www.kaggle.com/competitions/playground-series-s6e2/discussion/671846
---

Hello everyone,

I’m sharing my approach for this competition to exchange ideas and perhaps get some feedback. With the setup below I achieved a 0.95355 ROC AUC on the leaderboard. The solution runs on the cpu and takes around 10 minutes to compute. I followed the recommendation to merge the official Playground training set with the additional Heart Disease dataset. You can find a copy of the notebook [here](https://www.kaggle.com/code/lzr1992/playground-series-february-2026-posted-notebook).

**1️⃣Feature engineering**

I kept feature engineering lightweight and interpretable. All original features were used as inputs. The feature selection was handled implicitly by dropping only the target and ID columns and letting the preprocessing pipeline automatically include all remaining numeric and categorical variables.

The main idea I borrowed from the Magnussen et al. (2025) paper posted in the forums is that cardiovascular risk is driven by the presence and accumulation of major risk factors, and that risk increases nonlinearly with age. Since the Kaggle dataset does not include several variables used in the paper (e.g., smoking, diabetes, BMI, non-HDL cholesterol), I restricted feature engineering to variables that are actually available.

I added Binary “risk factor present” flags for:

- elevated blood pressure (elevated_bp)
- elevated cholesterol (using total cholesterol as a proxy since non-HDL is not available)

Also added:

- Risk factor burden: risk_factor_count = number of present risk-factor flags
- Age-modified risk: risk_age = Age × risk_factor_count to reflect that age modifies the impact of risk burden

This aligns with the paper’s emphasis on risk-factor burden and age-dependent risk accumulation, while staying within the dataset’s limitations.

**2️⃣Preprocessing**

I used a simple preprocessing setup:

- numeric features: median imputation + standard scaling
- categorical features: most-frequent imputation + one-hot encoding

Preprocessing was applied inside cross-validation to avoid leakage.

**3️⃣Model and validation**

I used a single XGBoost model with conservative regularization and subsampling. Performance was estimated using Stratified 5-Fold CV and I trained one model per fold, storing out-of-fold predictions and averaging test predictions across folds.

**4️⃣Things that did not work for me**

I experimented with several models, including CatBoost and LightGBM, but none improved on the XGBoost baseline. I also tested various ensembles combining two or three of these different types of models without any performance gains. I assume this is because the models were highly correlated with XGBoost and therefore did not add enough diversity to improve the results.

I would be interested to know what people think of this approach and where you see room for improvement. Happy to hear your feedback and thank you for reading.

Hello everyone,

I’m sharing my approach for this competition to exchange ideas and perhaps get some feedback. With the setup below I achieved a 0.95355 ROC AUC on the leaderboard. The solution runs on the cpu and takes around 10 minutes to compute. I followed the recommendation to merge the official Playground training set with the additional Heart Disease dataset. You can find a copy of the notebook [here](https://www.kaggle.com/code/lzr1992/playground-series-february-2026-posted-notebook).

**1️⃣Feature engineering**

I kept feature engineering lightweight and interpretable. All original features were used as inputs. The feature selection was handled implicitly by dropping only the target and ID columns and letting the preprocessing pipeline automatically include all remaining numeric and categorical variables.

The main idea I borrowed from the Magnussen et al. (2025) paper posted in the forums is that cardiovascular risk is driven by the presence and accumulation of major risk factors, and that risk increases nonlinearly with age. Since the Kaggle dataset does not include several variables used in the paper (e.g., smoking, diabetes, BMI, non-HDL cholesterol), I restricted feature engineering to variables that are actually available.

I added Binary “risk factor present” flags for:

- elevated blood pressure (elevated_bp)
- elevated cholesterol (using total cholesterol as a proxy since non-HDL is not available)

Also added:

- Risk factor burden: risk_factor_count = number of present risk-factor flags
- Age-modified risk: risk_age = Age × risk_factor_count to reflect that age modifies the impact of risk burden

This aligns with the paper’s emphasis on risk-factor burden and age-dependent risk accumulation, while staying within the dataset’s limitations.

**2️⃣Preprocessing**

I used a simple preprocessing setup:

- numeric features: median imputation + standard scaling
- categorical features: most-frequent imputation + one-hot encoding

Preprocessing was applied inside cross-validation to avoid leakage.

**3️⃣Model and validation**

I used a single XGBoost model with conservative regularization and subsampling. Performance was estimated using Stratified 5-Fold CV and I trained one model per fold, storing out-of-fold predictions and averaging test predictions across folds.

**4️⃣Things that did not work for me**

I experimented with several models, including CatBoost and LightGBM, but none improved on the XGBoost baseline. I also tested various ensembles combining two or three of these different types of models without any performance gains. I assume this is because the models were highly correlated with XGBoost and therefore did not add enough diversity to improve the results.

I would be interested to know what people think of this approach and where you see room for improvement. Happy to hear your feedback and thank you for reading.
