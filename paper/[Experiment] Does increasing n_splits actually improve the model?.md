Hi everyone,

While experimenting and looking at public codes, I noticed a trend: **increasing the number of CV splits often leads to a slightly higher CV score.**

Generally, when using **early stopping**, we indirectly use the validation data for training. As `n_splits` increases (and the validation set size per fold decreases), the risk of overfitting to the validation set naturally increases.

While there is no problem in optimizing within the same split strategy, I wondered: **"Can we conclude that the model accuracy has improved just because the CV score went up with more splits?"**

To verify this, I conducted a simple experiment using a Pseudo LB (Hold-out) approach.

## Experiment Setup

- **Data Split:**
    - Total Train Data → Split into **Experiment Train (80%)** and **Pseudo LB (20%)**.
    - The Pseudo LB acts as the "Private Test" set and is never used for training or early stopping.
- **Model:** XGBClassifier (Fixed parameters for all runs).
- **Validation:** StratifiedKFold with `n_splits` = [3, 5, 7, 10, 15, 20].
- **Metrics:**
    - **CV (OOF):** Overall AUC of Out-of-Fold predictions on Experiment Train.
    - **Pseudo LB:** AUC of the ensemble (average) predictions of all folds on the Pseudo LB set.

## Results

Here is the comparison between CV Score and Pseudo LB Score:

| n_splits | CV_OOF   | Pseudo_LB | Gap (CV - LB) |
| :------- | :------- | :-------- | :------------ |
| **3**    | 0.955068 | 0.956196  | -0.001128     |
| **5**    | 0.955147 | 0.956201  | -0.001054     |
| **7**    | 0.955156 | 0.956208  | -0.001052     |
| **10**   | 0.955174 | 0.956213  | -0.001039     |
| **15**   | 0.955193 | 0.956211  | -0.001018     |
| **20**   | 0.955218 | 0.956211  | -0.000992     |

## Conclusion

- **CV Score (OOF):** Consistent improvement as splits increased (0.9550 → 0.9552).
- **Pseudo LB:** Slight improvement (0.95619 → 0.95621).

Comparing `n_splits=3` vs `n_splits=20`:

- The CV score improved by **+0.00015**.
- The Pseudo LB improved by only **+0.000015**.

While the Pseudo LB did improve (likely due to the larger ensemble size), the CV score improved **10 times more**.
This suggests that **higher CV scores from high split numbers (e.g., 10, 20) can be "optimistic"**—at least in this experimental setup—and do not necessarily guarantee a proportional gain on the real Leaderboard.

We should be careful when comparing CV scores across different split strategies!

**Experiment Notebook:**
[Check the code here](https://www.kaggle.com/code/masayakawamata/s6e2-cv-split-experiment?scriptVersionId=295994142)

Happy Kaggling!
