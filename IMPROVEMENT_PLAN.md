# Heart Disease Competition - Improvement Plan

Current LB: **0.95366** (V3)
Target: **+0.0001 to +0.0003** (0.95376+)

## Top 3 Implemented Improvements

1. **Feature Engineering - Interactions** (Expected: +0.00008 to +0.00015)
   - Age×Sex, MaxHR/Age, ST depression×Slope, Thallium×ChestPain
   - Additional clinically meaningful combinations

2. **Bucketed Features** (Expected: +0.00005 to +0.0001)
   - Clinical threshold bins for ST depression, Cholesterol, Age, BP, MaxHR
   - Helps CatBoost capture non-linear patterns

3. **Hyperparameter Tuning** (Expected: +0.00003 to +0.00008)
   - Depth: 5→6 or 7
   - L2 regularization: 3-10
   - Random strength and bagging optimization

## Files Provided

- `kaggle_submission.py`: Single-file solution ready for Kaggle (recommended)
- `heart_disease_improvements.py`: Core feature + CV utilities
- `run_improvements.py`: End-to-end pipeline with search + validation
- `config_switcher.py`: Configuration presets and toggles

## Recommended Execution Plan

### Option A: Full Automated Pipeline (~8-9 hours)
1. Run `python run_improvements.py`
2. Pipeline will:
   - Evaluate feature sets (3-fold)
   - Hyperparameter search (3-fold)
   - Full validation (5-fold)
   - Final training with seed ensemble

### Option B: Direct Kaggle Submission (~6-7 hours)
1. Upload `kaggle_submission.py` to Kaggle
2. Set `STRATEGY = 'balanced'`
3. Run notebook, submit `submission.csv`

### Option C: Conservative (~3-5 hours)
1. Set `STRATEGY = 'quick_test'`
2. Validate quickly locally
3. Expand to `balanced` on Kaggle

## Data Paths

Local expected files (per `data/README.md`):
- `data/raw/train.csv`
- `data/raw/test.csv`
- `data/raw/sample_submission.csv`

Kaggle expected input:
- `/kaggle/input/playground-series-s6e2/{train.csv,test.csv,sample_submission.csv}`
