# Predicting Heart Disease

Kaggle Playground Series S6E2 — Heart Disease prediction experiments and baselines.

<img width="560" height="280" alt="image" src="https://github.com/user-attachments/assets/3571bea8-f527-4f47-a832-8d58e3bfd1f2" />

## Overview

- Predict heart disease risk for the Kaggle Playground Series S6E2 competition.
- Approach: tabular ML models with feature engineering and ensembling experiments.
- Data sources are competition data plus optional external datasets referenced in notebooks.
- Metric: ROC AUC Kaggle leaderboard metric
- Ranked top 2% (105/4,507 Participants)



## Results

| Model                        | CV AUC   | LB AUC  | Rank | Notes |
| ---------------------------- | -------- | ------- | ---- | ----- |
| Public | 0.952922 | 0.95406 | 210  | - |
| Private | 0.955446 | 0.95530 | 105 | nb/051_ensemble_v2_wall_breaker.ipynb |

## Project Structure

```bash
├── data           <---- Local data workspace
├── docs           <---- Documentation and logs
│   ├── Log.md
│   ├── Paper.md
│   └── Score.md
├── paper          <---- Reference notes and ideas
├── nb             <---- Author notebooks
├── nb_download    <---- Public Kaggle notebooks (mirrors)
├── cli            <---- Utility scripts
├── src            <---- Reusable code
│   ├── cli        <---- CLI entry points
│   ├── core       <---- Shared config and utilities
│   ├── data       <---- Data I/O and preprocessing
│   ├── features   <---- Feature engineering
│   ├── models     <---- Model implementations
│   └── pipelines  <---- Orchestration flows
```

### Dataset

The dataset provided for this competition consists of.

| Name                  | Detail                                                                                                                     | Size     | Link                                                                    |
| --------------------- | -------------------------------------------------------------------------------------------------------------------------- | -------- | ----------------------------------------------------------------------- |
| playground-series-s6e | The dataset for this competition was generated from a deep learning model trained on the Heart disease prediction dataset. | 45.43 MB | [Link](https://www.kaggle.com/competitions/playground-series-s6e2/data) |

## Docs and Notebooks

- Setup guide: `docs/Setup.md`
- Experiment log: `docs/Log.md`
- Score tracking: `docs/Score.md`
- Paper notes: `docs/Paper.md`
- Notebooks: `nb/` (originals), `nb_download/` (public Kaggle notebooks)
