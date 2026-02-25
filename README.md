# Premier League Match Predictor

## Overview
Python script that downloads the last three Premier League seasons, engineers simple form/strength features, trains four classifiers (RandomForest, Logistic Regression, KNN, XGBoost) on a shared train/test split, compares their accuracies, and outputs per-model win/draw/lose probabilities for a chosen fixture with bar and pie charts.

## Prerequisites
- Python 3.9+ recommended
- Git
- Plot display capability (Plotly opens an interactive window or browser tab)

## Setup
```bash
# Create virtual environment (any OS)
python -m venv .venv

# Activate
# Windows PowerShell
. .venv/Scripts/Activate.ps1
# Windows cmd
.venv\Scripts\activate.bat
# macOS/Linux
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install pandas numpy scikit-learn plotly xgboost requests
```

## Run
```bash
python baseCode.py
```
The script will:
1) Download data for 2022-23, 2023-24, 2024-25.
2) Train all models and show an accuracy comparison bar chart.
3) Predict the sample match (default: Leeds vs Liverpool) and show per-model probability pie charts.

## Changing the fixture
Edit `main()` in `baseCode.py` and replace the `home_team` and `away_team` arguments in `predict_match()`.

## Notes
- `.venv/` is ignored via `.gitignore`; don't commit your virtualenv.
- If Plotly windows don't appear, ensure you're running in an environment that can open browser windows.
- https://www.youtube.com/watch?v=3pnkARyrtMo&list=PLedeYskZY0vBOdQ6Uc9eZjZ2-nz1JT3R7&index=12 - Youtube link for match predictions
