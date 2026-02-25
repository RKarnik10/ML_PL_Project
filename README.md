# ML_PL_Project
PL ML Random Forest Project

###Notes:
## Original Base code

1. Beginning
- uses pandas and RF Classification to create a base model and data of all season matches
1. Data Acquisition
- Uses seasons 24-25 and 25-26
- May consider including 22-23, 3-24 (extra data on Burnley and Leeds)
1. Cleaning
- Skips missing data
- Computes result label → H, A, D
- Outputs df with match values
1. Feature Engineering
- Copies data and feature columns
- Grabs last 5 matches as reference, computes multiple stats including: point, goal diff., and strength “form”
1. Training
- Feature set with columns
- Train model
    - builds the features for all matches
    - uses first 50 rows
    - splits 80/20 and fits
    - Prints accuracy, prediction, and plotty
1. Prediction
- Uses last 100 match as “pool”
- Takes last 5 games for form
- Makes predictions

---

# Conversion to XGBoost

XGBoost in SciKitLearn Changes

- Modifications to code:
    - Encoded labels (`H/D/A` → 0/1/2).
    - `eval_set` and `early_stopping_rounds`.
- For multiclass classification you set:
    - Setting objective to model as multi-probs
        - `objective="multi:softprob"`
    - Setting 3 outcomes
        - `num_class=3`

## **Notes for differences:
XGBoost vs Random Forest (brief)**

**Random Forest**

- Builds **many independent decision trees** (bagging).
- Each tree sees a random bootstrap sample + random subset of features.
- Final prediction is **vote/average** across trees.
- Pros: solid baseline, less tuning, fairly robust.
- Cons: can be weaker on structured/tabular problems than boosting; probabilities can be less sharp; doesn’t “focus” on mistakes.

**XGBoost**

- Builds trees **sequentially**, where each new tree tries to **correct the errors** of the previous ones (gradient boosting).
- Usually stronger on tabular data, can model subtle patterns.
- Pros: often higher accuracy, better probability modeling, lots of regularization + early stopping.
- Cons: more hyperparameters, can overfit if you’re sloppy, needs careful validation (time-based split).

## Changes

- Changed to take form of last 10 matches, not just 5 (overly form dependent had leeds as a favorite)
- Did not just use last 100 games as pool

## Possible Changes to Consider

- Include more features (already calculated): goals_away, and conceded h/w
- Consider short term form (5) and longer form (15 matches) with manual or xgboost model weights
- Include 22-23, 23-24 season, mostly for training data (watch overfitting)
- Add team one-hot columns

- Switch to a time-based split (look more into)

###Future Steps:
- Look into dashboards (streamlit - Python library)
- Look into changes
