"""
Premier League Match Predictor — v2
=====================================
Key improvements over xgtest_codex_modified.py:
  - Dual form windows (5-game short-term + 15-game long-term) per team
  - All computed features included (away goals avg, conceded h/a)
  - Correct CSV filenames (PL_2425.csv / PL_2526.csv)
  - Optional historical seasons (22-23, 23-24) via include_historical flag
  - Team name validation with suggestions on mismatch
  - Chronological (time-based) train/val/test split — no leakage
  - Goal difference feature added
"""

import pandas as pd
import numpy as np
import warnings
from pathlib import Path

from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

import plotly.express as px

warnings.filterwarnings('ignore')


# ---------------------------------------------------------------------------
# Helper: compute stats for one form window
# ---------------------------------------------------------------------------
def _compute_stats(history: pd.DataFrame, team: str, default_goals: float = 1.5):
    """Return a dict of stats for `team` from a slice of historical matches."""
    if len(history) == 0:
        return dict(strength=50, form=5, goals_for=default_goals, goals_against=default_goals)

    points = goals_scored = goals_conceded = 0

    for _, m in history.iterrows():
        if m['home_team'] == team:
            goals_scored  += m['home_goals']
            goals_conceded += m['away_goals']
            if m['result'] == 'H':
                points += 3
            elif m['result'] == 'D':
                points += 1
        else:
            goals_scored  += m['away_goals']
            goals_conceded += m['home_goals']
            if m['result'] == 'A':
                points += 3
            elif m['result'] == 'D':
                points += 1

    n = len(history)
    strength = min(90, max(10, (points / n) * 20 + 20))

    return dict(
        strength=strength,
        form=points,
        goals_for=goals_scored / n,
        goals_against=goals_conceded / n,
    )


class PLPredictor:
    # Season CSV map — add more entries here if you have the files
    SEASON_FILES = {
        '2022-23': ['PL_2223.csv', 'PL 22-23.csv'],
        '2023-24': ['PL_2324.csv', 'PL 23-24.csv'],
        '2024-25': ['PL_2425.csv', 'PL 24-25.csv'],
        '2025-26': ['PL_2526.csv', 'PL 25-26.csv'],
    }

    def __init__(self, short_window: int = 5, long_window: int = 15):
        self.short_window = short_window
        self.long_window  = long_window

        self.label_encoder = LabelEncoder()

        self.model = XGBClassifier(
            n_estimators=1200,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            objective="multi:softprob",
            num_class=3,
            random_state=42,
            eval_metric="mlogloss",
            tree_method="hist",
            early_stopping_rounds=50,
        )

        self.match_data: pd.DataFrame | None = None
        self._feature_columns: list[str] = []

    # ------------------------------------------------------------------
    # 1. Data loading
    # ------------------------------------------------------------------
    def get_football_data(self, include_historical: bool = False):
        seasons = ['2024-25', '2025-26']
        if include_historical:
            seasons = ['2022-23', '2023-24'] + seasons

        all_matches = []
        for season in seasons:
            candidates = self.SEASON_FILES.get(season, [])
            path = next((p for p in candidates if Path(p).exists()), None)
            if path is None:
                print(f"  [skip] {season} — file not found ({candidates})")
                continue
            try:
                raw = pd.read_csv(path)
                cleaned = self._clean_season(raw, season)
                all_matches.append(cleaned)
                print(f"  [ok]   {season} — {len(cleaned)} matches from {path}")
            except Exception as e:
                print(f"  [err]  {season} — {e}")

        if not all_matches:
            print("No data loaded.")
            return False

        self.match_data = pd.concat(all_matches, ignore_index=True)
        self.match_data.sort_values('date', inplace=True)
        self.match_data.reset_index(drop=True, inplace=True)
        print(f"Total matches loaded: {len(self.match_data)}")
        return True

    def _clean_season(self, data: pd.DataFrame, season: str) -> pd.DataFrame:
        rows = []
        for _, row in data.iterrows():
            if pd.isna(row.get('FTHG')) or pd.isna(row.get('FTAG')):
                continue
            try:
                hg = int(row['FTHG'])
                ag = int(row['FTAG'])
            except (ValueError, TypeError):
                continue

            result = 'H' if hg > ag else ('A' if ag > hg else 'D')
            date   = pd.to_datetime(row.get('Date', pd.NaT), errors='coerce', dayfirst=True)

            rows.append(dict(
                season=season,
                date=date,
                home_team=row.get('HomeTeam', ''),
                away_team=row.get('AwayTeam', ''),
                home_goals=hg,
                away_goals=ag,
                result=result,
            ))

        return pd.DataFrame(rows).sort_values('date').reset_index(drop=True)

    # ------------------------------------------------------------------
    # 2. Feature engineering — dual form windows
    # ------------------------------------------------------------------
    def _build_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        For each match, compute short-form (5) and long-form (15) stats
        for both teams, plus goal-difference proxy.
        All lookbacks are strictly prior to the current match index.
        """
        df = data.copy().sort_values('date').reset_index(drop=True)

        # Pre-allocate columns
        for prefix in ('h_short', 'h_long', 'a_short', 'a_long'):
            for stat in ('strength', 'form', 'gf', 'ga'):
                df[f'{prefix}_{stat}'] = np.nan

        df['home_advantage'] = 1

        for i, match in df.iterrows():
            ht = match['home_team']
            at = match['away_team']

            prior = df[df.index < i]
            ht_prior = prior[(prior['home_team'] == ht) | (prior['away_team'] == ht)]
            at_prior = prior[(prior['home_team'] == at) | (prior['away_team'] == at)]

            hs = _compute_stats(ht_prior.tail(self.short_window), ht)
            hl = _compute_stats(ht_prior.tail(self.long_window),  ht)
            as_ = _compute_stats(at_prior.tail(self.short_window), at)
            al  = _compute_stats(at_prior.tail(self.long_window),  at)

            df.loc[i, 'h_short_strength'] = hs['strength']
            df.loc[i, 'h_short_form']     = hs['form']
            df.loc[i, 'h_short_gf']       = hs['goals_for']
            df.loc[i, 'h_short_ga']       = hs['goals_against']

            df.loc[i, 'h_long_strength']  = hl['strength']
            df.loc[i, 'h_long_form']      = hl['form']
            df.loc[i, 'h_long_gf']        = hl['goals_for']
            df.loc[i, 'h_long_ga']        = hl['goals_against']

            df.loc[i, 'a_short_strength'] = as_['strength']
            df.loc[i, 'a_short_form']     = as_['form']
            df.loc[i, 'a_short_gf']       = as_['goals_for']
            df.loc[i, 'a_short_ga']       = as_['goals_against']

            df.loc[i, 'a_long_strength']  = al['strength']
            df.loc[i, 'a_long_form']      = al['form']
            df.loc[i, 'a_long_gf']        = al['goals_for']
            df.loc[i, 'a_long_ga']        = al['goals_against']

        # Derived difference features (give the model relative signals)
        df['strength_diff_short'] = df['h_short_strength'] - df['a_short_strength']
        df['strength_diff_long']  = df['h_long_strength']  - df['a_long_strength']
        df['form_diff_short']     = df['h_short_form']     - df['a_short_form']
        df['form_diff_long']      = df['h_long_form']      - df['a_long_form']
        df['gf_diff_short']       = df['h_short_gf']       - df['a_short_gf']
        df['gf_diff_long']        = df['h_long_gf']        - df['a_long_gf']

        print(f"Features built for {len(df)} matches.")
        return df

    # ------------------------------------------------------------------
    # 3. Training
    # ------------------------------------------------------------------
    def _feature_cols(self) -> list[str]:
        return [
            # Short-form (5 games)
            'h_short_strength', 'h_short_form', 'h_short_gf', 'h_short_ga',
            'a_short_strength', 'a_short_form', 'a_short_gf', 'a_short_ga',
            # Long-form (15 games)
            'h_long_strength',  'h_long_form',  'h_long_gf',  'h_long_ga',
            'a_long_strength',  'a_long_form',  'a_long_gf',  'a_long_ga',
            # Relative differences
            'strength_diff_short', 'strength_diff_long',
            'form_diff_short',     'form_diff_long',
            'gf_diff_short',       'gf_diff_long',
            # Context
            'home_advantage',
        ]

    @staticmethod
    def _time_split(X, y, train_frac=0.70, val_frac=0.15):
        n = len(X)
        t = int(n * train_frac)
        v = int(n * (train_frac + val_frac))
        return (
            X.iloc[:t],  y[:t],
            X.iloc[t:v], y[t:v],
            X.iloc[v:],  y[v:],
        )

    def train_model(self, include_historical: bool = False):
        if self.match_data is None:
            print("No data loaded.")
            return False

        featured = self._build_features(self.match_data)
        # Drop the first long_window rows — those have sparse lookback history
        featured = featured.iloc[self.long_window:].reset_index(drop=True)

        self._feature_columns = self._feature_cols()
        X = featured[self._feature_columns].copy()
        y = self.label_encoder.fit_transform(featured['result'])

        print(f"Classes: {list(self.label_encoder.classes_)}")
        print(f"Dataset size after warmup rows removed: {len(X)}")

        X_train, y_train, X_val, y_val, X_test, y_test = self._time_split(X, y)
        print(f"Split → train: {len(X_train)}, val: {len(X_val)}, test: {len(X_test)}")

        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        preds = self.model.predict(X_test)
        acc   = accuracy_score(y_test, preds)
        print(f"\nTest accuracy: {acc:.1%}  ({len(X_test)} matches)")

        label_names = self.label_encoder.classes_
        print("\nClassification report:")
        print(classification_report(
            y_test, preds,
            target_names=label_names,
            zero_division=0,
        ))

        if hasattr(self.model, 'best_iteration') and self.model.best_iteration:
            print(f"Best XGBoost iteration (early stopping): {self.model.best_iteration}")

        # Feature importance chart
        fi = pd.DataFrame({
            'Feature':    self._feature_columns,
            'Importance': self.model.feature_importances_,
        }).sort_values('Importance', ascending=False)

        fig = px.bar(
            fi, x='Importance', y='Feature', orientation='h',
            title='Feature Importance (XGBoost v2)',
            text=fi['Importance'].round(3),
        )
        fig.update_traces(textposition='outside')
        fig.update_layout(yaxis=dict(autorange='reversed'), template='plotly_white')
        fig.show()

        return True

    # ------------------------------------------------------------------
    # 4. Prediction
    # ------------------------------------------------------------------
    def known_teams(self) -> list[str]:
        if self.match_data is None:
            return []
        return sorted(set(self.match_data['home_team']) | set(self.match_data['away_team']))

    def _validate_team(self, name: str) -> str:
        """Return the name if known; print suggestions and raise if not."""
        teams = self.known_teams()
        if name in teams:
            return name
        # Simple case-insensitive suggestion
        suggestions = [t for t in teams if name.lower() in t.lower() or t.lower() in name.lower()]
        msg = f"Unknown team '{name}'."
        if suggestions:
            msg += f" Did you mean one of: {suggestions}?"
        else:
            msg += f" Known teams: {teams}"
        raise ValueError(msg)

    def predict_match(self, home_team: str, away_team: str):
        home_team = self._validate_team(home_team)
        away_team = self._validate_team(away_team)

        print(f"\nPREDICTING: {home_team} (H)  vs  {away_team} (A)")
        print("=" * 55)

        prior = self.match_data

        ht_prior = prior[(prior['home_team'] == home_team) | (prior['away_team'] == home_team)]
        at_prior = prior[(prior['home_team'] == away_team) | (prior['away_team'] == away_team)]

        hs  = _compute_stats(ht_prior.tail(self.short_window), home_team)
        hl  = _compute_stats(ht_prior.tail(self.long_window),  home_team)
        as_ = _compute_stats(at_prior.tail(self.short_window), away_team)
        al  = _compute_stats(at_prior.tail(self.long_window),  away_team)

        feat = pd.DataFrame([{
            'h_short_strength': hs['strength'],
            'h_short_form':     hs['form'],
            'h_short_gf':       hs['goals_for'],
            'h_short_ga':       hs['goals_against'],
            'a_short_strength': as_['strength'],
            'a_short_form':     as_['form'],
            'a_short_gf':       as_['goals_for'],
            'a_short_ga':       as_['goals_against'],
            'h_long_strength':  hl['strength'],
            'h_long_form':      hl['form'],
            'h_long_gf':        hl['goals_for'],
            'h_long_ga':        hl['goals_against'],
            'a_long_strength':  al['strength'],
            'a_long_form':      al['form'],
            'a_long_gf':        al['goals_for'],
            'a_long_ga':        al['goals_against'],
            'strength_diff_short': hs['strength'] - as_['strength'],
            'strength_diff_long':  hl['strength'] - al['strength'],
            'form_diff_short':     hs['form']     - as_['form'],
            'form_diff_long':      hl['form']     - al['form'],
            'gf_diff_short':       hs['goals_for'] - as_['goals_for'],
            'gf_diff_long':        hl['goals_for'] - al['goals_for'],
            'home_advantage':      1,
        }])

        print("\nFeatures fed to model:")
        print(feat.T.rename(columns={0: 'value'}).to_string())

        pred_int  = self.model.predict(feat)[0]
        proba     = self.model.predict_proba(feat)[0]
        pred_lbl  = self.label_encoder.inverse_transform([pred_int])[0]
        classes   = self.label_encoder.inverse_transform(np.arange(len(proba)))

        labels = {'H': f'{home_team} Win', 'D': 'Draw', 'A': f'{away_team} Win'}

        print(f"\nPREDICTION: {labels[pred_lbl]}")
        print("Probabilities:")
        for lbl, p in sorted(zip(classes, proba), key=lambda x: -x[1]):
            print(f"  {labels[lbl]:<25} {p:.1%}")

        prob_df = pd.DataFrame({
            'Outcome':     [labels[l] for l in classes],
            'Probability': proba,
        })
        fig = px.pie(
            prob_df, names='Outcome', values='Probability',
            title=f'{home_team} vs {away_team}',
        )
        fig.show()

        return pred_lbl, dict(zip(classes, proba))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    print("Premier League Predictor — v2")
    print("=" * 55)

    predictor = PLPredictor(short_window=5, long_window=15)

    # Set include_historical=True once you have PL_2223.csv / PL_2324.csv
    if not predictor.get_football_data(include_historical=False):
        return

    if not predictor.train_model():
        return

    # Example predictions — use exact team names from the CSV
    # Run predictor.known_teams() to see the full list
    print("\nKnown teams:", predictor.known_teams())

    matches_to_predict = [
        ('Tottenham', 'Arsenal'),
        ('Liverpool', 'Man City'),
        ('Chelsea',   'Man United'),
    ]

    for home, away in matches_to_predict:
        try:
            predictor.predict_match(home, away)
        except ValueError as e:
            print(f"[skip] {e}")


if __name__ == '__main__':
    main()