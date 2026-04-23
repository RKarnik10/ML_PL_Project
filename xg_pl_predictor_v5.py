"""
Premier League Match Predictor — v5
=====================================
Features vs v4:
  - Auto-downloads CSVs from football-data.co.uk (local files used as fallback)
  - Shots on target (SOT) rolling features
  - Rolling market-implied win probability + Asian handicap line from historical odds
  - All features consistent between training and prediction (no external odds input needed)
"""

import gc
import warnings
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import requests
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Auto-download helpers
# ---------------------------------------------------------------------------
_FD_BASE = "https://www.football-data.co.uk/mmz4281/{code}/E0.csv"
_SEASON_CODES = {
    "2022-23": "2223",
    "2023-24": "2324",
    "2024-25": "2425",
    "2025-26": "2526",
}


def _download_season(season: str) -> pd.DataFrame | None:
    code = _SEASON_CODES.get(season)
    if not code:
        return None
    try:
        r = requests.get(_FD_BASE.format(code=code), timeout=10)
        r.raise_for_status()
        return pd.read_csv(StringIO(r.text))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Helper: compute stats for one form window
# ---------------------------------------------------------------------------
def _compute_stats(history: pd.DataFrame, team: str, default_goals: float = 1.5):
    if len(history) == 0:
        return dict(strength=50, form=5, goals_for=default_goals, goals_against=default_goals,
                    sot=4.0, mkt_win=1/3, ah=0.0)

    points = goals_scored = goals_conceded = shots = mkt_win_sum = ah_sum = 0.0

    for _, m in history.iterrows():
        if m["home_team"] == team:
            goals_scored += m["home_goals"]
            goals_conceded += m["away_goals"]
            shots += float(m.get("hst") or 4) if not pd.isna(m.get("hst") or 0) else 4.0
            mkt_win_sum += float(m.get("implied_h") or 1/3)
            ah_sum += float(m.get("ah_line") or 0)
            if m["result"] == "H":
                points += 3
            elif m["result"] == "D":
                points += 1
        else:
            goals_scored += m["away_goals"]
            goals_conceded += m["home_goals"]
            shots += float(m.get("ast") or 4) if not pd.isna(m.get("ast") or 0) else 4.0
            mkt_win_sum += float(m.get("implied_a") or 1/3)
            ah_sum += -float(m.get("ah_line") or 0)
            if m["result"] == "A":
                points += 3
            elif m["result"] == "D":
                points += 1

    n = len(history)
    strength = min(90, max(10, (points / n) * 20 + 20))
    return dict(
        strength=strength,
        form=points,
        goals_for=goals_scored / n,
        goals_against=goals_conceded / n,
        sot=shots / n,
        mkt_win=mkt_win_sum / n,
        ah=ah_sum / n,
    )


class PLPredictor:
    SEASON_FILES = {
        "2022-23": ["PL_2223.csv", "PL 22-23.csv"],
        "2023-24": ["PL_2324.csv", "PL 23-24.csv"],
        "2024-25": ["PL_2425.csv", "PL 24-25.csv"],
        "2025-26": ["PL_2526.csv", "PL 25-26.csv"],
    }

    def __init__(self, short_window: int = 5, long_window: int = 15, prior_season_weight: float = 0.4):
        self.short_window = short_window
        self.long_window = long_window
        self.prior_season_weight = prior_season_weight
        self.label_encoder = LabelEncoder()
        self.model = XGBClassifier(
            n_estimators=600,
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
    def get_football_data(self, include_historical: bool = False) -> bool:
        seasons = ["2024-25", "2025-26"]
        if include_historical:
            seasons = ["2022-23", "2023-24"] + seasons

        all_matches = []
        for season in seasons:
            raw = _download_season(season)
            source = "football-data.co.uk"
            if raw is None:
                candidates = self.SEASON_FILES.get(season, [])
                path = next((p for p in candidates if Path(p).exists()), None)
                if path is None:
                    print(f"  [skip] {season} — not found online or locally")
                    continue
                try:
                    raw = pd.read_csv(path)
                    source = "local file"
                except Exception as e:
                    print(f"  [err]  {season} — {e}")
                    continue
            try:
                cleaned = self._clean_season(raw, season)
                all_matches.append(cleaned)
                print(f"  [ok]   {season} — {len(cleaned)} matches ({source})")
            except Exception as e:
                print(f"  [err]  {season} — {e}")

        if not all_matches:
            print("No data loaded.")
            return False

        self.match_data = pd.concat(all_matches, ignore_index=True)
        self.match_data.sort_values("date", inplace=True)
        self.match_data.reset_index(drop=True, inplace=True)
        print(f"Total matches loaded: {len(self.match_data)}")
        return True

    def _clean_season(self, data: pd.DataFrame, season: str) -> pd.DataFrame:
        rows = []
        for _, row in data.iterrows():
            if pd.isna(row.get("FTHG")) or pd.isna(row.get("FTAG")):
                continue
            try:
                hg, ag = int(row["FTHG"]), int(row["FTAG"])
            except (ValueError, TypeError):
                continue

            result = "H" if hg > ag else ("A" if ag > hg else "D")
            date = pd.to_datetime(row.get("Date", pd.NaT), errors="coerce", dayfirst=True)

            try:
                hst = int(float(row.get("HST") or 4))
                ast_val = int(float(row.get("AST") or 4))
            except (ValueError, TypeError):
                hst, ast_val = 4, 4

            try:
                raw_h = 1.0 / float(row.get("AvgH") or row.get("B365H") or 3.0)
                raw_d = 1.0 / float(row.get("AvgD") or row.get("B365D") or 3.3)
                raw_a = 1.0 / float(row.get("AvgA") or row.get("B365A") or 3.0)
                total = raw_h + raw_d + raw_a
                imp_h, imp_d, imp_a = raw_h / total, raw_d / total, raw_a / total
            except (ValueError, TypeError, ZeroDivisionError):
                imp_h, imp_d, imp_a = 1/3, 1/3, 1/3

            try:
                ah = float(row.get("AHh") or 0)
            except (ValueError, TypeError):
                ah = 0.0

            rows.append(dict(
                season=season,
                date=date,
                home_team=row.get("HomeTeam", ""),
                away_team=row.get("AwayTeam", ""),
                home_goals=hg,
                away_goals=ag,
                result=result,
                hst=hst,
                ast=ast_val,
                implied_h=imp_h,
                implied_d=imp_d,
                implied_a=imp_a,
                ah_line=ah,
            ))

        return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)

    # ------------------------------------------------------------------
    # 2. Feature engineering
    # ------------------------------------------------------------------
    def _build_features(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy().sort_values("date").reset_index(drop=True)

        df["h_pts"] = df["result"].map({"H": 3, "D": 1, "A": 0})
        df["a_pts"] = df["result"].map({"A": 3, "D": 1, "H": 0})
        df["h_gf"], df["h_ga"] = df["home_goals"], df["away_goals"]
        df["a_gf"], df["a_ga"] = df["away_goals"], df["home_goals"]

        def _rolling_team_stats(df: pd.DataFrame, window: int) -> pd.DataFrame:
            home_rows = df[["date", "season", "home_team", "h_pts", "h_gf", "h_ga", "hst", "implied_h", "ah_line"]].copy()
            home_rows.columns = ["date", "season", "team", "pts", "gf", "ga", "sot", "mkt_win", "ah"]
            away_rows = df[["date", "season", "away_team", "a_pts", "a_gf", "a_ga", "ast", "implied_a", "ah_line"]].copy()
            away_rows.columns = ["date", "season", "team", "pts", "gf", "ga", "sot", "mkt_win", "ah"]
            away_rows["ah"] = -away_rows["ah"]

            combined = pd.concat([home_rows, away_rows]).sort_values(["team", "date"]).reset_index(drop=True)

            current_season = combined["season"].max()
            weight = combined["season"].apply(lambda s: 1.0 if s == current_season else self.prior_season_weight)
            combined["pts"] *= weight
            combined["gf"] *= weight
            combined["ga"] *= weight

            grp = combined.groupby("team")[["pts", "gf", "ga", "sot", "mkt_win", "ah"]]
            rolled = grp.apply(lambda x: x.shift(1).rolling(window, min_periods=1).sum()).reset_index(level=0, drop=True)
            count = grp.apply(lambda x: x["pts"].shift(1).rolling(window, min_periods=1).count()).reset_index(level=0, drop=True).rename("n")

            combined = combined.join(rolled, rsuffix="_roll").join(count)
            combined["n"] = combined["n"].clip(lower=1)
            combined["gf_avg"] = combined["gf_roll"] / combined["n"]
            combined["ga_avg"] = combined["ga_roll"] / combined["n"]
            combined["sot_avg"] = combined["sot_roll"] / combined["n"]
            combined["mkt_win_avg"] = combined["mkt_win_roll"] / combined["n"]
            combined["ah_avg"] = combined["ah_roll"] / combined["n"]
            combined["strength"] = (combined["pts_roll"] / combined["n"]) * 20 + 20
            combined["strength"] = combined["strength"].clip(10, 90)
            return combined[["date", "season", "team", "pts_roll", "gf_avg", "ga_avg", "strength", "sot_avg", "mkt_win_avg", "ah_avg"]]

        short = _rolling_team_stats(df, self.short_window)
        long = _rolling_team_stats(df, self.long_window)

        for side, team_col in [("h", "home_team"), ("a", "away_team")]:
            for prefix, stats_df in [("short", short), ("long", long)]:
                merged = df[["date", team_col]].merge(
                    stats_df, left_on=["date", team_col], right_on=["date", "team"], how="left"
                )
                df[f"{side}_{prefix}_form"] = merged["pts_roll"].values
                df[f"{side}_{prefix}_gf"] = merged["gf_avg"].values
                df[f"{side}_{prefix}_ga"] = merged["ga_avg"].values
                df[f"{side}_{prefix}_strength"] = merged["strength"].values
                df[f"{side}_{prefix}_sot"] = merged["sot_avg"].values
                df[f"{side}_{prefix}_mkt"] = merged["mkt_win_avg"].values
                df[f"{side}_{prefix}_ah"] = merged["ah_avg"].values

        for prefix in ("h_short", "h_long", "a_short", "a_long"):
            df[f"{prefix}_form"].fillna(5, inplace=True)
            df[f"{prefix}_gf"].fillna(1.5, inplace=True)
            df[f"{prefix}_ga"].fillna(1.5, inplace=True)
            df[f"{prefix}_strength"].fillna(50, inplace=True)
            df[f"{prefix}_sot"].fillna(4.0, inplace=True)
            df[f"{prefix}_mkt"].fillna(1/3, inplace=True)
            df[f"{prefix}_ah"].fillna(0.0, inplace=True)

        df["home_advantage"] = 1
        df.drop(columns=["h_pts", "a_pts", "h_gf", "h_ga", "a_gf", "a_ga"], inplace=True)
        del short, long
        gc.collect()

        df["strength_diff_short"] = df["h_short_strength"] - df["a_short_strength"]
        df["strength_diff_long"] = df["h_long_strength"] - df["a_long_strength"]
        df["form_diff_short"] = df["h_short_form"] - df["a_short_form"]
        df["form_diff_long"] = df["h_long_form"] - df["a_long_form"]
        df["gf_diff_short"] = df["h_short_gf"] - df["a_short_gf"]
        df["gf_diff_long"] = df["h_long_gf"] - df["a_long_gf"]
        df["sot_diff_short"] = df["h_short_sot"] - df["a_short_sot"]
        df["sot_diff_long"] = df["h_long_sot"] - df["a_long_sot"]
        df["mkt_diff_short"] = df["h_short_mkt"] - df["a_short_mkt"]
        df["mkt_diff_long"] = df["h_long_mkt"] - df["a_long_mkt"]

        print(f"Features built for {len(df)} matches.")
        return df

    # ------------------------------------------------------------------
    # 3. Training
    # ------------------------------------------------------------------
    def _feature_cols(self) -> list[str]:
        return [
            "h_short_strength", "h_short_form", "h_short_gf", "h_short_ga", "h_short_sot", "h_short_mkt", "h_short_ah",
            "a_short_strength", "a_short_form", "a_short_gf", "a_short_ga", "a_short_sot", "a_short_mkt", "a_short_ah",
            "h_long_strength", "h_long_form", "h_long_gf", "h_long_ga", "h_long_sot", "h_long_mkt", "h_long_ah",
            "a_long_strength", "a_long_form", "a_long_gf", "a_long_ga", "a_long_sot", "a_long_mkt", "a_long_ah",
            "strength_diff_short", "strength_diff_long",
            "form_diff_short", "form_diff_long",
            "gf_diff_short", "gf_diff_long",
            "sot_diff_short", "sot_diff_long",
            "mkt_diff_short", "mkt_diff_long",
            "home_advantage",
        ]

    @staticmethod
    def _time_split(X, y, train_frac=0.70, val_frac=0.15):
        n = len(X)
        t, v = int(n * train_frac), int(n * (train_frac + val_frac))
        return X.iloc[:t], y[:t], X.iloc[t:v], y[t:v], X.iloc[v:], y[v:]

    def train_model(self) -> bool:
        if self.match_data is None:
            print("No data loaded.")
            return False

        featured = self._build_features(self.match_data)
        featured = featured.iloc[self.long_window:].reset_index(drop=True)

        self._feature_columns = self._feature_cols()
        X = featured[self._feature_columns].copy()
        y = self.label_encoder.fit_transform(featured["result"])

        del featured
        gc.collect()

        print(f"Classes: {list(self.label_encoder.classes_)}")
        print(f"Dataset size: {len(X)}")

        X_train, y_train, X_val, y_val, X_test, y_test = self._time_split(X, y)
        print(f"Split → train: {len(X_train)}, val: {len(X_val)}, test: {len(X_test)}")

        sample_weights = np.exp(np.linspace(0, 1, len(X_train)))
        self.model.fit(X_train, y_train, sample_weight=sample_weights, eval_set=[(X_val, y_val)], verbose=False)

        preds = self.model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        print(f"\nTest accuracy: {acc:.1%}  ({len(X_test)} matches)")
        print("\nClassification report:")
        print(classification_report(y_test, preds, target_names=self.label_encoder.classes_, zero_division=0))

        if hasattr(self.model, "best_iteration") and self.model.best_iteration:
            print(f"Best XGBoost iteration: {self.model.best_iteration}")

        fi = pd.DataFrame({
            "Feature": self._feature_columns,
            "Importance": self.model.feature_importances_,
        }).sort_values("Importance", ascending=False)

        fig = px.bar(
            fi, x="Importance", y="Feature", orientation="h",
            title="Feature Importance (XGBoost v5)",
            text=fi["Importance"].round(3),
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(yaxis=dict(autorange="reversed"), template="plotly_white")
        fig.show()

        return True

    # ------------------------------------------------------------------
    # 4. Prediction
    # ------------------------------------------------------------------
    def known_teams(self) -> list[str]:
        if self.match_data is None:
            return []
        return sorted(set(self.match_data["home_team"]) | set(self.match_data["away_team"]))

    def _validate_team(self, name: str) -> str:
        teams = self.known_teams()
        if name in teams:
            return name
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
        ht_p = prior[(prior["home_team"] == home_team) | (prior["away_team"] == home_team)]
        at_p = prior[(prior["home_team"] == away_team) | (prior["away_team"] == away_team)]

        hs  = _compute_stats(ht_p.tail(self.short_window), home_team)
        hl  = _compute_stats(ht_p.tail(self.long_window),  home_team)
        as_ = _compute_stats(at_p.tail(self.short_window), away_team)
        al  = _compute_stats(at_p.tail(self.long_window),  away_team)

        feat = pd.DataFrame([{
            "h_short_strength": hs["strength"], "h_short_form": hs["form"],
            "h_short_gf": hs["goals_for"],      "h_short_ga": hs["goals_against"],
            "h_short_sot": hs["sot"],           "h_short_mkt": hs["mkt_win"],   "h_short_ah": hs["ah"],
            "a_short_strength": as_["strength"], "a_short_form": as_["form"],
            "a_short_gf": as_["goals_for"],      "a_short_ga": as_["goals_against"],
            "a_short_sot": as_["sot"],           "a_short_mkt": as_["mkt_win"],  "a_short_ah": as_["ah"],
            "h_long_strength": hl["strength"],   "h_long_form": hl["form"],
            "h_long_gf": hl["goals_for"],        "h_long_ga": hl["goals_against"],
            "h_long_sot": hl["sot"],             "h_long_mkt": hl["mkt_win"],    "h_long_ah": hl["ah"],
            "a_long_strength": al["strength"],   "a_long_form": al["form"],
            "a_long_gf": al["goals_for"],        "a_long_ga": al["goals_against"],
            "a_long_sot": al["sot"],             "a_long_mkt": al["mkt_win"],    "a_long_ah": al["ah"],
            "strength_diff_short": hs["strength"] - as_["strength"],
            "strength_diff_long":  hl["strength"] - al["strength"],
            "form_diff_short":     hs["form"]     - as_["form"],
            "form_diff_long":      hl["form"]     - al["form"],
            "gf_diff_short":       hs["goals_for"] - as_["goals_for"],
            "gf_diff_long":        hl["goals_for"] - al["goals_for"],
            "sot_diff_short":      hs["sot"] - as_["sot"],
            "sot_diff_long":       hl["sot"] - al["sot"],
            "mkt_diff_short":      hs["mkt_win"] - as_["mkt_win"],
            "mkt_diff_long":       hl["mkt_win"] - al["mkt_win"],
            "home_advantage": 1,
        }])

        print("\nFeatures fed to model:")
        print(feat.T.rename(columns={0: "value"}).to_string())

        pred_int = self.model.predict(feat)[0]
        proba    = self.model.predict_proba(feat)[0]
        pred_lbl = self.label_encoder.inverse_transform([pred_int])[0]
        classes  = self.label_encoder.inverse_transform(np.arange(len(proba)))

        labels = {"H": f"{home_team} Win", "D": "Draw", "A": f"{away_team} Win"}
        print(f"\nPREDICTION: {labels[pred_lbl]}")
        print("Probabilities:")
        for lbl, p in sorted(zip(classes, proba), key=lambda x: -x[1]):
            print(f"  {labels[lbl]:<25} {p:.1%}")

        prob_df = pd.DataFrame({
            "Outcome": [labels[l] for l in classes],
            "Probability": proba,
        })
        fig = px.pie(
            prob_df, names="Outcome", values="Probability",
            title=f"{home_team} vs {away_team}",
        )
        fig.show()

        return pred_lbl, dict(zip(classes, proba))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    print("Premier League Predictor — v5")
    print("=" * 55)

    predictor = PLPredictor(short_window=5, long_window=15, prior_season_weight=0.4)

    if not predictor.get_football_data(include_historical=True):
        return

    if not predictor.train_model():
        return

    print("\nKnown teams:", predictor.known_teams())

    matches_to_predict = [
        ("Tottenham", "Arsenal"),
        ("Liverpool", "Man City"),
        ("Chelsea", "Man United"),
    ]

    for home, away in matches_to_predict:
        try:
            predictor.predict_match(home, away)
        except ValueError as e:
            print(f"[skip] {e}")


if __name__ == "__main__":
    main()
