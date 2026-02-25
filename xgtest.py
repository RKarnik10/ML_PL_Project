import pandas as pd
import numpy as np
import requests
import warnings
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

from xgboost import XGBClassifier

warnings.filterwarnings('ignore')


class PLPredictor:
    def __init__(self):
        # Label encoder for H/D/A
        self.label_encoder = LabelEncoder()

        # XGBoost model (good starting defaults)
        self.model = XGBClassifier(
            n_estimators=800,          # boosted trees; early stopping will pick best
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
            early_stopping_rounds=50,# fast on CPU
        )

        self.match_data = None

    def get_football_data(self):
        print("Downloading Premier League data...")
        season_files = {
            '2024-25': 'PL 24-25.csv',
            '2025-26': 'Pl 25-26.csv'
        }

        all_matches = []



        for season, path in season_files.items():
            try:
                print(f"  Reading {season} from {path} ...")
                season_data = pd.read_csv(path)
                clean_data = self.clean_season_data(season_data, season)
                all_matches.append(clean_data)
                print(f"Got {len(clean_data)} matches from {season}")

            except Exception as e:
                print(f"Couldn't read {season} data from {path}: {e}")

        if all_matches:
            self.match_data = pd.concat(all_matches, ignore_index=True)
            print(f"Total matches loaded: {len(self.match_data)}")
            return True
        else:
            print("No data could be loaded")
            return False

    def clean_season_data(self, data, season):
        matches = []

        for _, row in data.iterrows():
            try:
                if pd.isna(row.get('FTHG')) or pd.isna(row.get('FTAG')):
                    continue

                home_team = row.get('HomeTeam', '')
                away_team = row.get('AwayTeam', '')
                home_goals = int(row['FTHG'])
                away_goals = int(row['FTAG'])

                if home_goals > away_goals:
                    result = 'H'
                elif away_goals > home_goals:
                    result = 'A'
                else:
                    result = 'D'

                matches.append({
                    'season': season,
                    'home_team': home_team,
                    'away_team': away_team,
                    'home_goals': home_goals,
                    'away_goals': away_goals,
                    'result': result
                })

            except (ValueError, TypeError):
                continue

        return pd.DataFrame(matches)

    def calculate_simple_features(self, data):
        enhanced_data = data.copy().sort_values(['season']).reset_index(drop=True)

        enhanced_data['home_team_strength'] = 50
        enhanced_data['away_team_strength'] = 50
        enhanced_data['home_recent_form'] = 5
        enhanced_data['away_recent_form'] = 5
        enhanced_data['home_goals_avg'] = 1.5
        enhanced_data['away_goals_avg'] = 1.5
        enhanced_data['home_goals_conceded_avg'] = 1.5
        enhanced_data['away_goals_conceded_avg'] = 1.5
        enhanced_data['home_advantage'] = 1

        for i, match in enhanced_data.iterrows():
            home_team = match['home_team']
            away_team = match['away_team']

            home_history = self.get_team_history(enhanced_data, home_team, i, games=10)
            away_history = self.get_team_history(enhanced_data, away_team, i, games=10)

            home_stats = self.calculate_team_stats(home_history, home_team)
            away_stats = self.calculate_team_stats(away_history, away_team)

            enhanced_data.loc[i, 'home_team_strength'] = home_stats['strength']
            enhanced_data.loc[i, 'away_team_strength'] = away_stats['strength']
            enhanced_data.loc[i, 'home_recent_form'] = home_stats['form']
            enhanced_data.loc[i, 'away_recent_form'] = away_stats['form']
            enhanced_data.loc[i, 'home_goals_avg'] = home_stats['goals_for']
            enhanced_data.loc[i, 'away_goals_avg'] = away_stats['goals_for']
            enhanced_data.loc[i, 'home_goals_conceded_avg'] = home_stats['goals_against']
            enhanced_data.loc[i, 'away_goals_conceded_avg'] = away_stats['goals_against']

        print(f"Features calculated for {len(enhanced_data)} matches")
        return enhanced_data

    def get_team_history(self, data, team, current_match_index, games=5):
        team_matches = data[
            ((data['home_team'] == team) | (data['away_team'] == team)) &
            (data.index < current_match_index)
        ]
        return team_matches.tail(games)

    def calculate_team_stats(self, history, team):
        if len(history) == 0:
            return {
                'strength': 50,
                'form': 5,
                'goals_for': 1.5,
                'goals_against': 1.5
            }

        points = 0
        goals_scored = 0
        goals_conceded = 0

        for _, match in history.iterrows():
            if match['home_team'] == team:
                goals_scored += match['home_goals']
                goals_conceded += match['away_goals']
                if match['result'] == 'H':
                    points += 3
                elif match['result'] == 'D':
                    points += 1
            else:
                goals_scored += match['away_goals']
                goals_conceded += match['home_goals']
                if match['result'] == 'A':
                    points += 3
                elif match['result'] == 'D':
                    points += 1

        num_games = len(history)

        goals_per_game = goals_scored / num_games
        goals_conceded_per_game = goals_conceded / num_games

        strength = (points / num_games) * 20 + 20

        return {
            'strength': min(90, max(10, strength)),
            'form': points,
            'goals_for': goals_per_game,
            'goals_against': goals_conceded_per_game
        }

    def prepare_training_data(self, data):
        print("Preparing data for training...")

        feature_columns = [
            'home_team_strength',
            'away_team_strength',
            'home_recent_form',
            'away_recent_form',
            'home_goals_avg',
            # optional extras you already computed:
            # 'away_goals_avg',
            # 'home_goals_conceded_avg',
            # 'away_goals_conceded_avg',
            # 'home_advantage'
        ]

        X = data[feature_columns].copy()
        y_raw = data['result'].copy()

        # Fit encoder once on all labels present (H/D/A)
        y = self.label_encoder.fit_transform(y_raw)

        print(f"Training data shape: {X.shape}")
        print(f"Features: {feature_columns}")
        print(f"Classes: {list(self.label_encoder.classes_)}")

        return X, y, feature_columns

    def train_model(self):
        if self.match_data is None:
            print("No data loaded! Run get_football_data() first.")
            return False

        data_with_features = self.calculate_simple_features(self.match_data)
        training_data = data_with_features.iloc[50:].reset_index(drop=True)

        X, y, feature_columns = self.prepare_training_data(training_data)

        # Use a validation split for early stopping
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.2, random_state=41, stratify=y
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=41, stratify=y_temp
        )

        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        preds = self.model.predict(X_test)
        acc = accuracy_score(y_test, preds)

        print(f"Accuracy: {acc:.1%} (on {len(X_test)} test matches)")

        # Feature importance
        importance = self.model.feature_importances_
        feature_importances = pd.DataFrame({
            'Feature': feature_columns,
            'Importance': importance
        }).sort_values('Importance', ascending=False)

        print("Most important features:")
        for _, r in feature_importances.iterrows():
            print(f"  {r['Feature']}: {r['Importance']:.3f}")

        fig = px.bar(
            feature_importances,
            x='Importance',
            y='Feature',
            orientation='h',
            title="Feature Importance (XGBoost)",
            text=feature_importances['Importance'].round(3)
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(yaxis=dict(autorange="reversed"), template="plotly_white")
        fig.show()

        return True

    def predict_match(self, home_team, away_team):
        print(f"PREDICTING: {home_team} vs {away_team}")
        print("=" * 50)

        home_recent = self.match_data[
            (self.match_data['home_team'] == home_team) |
            (self.match_data['away_team'] == home_team)
            ].tail(10)

        away_recent = self.match_data[
            (self.match_data['home_team'] == away_team) |
            (self.match_data['away_team'] == away_team)
            ].tail(10)

        home_stats = self.calculate_team_stats(home_recent, home_team)
        away_stats = self.calculate_team_stats(away_recent, away_team)

        match_features = pd.DataFrame({
            'home_team_strength': [home_stats['strength']],
            'away_team_strength': [away_stats['strength']],
            'home_recent_form': [home_stats['form']],
            'away_recent_form': [away_stats['form']],
            'home_goals_avg': [home_stats['goals_for']]
        })

        pred_int = self.model.predict(match_features)[0]
        proba = self.model.predict_proba(match_features)[0]

        # Convert back to H/D/A labels
        pred_label = self.label_encoder.inverse_transform([pred_int])[0]
        class_labels = self.label_encoder.inverse_transform(np.arange(len(proba)))

        results = {'H': f'{home_team} Win', 'D': 'Draw', 'A': f'{away_team} Win'}

        print("Features being fed to model:")
        print(match_features)

        print(f"PREDICTION: {results[pred_label]}")
        print("Probabilities:")
        for lab, p in zip(class_labels, proba):
            print(f"  {results[lab]}: {p:.1%}")

        prob_df = pd.DataFrame({
            'Outcome': [results[lab] for lab in class_labels],
            'Probability': proba
        })

        fig = px.pie(
            prob_df,
            names='Outcome',
            values='Probability',
            title=f"Prediction Probabilities: {home_team} vs {away_team}"
        )
        fig.show()

        return pred_label, proba


def main():
    print("Premier League Match Prediction")
    print("=" * 60)

    predictor = PLPredictor()

    if not predictor.get_football_data():
        print("Couldn't get data. Stopping here.")
        return

    if not predictor.train_model():
        print("Couldn't train model. Stopping here.")
        return

    print("\nMATCH PREDICTION FOR TODAY")
    print("Match: Arsenal (Home) vs Chelsea (Away)")
    print("-" * 60)

    predictor.predict_match(home_team='Arsenal', away_team='Chelsea')
    return predictor


if __name__ == "__main__":
    model = main()