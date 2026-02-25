import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import requests
import warnings
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings('ignore')


class PLPredictor:
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )
        self.models = {}
        self.match_data = None

    def get_football_data(self):
        print("Downloading Premier League data...")
        season_urls = {
            '2022-23': 'https://www.football-data.co.uk/mmz4281/2223/E0.csv',
            '2023-24': 'https://www.football-data.co.uk/mmz4281/2324/E0.csv',
            '2024-25': 'https://www.football-data.co.uk/mmz4281/2425/E0.csv'
        }

        all_matches = []

        for season, url in season_urls.items():
            try:
                print(f"  Getting {season} season...")
                response = requests.get(url, timeout=10)
                from io import StringIO
                season_data = pd.read_csv(StringIO(response.text))
                clean_data = self.clean_season_data(season_data, season)
                all_matches.append(clean_data)
                print(f"Got {len(clean_data)} matches from {season}")

            except Exception as e:
                print(f"Couldn't get {season} data: {e}")

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

        # Initialize engineered feature columns as floats to allow fractional updates.
        enhanced_data['home_team_strength'] = 50.0
        enhanced_data['away_team_strength'] = 50.0
        enhanced_data['home_recent_form'] = 5.0
        enhanced_data['away_recent_form'] = 5.0
        enhanced_data['home_goals_avg'] = 1.5
        enhanced_data['away_goals_avg'] = 1.5
        enhanced_data['home_goals_conceded_avg'] = 1.5
        enhanced_data['away_goals_conceded_avg'] = 1.5
        enhanced_data['home_advantage'] = 1.0

        for i, match in enhanced_data.iterrows():
            home_team = match['home_team']
            away_team = match['away_team']

            home_history = self.get_team_history(enhanced_data, home_team, i, games=5)
            away_history = self.get_team_history(enhanced_data, away_team, i, games=5)

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
            'home_goals_avg'
        ]

        X = data[feature_columns].copy()
        y = data['result']

        print(f"Training data shape: {X.shape}")
        print(f"Features: {feature_columns}")

        return X, y

    def train_model(self):
        if self.match_data is None:
            print("No data loaded! Run get_football_data() first.")
            return False

        data_with_features = self.calculate_simple_features(self.match_data)
        training_data = data_with_features.iloc[50:].reset_index(drop=True)

        X, y = self.prepare_training_data(training_data)

        # Encode labels to ints for consistency across models (required by XGBoost multi-class).
        class_labels = ['H', 'D', 'A']
        label_to_int = {lbl: idx for idx, lbl in enumerate(class_labels)}
        int_to_label = {idx: lbl for lbl, idx in label_to_int.items()}

        y_encoded = y.map(label_to_int)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=41, stratify=y_encoded
        )

        print("Training models...")

        model_defs = {
            'RandomForest': RandomForestClassifier(n_estimators=200, random_state=42),
            # multi_class omitted for compatibility with older sklearn versions.
            'LogisticRegression': LogisticRegression(max_iter=2000),
            'KNN': KNeighborsClassifier(n_neighbors=15, weights='distance'),
            'XGBoost': XGBClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=5,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=42,
                objective='multi:softprob',
                num_class=len(class_labels)
            )
        }

        self.models = {}
        accuracies = {}

        for name, model in model_defs.items():
            print(f"  Fitting {name}...")
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            acc = accuracy_score(y_test, preds)
            accuracies[name] = acc
            self.models[name] = model
            print(f"    {name} accuracy: {acc:.1%}")

        # keep RandomForest as default for text output
        self.model = self.models.get('RandomForest', next(iter(self.models.values())))

        # store label mappings for prediction phase
        self.label_to_int = label_to_int
        self.int_to_label = int_to_label
        self.class_labels = class_labels

        acc_df = pd.DataFrame({
            'Model': list(accuracies.keys()),
            'Accuracy': list(accuracies.values())
        }).sort_values('Accuracy', ascending=False)

        fig_acc = px.bar(
            acc_df,
            x='Model',
            y='Accuracy',
            title="Model Accuracy Comparison",
            text=acc_df['Accuracy'].map(lambda x: f"{x:.1%}")
        )
        fig_acc.update_traces(textposition="outside")
        fig_acc.update_layout(yaxis_tickformat=".0%", template="plotly_white")
        fig_acc.show()

        return True

    def predict_match(self, home_team, away_team):
        print(f"PREDICTING: {home_team} vs {away_team}")
        print("=" * 50)

        if not self.models:
            print("Models not trained. Run train_model() first.")
            return None, None

        recent_matches = self.match_data.tail(100)

        home_recent = recent_matches[
            (recent_matches['home_team'] == home_team) |
            (recent_matches['away_team'] == home_team)
        ].tail(5)

        away_recent = recent_matches[
            (recent_matches['home_team'] == away_team) |
            (recent_matches['away_team'] == away_team)
        ].tail(5)

        home_stats = self.calculate_team_stats(home_recent, home_team)
        away_stats = self.calculate_team_stats(away_recent, away_team)

        match_features = pd.DataFrame({
            'home_team_strength': [home_stats['strength']],
            'away_team_strength': [away_stats['strength']],
            'home_recent_form': [home_stats['form']],
            'away_recent_form': [away_stats['form']],
            'home_goals_avg': [home_stats['goals_for']]
        })

        results_label = {'H': f'{home_team} Win', 'D': 'Draw', 'A': f'{away_team} Win'}

        # Text summary per model
        for name, model in self.models.items():
            pred_raw = model.predict(match_features)[0]
            classes = model.classes_

            # Map integer classes back to labels if needed
            if self.int_to_label and isinstance(pred_raw, (int, np.integer)):
                prediction = self.int_to_label[pred_raw]
                class_labels = [self.int_to_label[c] if isinstance(c, (int, np.integer)) else c for c in classes]
            else:
                prediction = pred_raw
                class_labels = classes

            probabilities = model.predict_proba(match_features)[0]

            print(f"{name} -> {results_label[prediction]}")
            for i, outcome in enumerate(class_labels):
                prob = probabilities[i]
                print(f"    {results_label[outcome]}: {prob:.1%}")

        # Grouped pie chart (one pie per model)
        num_models = len(self.models)
        fig = make_subplots(
            rows=1,
            cols=num_models,
            specs=[[{'type': 'domain'}] * num_models],
            subplot_titles=list(self.models.keys())
        )

        for idx, (name, model) in enumerate(self.models.items(), start=1):
            probs = model.predict_proba(match_features)[0]
            classes = model.classes_
            outcomes = [
                results_label[self.int_to_label[c] if isinstance(c, (int, np.integer)) else c]
                for c in classes
            ]
            fig.add_trace(
                go.Pie(labels=outcomes, values=probs, name=name, hole=0.25),
                row=1, col=idx
            )

        fig.update_traces(textinfo='label+percent')
        fig.update_layout(
            title_text=f"Prediction Probabilities: {home_team} vs {away_team}",
            template="plotly_white"
        )
        fig.show()

        # Return the default model's prediction for compatibility
        default_pred_raw = self.model.predict(match_features)[0]
        default_pred = self.int_to_label.get(default_pred_raw, default_pred_raw)
        default_probs = self.model.predict_proba(match_features)[0]
        return default_pred, default_probs


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
    print("Match: Leeds (Home) vs Liverpool (Away)")
    print("-" * 60)

    prediction, probabilities = predictor.predict_match(
        home_team='Leeds',
        away_team='Liverpool'
    )

    return predictor


if __name__ == "__main__":
    model = main()
