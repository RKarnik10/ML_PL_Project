# ⚽ Premier League Match Predictor

> An XGBoost-powered match prediction engine for the English Premier League — built with real historical data, dual form windows, and a live Streamlit dashboard.

---

## 🧠 The Problem

Predicting football results is hard. Bookmakers employ teams of analysts and still get it wrong. Most hobbyist models use overly simple features — season-long averages that ignore momentum — or leak future data into training, making accuracy look better than it really is.

We built something better.

---

## 💡 Our Solution

A machine learning pipeline that:
- Trains on **4 seasons of Premier League data** (2022–2026)
- Uses **dual rolling form windows** (5-game short-term + 15-game long-term) to capture both momentum and underlying quality
- Applies **season-aware weighting** so last season's form informs but doesn't distort current predictions
- Uses a **chronological train/val/test split** — no data leakage, no inflated accuracy
- Surfaces everything through a clean, interactive **Streamlit dashboard**

---

## 🚀 Demo

```bash
# 1. Clone the repo
git clone https://github.com/RKarnik10/-ML_PL_Project.git
cd ML_PL_Project

# 2. Install dependencies
pip install streamlit xgboost scikit-learn plotly pandas

# 3. Run the app
streamlit run app.py
```

> The app opens at `http://localhost:8501`

---

## 🖥️ App Features

| Feature | Description |
|---|---|
| **Team selector** | Pick any two PL teams from dropdowns |
| **Live prediction** | Win/Draw/Loss probabilities with colour-coded result banner |
| **Form comparison table** | Short & long-term stats side by side for both teams |
| **Feature importance chart** | See which signals the model weights most heavily |
| **Configurable settings** | Tune form windows and season weights from the sidebar |
| **Training log** | Transparent view of data loaded and model accuracy |

---

## 🏗️ Architecture

```
ML_PL_Project/
│
├── app.py                  # Streamlit frontend + UI logic
├── xg_pl_predictor_v5.py   # Core model (standalone, importable)
│
├── PL_2223.csv             # Season data (football-data.co.uk)
├── PL_2324.csv
├── PL_2425.csv
├── PL_2526.csv
│
└── README.md
```

### Tech Stack

- **Model** — XGBoost (multi-class classification: Home / Draw / Away)
- **Feature engineering** — Pandas rolling windows, vectorised (no per-row loops)
- **Frontend** — Streamlit + Plotly
- **Data** — [football-data.co.uk](https://www.football-data.co.uk)

---

## 📊 How the Model Works

### Features (25 total)

Each match is represented by rolling stats computed **strictly from prior matches** — no lookahead bias.

| Group | Features |
|---|---|
| Short form (last 5 games) | Strength, form points, goals for/against — home & away |
| Long form (last 15 games) | Same as above, capturing underlying quality |
| Difference features | Home minus away for each stat — gives the model relative signals |
| Context | Home advantage flag |

### Training

- **Algorithm**: XGBoost with `multi:softprob` objective
- **Split**: Chronological 70% train / 15% val / 15% test
- **Recency weighting**: Exponential sample weights (~2.7x for newest vs oldest matches)
- **Early stopping**: Prevents overfitting via validation loss monitoring
- **Prior season discount**: Older seasons weighted at 0.4x to avoid stale signal

---

## 📈 Results

| Metric | Value |
|---|---|
| Test Accuracy | ~52–55% |
| Baseline (always predict home win) | ~45% |
| Classes | Home Win / Draw / Away Win |

> Football is inherently unpredictable — a model beating the always-home-win baseline consistently is genuinely useful. Bookmaker models typically sit around 55–60% with far more features.

---

## 🔮 What's Next

- **Bookmaker odds as features** — the CSVs already contain Bet365 odds (`B365H/D/A`), which are highly predictive and not yet used
- **Home/away split stats** — separate rolling windows for home performance vs away performance
- **Scoreline prediction** — Poisson model to predict exact scores, not just outcomes
- **Node.js + FastAPI** — rebuild the frontend in Next.js with the Python model exposed as a REST API
- **Auto-updating data** — scheduled scraping so the model stays current without manual CSV updates
- **Claude API integration** — AI-generated match previews based on team stats

---

## 👥 Team

| Name | Role |
|---|---|
| Your Name | ML Model, Data Pipeline |
| Teammate | Frontend, Streamlit UI |
| Teammate | Research, Pitch |

---

## 📂 Data Source

All match data from [football-data.co.uk](https://www.football-data.co.uk/englandm.php) — free, reliable, updated weekly during the season.

---

## 📜 License

MIT — free to use, fork, and improve.
