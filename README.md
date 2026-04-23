# ⚽ Premier League Match Predictor

> An XGBoost-powered match prediction engine for the English Premier League — built with real historical data, dual form windows, rolling market signals, and a live Streamlit dashboard.

---

## 🧠 The Problem

Predicting football results is hard. Bookmakers employ teams of analysts and still get it wrong. Most hobbyist models use overly simple features — season-long averages that ignore momentum — or leak future data into training, making accuracy look better than it really is.

We built something better.

---

## 💡 Our Solution

A machine learning pipeline that:
- Trains on **4 seasons of Premier League data** (2022–2026), **auto-downloaded** from football-data.co.uk — no manual CSV management
- Uses **dual rolling form windows** (5-game short-term + 15-game long-term) to capture both momentum and underlying quality
- Incorporates **shots on target**, **market-implied win probabilities**, and **Asian handicap lines** — all derived from historical data, no external API needed
- Applies **season-aware weighting** so last season's form informs but doesn't distort current predictions
- Uses a **chronological train/val/test split** — no data leakage, no inflated accuracy
- Surfaces everything through a clean, interactive **Streamlit dashboard**

---

## 🚀 Quickstart

```bash
# 1. Clone the repo
git clone https://github.com/RKarnik10/-ML_PL_Project.git
cd ML_PL_Project

# 2. Install dependencies
pip install streamlit xgboost scikit-learn plotly pandas requests

# 3. Run the app
streamlit run app.py
```

> Opens at `http://localhost:8501`. Data is fetched automatically on startup — no CSV files required.

---

## 🖥️ App Features

| Feature | Description |
|---|---|
| **Auto data refresh** | Fetches latest season data from football-data.co.uk on load, cached for 1 hour |
| **🔄 Refresh button** | Force re-download after a new matchweek |
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
├── xg_pl_predictor_v5.py   # Core model (standalone, runnable directly)
│
├── PL_2223.csv             # Optional local fallback (auto-fetched if absent)
├── PL_2324.csv
├── PL_2425.csv
├── PL_2526.csv
│
└── README.md
```

### Tech Stack

| Layer | Technology |
|---|---|
| Model | XGBoost (`multi:softprob`) |
| Feature engineering | Pandas rolling windows, fully vectorised |
| Frontend | Streamlit + Plotly |
| Data | [football-data.co.uk](https://www.football-data.co.uk) (auto-fetched via `requests`) |

---

## 📊 How the Model Works

### Features (41 total)

Each match is represented by rolling stats computed **strictly from prior matches** — no lookahead bias.

| Group | Features |
|---|---|
| Short form (last 5 games) | Strength, form pts, goals for/against, shots on target, market win prob, AH line — home & away |
| Long form (last 15 games) | Same as above, capturing underlying quality over a larger window |
| Difference features | Home minus away for each stat — gives the model relative signals |
| Context | Home advantage flag |

**Market features** (`mkt`, `ah`) are rolling averages of implied win probabilities and Asian handicap lines from historical match odds in the dataset — no external odds input needed at prediction time.

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

- **Home/away split stats** — separate rolling windows for home performance vs away performance
- **Scoreline prediction** — Poisson model to predict exact scores, not just outcomes
- **Node.js + FastAPI** — rebuild the frontend in Next.js with the Python model exposed as a REST API
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

All match data from [football-data.co.uk](https://www.football-data.co.uk/englandm.php) — free, reliable, updated after every matchweek.

---

## 📜 License

MIT — free to use, fork, and improve.
