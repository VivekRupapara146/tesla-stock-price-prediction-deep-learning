# Tesla Stock Price Prediction — GRU Deep Learning App

A production-quality Streamlit web application that predicts Tesla's next-day closing price using a trained GRU (Gated Recurrent Unit) neural network.

---

## Model Performance

| Metric | Value |
|--------|-------|
| MAE    | 8.81 USD |
| RMSE   | 15.22 USD |
| MAPE   | 2.77% |
| R²     | 0.9603 |

---

## Project Structure

```
Tesla-Stock-Prediction/
│
├── app.py                   # Streamlit entry point (UI orchestration)
│
├── model/
│   ├── tesla_gru_model.keras  # Trained GRU model
│   └── tesla_scaler.pkl       # Fitted MinMaxScaler (training data only)
│
├── utils/
│   ├── data_loader.py         # yfinance fetch + CSV validation
│   ├── preprocessing.py       # Scaling + reshaping to (1, 60, 5)
│   ├── predictor.py           # Model inference + inverse transform
│   └── visualizer.py          # Plotly charts + metric cards
│
├── assets/                    # Static files (logo, CSS overrides)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/your-username/Tesla-Stock-Prediction.git
cd Tesla-Stock-Prediction
```

### 2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add model files
Place your trained artifacts inside the `model/` directory:
```
model/tesla_gru_model.keras
model/tesla_scaler.pkl
```

### 5. Run the app
```bash
streamlit run app.py
```

---

## Prediction Modes

### Mode A — Live Tesla Data
Fetches the latest OHLCV data from Yahoo Finance automatically.
Uses the most recent 60 trading days for prediction.

### Mode B — Upload CSV
Upload your own historical Tesla CSV file.
Required columns: `Open`, `High`, `Low`, `Close`, `Volume`
Minimum rows: 60

---

## Deployment — Streamlit Cloud

1. Push the repository to GitHub (exclude model files via `.gitignore`).
2. Upload `tesla_gru_model.keras` and `tesla_scaler.pkl` to a cloud bucket or use Git LFS.
3. Connect the repo at [share.streamlit.io](https://share.streamlit.io).
4. Set `app.py` as the entry point.

---

## Tech Stack

- **Model**: TensorFlow / Keras (GRU)
- **Data**: yfinance, pandas, NumPy
- **Scaling**: scikit-learn MinMaxScaler
- **App**: Streamlit
- **Charts**: Plotly
