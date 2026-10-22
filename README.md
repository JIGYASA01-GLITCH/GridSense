# ⚡ GridSense

**Energy Demand Forecasting · Anomaly Detection · Model Monitoring**

A time-series machine learning project that analyzes historical electricity demand, compares forecasting models, identifies unusual demand patterns, and monitors prediction errors through an interactive Streamlit dashboard.

## 📊 Key Results

| Metric | Result |
|---|---:|
| Random Forest V2 MAE | 331.36 MW |
| Random Forest V2 RMSE | 458.25 MW |
| Persistence Baseline MAE | 1,070.15 MW |
| MAE Improvement | 69% |
| Seasonality-Aware Anomaly Candidates | 6,048 |

*Forecast metrics are from one-step-ahead test predictions. Anomaly candidates are not confirmed grid incidents.*

## 🚀 Dashboard Features

- **Demand Overview** — Explore historical electricity demand.
- **Forecast Performance** — Compare actual demand, Random Forest predictions, and a persistence baseline.
- **Potential Anomalies** — Inspect statistically unusual demand observations.
- **Monitoring & Alerts** — Review rolling prediction errors and threshold-based warnings.

## 🛠️ Tech Stack

Python · Pandas · NumPy · Scikit-learn · Streamlit · Plotly · Jupyter Notebook

## ⚙️ Run Locally

**1. Clone the repository**

```bash
git clone https://github.com/JIGYASA01-GLITCH/GridSense.git
cd GridSense
```

**2. Create and activate a virtual environment**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Add the dataset**

Place `PJME_hourly.csv` inside:

```text
data/raw/PJME_hourly.csv
```

**5. Launch the dashboard**

```bash
streamlit run app.py
```

## 📁 Project Structure

```text
GridSense/
├── app.py
├── requirements.txt
├── notebook/
│   ├── 01_data_quality.ipynb
│   ├── 02_eda_seasonality.ipynb
│   ├── 03_anomaly_detection.ipynb
│   ├── 04_demand_forecasting.ipynb
│   ├── 05_model_evaluation.ipynb
│   ├── 06_monitoring.ipynb
│   ├── 07_forecasting_lab.ipynb
│   ├── 08_anomaly_validation.ipynb
│   └── 09_future_demand_forecast.ipynb
└── reports/
```

## ⚠️ Limitations

- Uses historical PJM East electricity demand data ending on **3 August 2018**; it is not live grid data.
- The 24-hour forecast is historical, not a forecast for the present day.
- Anomaly flags are candidates for investigation; ground-truth incident labels are unavailable.
- One-step-ahead evaluation metrics do not establish the accuracy of the recursive 24-hour forecast.

---

*GridSense — A time-series machine learning and analytics portfolio project.*
