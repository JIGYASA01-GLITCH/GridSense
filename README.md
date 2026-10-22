<div align="center">
⚡ GridSense
Energy Demand Forecasting · Anomaly Screening · Model Monitoring
From historical electricity demand to actionable time-series insights.
A machine-learning dashboard for exploring PJM electricity demand, comparing forecasts, screening unusual patterns, and monitoring prediction errors.
</div>
---
📊 At a glance
Random Forest MAE	Random Forest RMSE	MAE improvement	Potential seasonal anomalies
331 MW	458 MW	69%	6,048
<sub>Forecast metrics are from one-step-ahead test predictions. Anomaly counts are statistical candidates, not confirmed grid incidents.</sub>
👀 Dashboard preview
The dashboard brings the workflow into one place:
Demand Overview — explore hourly demand and daily averages.
Forecast Performance — compare actual demand, Random Forest forecasts, and a persistence baseline.
Potential Anomalies — inspect and download observations flagged by statistical detectors.
Monitoring & Alerts — track rolling forecast error and review warning records.
> **Built with:** Python · pandas · scikit-learn · Streamlit · Plotly · Jupyter
🚀 What makes it useful?
Forecast, then compare: Random Forest V2 achieved a test MAE of 331.36 MW, compared with 1,070.15 MW for the persistence baseline.
Look beyond one detector: rolling median/MAD and seasonality-aware screening identify different candidate observations.
Monitor model errors: rolling MAE and threshold-based warnings make larger-error periods easier to inspect.
Explore interactively: filter the forecast test period, inspect charts and tables, and download available results.
Generate a 24-hour forecast: a recursive forecast is saved for the 24 hours after the dataset's final timestamp.
🧭 How it works
```text
Historical PJME demand
        ↓
Data quality & seasonal exploration
        ↓
Anomaly screening ── Forecast model
                         ↓
                 Evaluation vs baseline
                         ↓
                 Error monitoring
                         ↓
                 Streamlit dashboard
```
⚡ Run it locally
1. Clone your GitHub repository
```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd GridSense
```
2. Create an environment and install packages
```bash
python -m venv .venv
```
Windows PowerShell:
```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
3. Add the dataset
Place `PJME_hourly.csv` here:
```text
data/raw/PJME_hourly.csv
```
4. Launch the dashboard
```bash
streamlit run app.py
```
Open the local URL shown in your terminal (usually `http://localhost:8501`).
🗂️ Project layout
```text
GridSense/
├── app.py
├── requirements.txt
├── data/
│   ├── raw/
│   └── processed/
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
├── reports/
├── src/
└── tests/
```
📈 Key results
Evaluation	Result
Persistence baseline MAE	1,070.15 MW
Random Forest V2 MAE	331.36 MW
Random Forest V2 RMSE	458.25 MW
MAE reduction vs. baseline	69.0%
Rolling median/MAD candidates	5,061
Seasonality-aware candidates	6,048
Monitoring warning records	774
⚠️ Important context
The dataset is historical PJM East demand data, ending 3 August 2018; this is not live grid data.
The 24-hour forecast is a historical backdated forecast for 3–4 August 2018, not a forecast for today.
One-step-ahead test metrics do not validate the accuracy of the recursive 24-hour forecast.
Anomaly flags are candidates for investigation; there are no ground-truth incident labels to establish detection accuracy.
Monitoring warnings are threshold-based records, not necessarily distinct incidents.
Timestamp gaps and duplicates were identified; daylight-saving-time ambiguities have not been fully resolved.
---
<div align="center">
GridSense · A time-series ML and analytics portfolio project
Historical analysis only — not an operational grid-control or safety system.
</div>
