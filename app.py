
from pathlib import Path

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


st.set_page_config(
    page_title="GridSense | Energy Intelligence",
    page_icon="⚡",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
RAW_PATH = DATA_DIR / "raw" / "PJME_hourly.csv"

FORECAST_PATH = PROCESSED_DIR / "forecast_test_results_v2.csv"
MONITORING_PATH = PROCESSED_DIR / "monitoring_results.csv"
ALERT_PATH = PROCESSED_DIR / "alert_history.csv"
ANOMALY_PATH = PROCESSED_DIR / "anomaly_detection_results.csv"


@st.cache_data
def load_csv(path, date_columns=None):
    if not path.exists():
        return None

    return pd.read_csv(
        path,
        parse_dates=date_columns or []
    )


st.title("⚡ GridSense")
st.caption(
    "Energy Demand Forecasting, Anomaly Screening "
    "& Forecast Error Monitoring"
)

st.info(
    "Historical PJM electricity-demand analysis. "
    "Alerts are analytical warnings, not confirmed grid incidents."
)

forecast = load_csv(
    FORECAST_PATH,
    ["Datetime"]
)

monitoring = load_csv(
    MONITORING_PATH,
    ["Datetime"]
)

alerts = load_csv(
    ALERT_PATH,
    ["Datetime"]
)

raw = load_csv(
    RAW_PATH,
    ["Datetime"]
)

anomalies = load_csv(
    ANOMALY_PATH,
    ["Datetime"]
)


if raw is None:
    st.error(f"Raw dataset not found: {RAW_PATH}")
    st.stop()

if forecast is None:
    st.error(
        "Forecast results not found. Please check "
        "forecast_test_results_v2.csv in data/processed."
    )
    st.stop()

required_forecast_columns = {
    "Datetime", "PJME_MW", "Forecast"
}

if not required_forecast_columns.issubset(forecast.columns):
    st.error(
        "Forecast CSV must contain Datetime, PJME_MW "
        "and Forecast columns."
    )
    st.stop()

raw = raw.dropna(
    subset=["Datetime", "PJME_MW"]
).sort_values("Datetime")

forecast = forecast.dropna(
    subset=["Datetime", "PJME_MW", "Forecast"]
).sort_values("Datetime")

if forecast.empty or raw.empty:
    st.error("The loaded datasets contain no usable rows.")
    st.stop()


st.sidebar.header("Dashboard Filters")

min_date = forecast["Datetime"].min().date()
max_date = forecast["Datetime"].max().date()

selected_dates = st.sidebar.date_input(
    "Forecast test period",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

# Streamlit can return one date or a tuple of two dates
if isinstance(selected_dates, (tuple, list)):
    if len(selected_dates) == 2:
        start_date, end_date = selected_dates
    elif len(selected_dates) == 1:
        start_date = end_date = selected_dates[0]
    else:
        start_date, end_date = min_date, max_date
else:
    start_date = end_date = selected_dates

filtered = forecast.loc[
    forecast["Datetime"].dt.date.between(start_date, end_date)
].copy()

if filtered.empty:
    st.warning(
        "No forecast data exists for this date range. "
        "Please select another date range."
    )
    st.stop()


mae = (filtered["PJME_MW"] - filtered["Forecast"]).abs().mean()

rmse = np.sqrt(
    ((filtered["PJME_MW"] - filtered["Forecast"]) ** 2).mean()
)

avg_demand = filtered["PJME_MW"].mean()
peak_demand = filtered["PJME_MW"].max()

c1, c2, c3, c4 = st.columns(4)

c1.metric("Average Demand", f"{avg_demand:,.0f} MW")
c2.metric("Peak Demand", f"{peak_demand:,.0f} MW")
c3.metric("Forecast MAE", f"{mae:,.0f} MW")
c4.metric("Forecast RMSE", f"{rmse:,.0f} MW")


overview_tab, forecast_tab, anomaly_tab, monitoring_tab = st.tabs(
    [
        "Demand Overview",
        "Forecast Performance",
        "Potential Anomalies",
        "Monitoring & Alerts"
    ]
)


with overview_tab:
    st.subheader("Historical Electricity Demand")

    fig = px.line(
        raw,
        x="Datetime",
        y="PJME_MW",
        title="Hourly PJM Electricity Consumption",
        labels={
            "Datetime": "Date",
            "PJME_MW": "Demand (MW)"
        }
    )

    fig.update_layout(hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    daily = (
        raw.set_index("Datetime")["PJME_MW"]
        .resample("D")
        .mean()
        .reset_index()
    )

    fig_daily = px.line(
        daily,
        x="Datetime",
        y="PJME_MW",
        title="Daily Average Demand",
        labels={"PJME_MW": "Average Demand (MW)"}
    )

    st.plotly_chart(fig_daily, use_container_width=True)



with forecast_tab:
    st.subheader("Actual vs Predicted Demand")

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=filtered["Datetime"],
        y=filtered["PJME_MW"],
        name="Actual Demand",
        mode="lines"
    ))

    fig.add_trace(go.Scatter(
        x=filtered["Datetime"],
        y=filtered["Forecast"],
        name="Random Forest Forecast",
        mode="lines"
    ))

    fig.update_layout(
        title="Actual vs Forecast",
        xaxis_title="Date",
        yaxis_title="Demand (MW)",
        hovermode="x unified"
    )

    st.plotly_chart(fig, use_container_width=True)

    if "Baseline_Prediction" in filtered.columns:
        baseline_mae = (
            filtered["PJME_MW"] -
            filtered["Baseline_Prediction"]
        ).abs().mean()

        comparison = pd.DataFrame({
            "Model": ["Persistence Baseline", "Random Forest V2"],
            "MAE (MW)": [baseline_mae, mae],
            "RMSE (MW)": [
                np.sqrt(
                    ((filtered["PJME_MW"] -
                      filtered["Baseline_Prediction"]) ** 2).mean()
                ),
                rmse
            ]
        })

        st.subheader("Model Comparison")
        st.dataframe(comparison, use_container_width=True)

        if baseline_mae > 0:
            improvement = (baseline_mae - mae) / baseline_mae * 100
            st.metric("MAE Reduction vs Baseline", f"{improvement:.1f}%")

    st.caption(
        "Evaluation uses one-step-ahead test predictions with "
        "historical actual demand available for lag features."
    )


with anomaly_tab:
    st.subheader("Potential Demand Anomalies")

    if anomalies is None:
        st.warning(
            "Anomaly results file is missing. "
            "Generate anomaly_detection_results.csv first."
        )
    else:
        st.write("Available columns:", ", ".join(anomalies.columns))

        anomaly_columns = [
            col for col in anomalies.columns
            if "anomaly" in col.lower() or "potential" in col.lower()
        ]

        if not anomaly_columns:
            st.warning(
                "No anomaly flag column was detected. "
                "Check the column names in the anomaly CSV."
            )
            st.dataframe(anomalies.head(20), use_container_width=True)
        else:
            flag_col = st.selectbox(
                "Select anomaly flag",
                anomaly_columns
            )

            flags = anomalies[flag_col].astype(str).str.lower().isin(
                ["true", "1", "yes"]
            )

            anomaly_view = anomalies.loc[flags].copy()

            st.metric("Flagged Observations", f"{len(anomaly_view):,}")

            st.caption(
                "These are statistical candidates, not confirmed "
                "events or proven causes."
            )

            st.dataframe(
                anomaly_view.head(500),
                use_container_width=True
            )

            st.download_button(
                "Download Potential Anomalies",
                data=anomaly_view.to_csv(index=False).encode("utf-8"),
                file_name="potential_anomalies.csv",
                mime="text/csv"
            )


with monitoring_tab:
    st.subheader("Forecast Error Monitoring")

    if monitoring is None:
        st.warning(
            "Monitoring file is missing. "
            "Generate monitoring_results.csv first."
        )
    else:
        if "Rolling_MAE_24h" in monitoring.columns:
            fig = px.line(
                monitoring,
                x="Datetime",
                y="Rolling_MAE_24h",
                title="Rolling Forecast Error"
            )

            if "Alert" in monitoring.columns:
                alert_mask = (
                    monitoring["Alert"].astype(str).str.lower()
                    .isin(["true", "1", "yes"])
                )

                flagged = monitoring.loc[alert_mask]

                fig.add_trace(go.Scatter(
                    x=flagged["Datetime"],
                    y=flagged["Rolling_MAE_24h"],
                    mode="markers",
                    name="Warning observations"
                ))

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning(
                "Rolling_MAE_24h column is missing in monitoring results."
            )

        if alerts is not None:
            st.subheader("Alert History")

            st.metric("Alert Records", f"{len(alerts):,}")

            st.dataframe(
                alerts.head(500),
                use_container_width=True
            )

            st.download_button(
                "Download Alert History",
                data=alerts.to_csv(index=False).encode("utf-8"),
                file_name="alert_history.csv",
                mime="text/csv"
            )
        else:
            st.warning("Alert history file is missing.")


st.divider()

st.caption(
    "GridSense Portfolio Project | Historical data analysis | "
    "Not a real-time grid control or operational alerting system"
)
