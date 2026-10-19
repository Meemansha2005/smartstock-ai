
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="SmartStock AI",
    page_icon="📦",
    layout="wide",
)

DATA_PATH = Path("data/train.csv")
MODEL_PATH = Path("models/demand_model.joblib")


@st.cache_data
def load_sales_data():
    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["sales"] = pd.to_numeric(df["sales"], errors="coerce")
    df = df.dropna(subset=["date", "store", "item", "sales"])
    return df.sort_values(["store", "item", "date"])


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def forecast_next_seven_days(df, bundle, store_id, item_id):
    history = df[
        (df["store"] == store_id) & (df["item"] == item_id)
    ].sort_values("date").copy()

    if history.empty:
        return pd.DataFrame()

    history = history.set_index("date")["sales"].resample("D").sum()
    history = history.asfreq("D", fill_value=0)

    model = bundle["model"]
    features = bundle["feature_columns"]
    forecast_rows = []

    for _ in range(7):
        next_date = history.index.max() + pd.Timedelta(days=1)

        row = {
            "store": store_id,
            "item": item_id,
            "lag_1": float(history.iloc[-1]),
            "lag_7": float(history.iloc[-7]) if len(history) >= 7 else 0.0,
            "lag_14": float(history.iloc[-14]) if len(history) >= 14 else 0.0,
            "day_of_week": next_date.dayofweek,
            "day_of_month": next_date.day,
            "month": next_date.month,
        }

        prediction = max(0.0, float(model.predict(
            pd.DataFrame([row])[features]
        )[0]))

        forecast_rows.append({
            "date": next_date,
            "predicted_sales": round(prediction, 2),
        })

        history.loc[next_date] = prediction

    return pd.DataFrame(forecast_rows)


st.title("📦 SmartStock AI")
st.caption("Demand Forecasting & Inventory Optimization")
st.markdown(
    "Analyze historical sales, forecast the next seven days, "
    "and estimate when stock may need replenishment."
)

if not DATA_PATH.exists():
    st.error("Dataset not found. Put train.csv inside the data folder.")
    st.stop()

if not MODEL_PATH.exists():
    st.error("Trained model not found. Run `python train_model.py` first.")
    st.stop()

df = load_sales_data()
bundle = load_model()

# Sidebar filters
st.sidebar.header("Forecast Settings")

store_options = sorted(df["store"].unique().tolist())
item_options = sorted(df["item"].unique().tolist())

store_id = st.sidebar.selectbox("Select store", store_options)
item_id = st.sidebar.selectbox("Select product", item_options)

# Main KPIs
total_sales = df["sales"].sum()
average_daily_sales = df.groupby("date")["sales"].sum().mean()

col1, col2, col3 = st.columns(3)
col1.metric("Total Historical Sales", f"{total_sales:,.0f} units")
col2.metric("Average Daily Sales", f"{average_daily_sales:,.0f} units")
col3.metric("Products", f"{df['item'].nunique():,}")

st.divider()

# Historical sales trends
st.subheader("📈 Historical Sales Trends")

daily_sales = (
    df.groupby("date", as_index=False)["sales"]
    .sum()
    .sort_values("date")
)

trend_fig = px.line(
    daily_sales,
    x="date",
    y="sales",
    title="Total Sales Over Time",
    labels={"date": "Date", "sales": "Units Sold"},
)
st.plotly_chart(trend_fig, use_container_width=True)

top_items = (
    df.groupby("item", as_index=False)["sales"]
    .sum()
    .sort_values("sales", ascending=False)
    .head(10)
)

top_fig = px.bar(
    top_items,
    x="item",
    y="sales",
    title="Top 10 Products by Total Sales",
    labels={"item": "Product ID", "sales": "Total Units Sold"},
)
st.plotly_chart(top_fig, use_container_width=True)

st.divider()

# Forecasting
st.subheader("🔮 Seven-Day Demand Forecast")

st.write(f"Forecast for **Store {store_id} — Product {item_id}**")

forecast_df = forecast_next_seven_days(
    df, bundle, store_id, item_id
)

if forecast_df.empty:
    st.warning("No sales history found for this selection.")
else:
    forecast_fig = px.line(
        forecast_df,
        x="date",
        y="predicted_sales",
        markers=True,
        title="Predicted Daily Demand",
        labels={
            "date": "Date",
            "predicted_sales": "Predicted Units",
        },
    )
    st.plotly_chart(forecast_fig, use_container_width=True)
    st.dataframe(forecast_df, use_container_width=True, hide_index=True)

    st.download_button(
        "⬇️ Download Forecast CSV",
        data=forecast_df.to_csv(index=False).encode("utf-8"),
        file_name=f"forecast_store_{store_id}_item_{item_id}.csv",
        mime="text/csv",
    )

    st.divider()

    # Inventory recommendation
    st.subheader("📦 Inventory Reorder Recommendation")

    c1, c2, c3 = st.columns(3)

    with c1:
        current_stock = st.number_input(
            "Current stock (units)",
            min_value=0,
            value=50,
            step=1,
        )

    with c2:
        lead_time = st.number_input(
            "Supplier lead time (days)",
            min_value=1,
            max_value=365,
            value=3,
            step=1,
        )

    with c3:
        safety_stock = st.number_input(
            "Safety stock (units)",
            min_value=0,
            value=10,
            step=1,
        )

    average_forecast = forecast_df["predicted_sales"].mean()
    lead_time_demand = average_forecast * lead_time
    reorder_point = lead_time_demand + safety_stock
    target_stock = lead_time_demand + safety_stock
    order_quantity = max(0, int(np.ceil(target_stock - current_stock)))

    r1, r2, r3 = st.columns(3)
    r1.metric("Estimated Lead-Time Demand", f"{lead_time_demand:.1f} units")
    r2.metric("Reorder Point", f"{reorder_point:.1f} units")
    r3.metric("Suggested Order Quantity", f"{order_quantity} units")

    if current_stock <= reorder_point:
        st.warning(
            "Reorder recommended based on the forecast and your stock inputs."
        )
    else:
        st.success("Current stock is above the estimated reorder point.")

    st.caption(
        "Inventory recommendations are estimates based on predicted demand, "
        "supplier lead time, and user-entered safety stock. "
        "They are not a guarantee of future sales."
    )

st.divider()

st.subheader("ℹ️ Model Information")
st.write(f"**Random Forest test MAE:** {bundle['mae']:.3f} units")
st.write(f"**Previous-day baseline MAE:** {bundle['baseline_mae']:.3f} units")
st.caption(
    "MAE is the mean absolute error on the model's held-out test period. "
    "A lower MAE generally indicates more accurate predictions."
)

st.caption("SmartStock AI | Academic project prototype")
