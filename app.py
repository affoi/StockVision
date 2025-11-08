
#Streamlit App: Real-Time Stock Predictor

import streamlit as st
import matplotlib.pyplot as plt
from model import predict_today_and_tomorrow

st.set_page_config(page_title="📈 StockVision - Dual Prediction", layout="wide")

st.title("📈 StockVision - Predict Today's & Tomorrow's Closing Prices")
st.markdown("""
This app predicts **today's** and **tomorrow's** stock closing prices using an ensemble of:
**Linear Regression**, **Random Forest**, **XGBoost**, and **LSTM**.
""")

ticker = st.text_input("Enter Stock Ticker (e.g. AAPL, TSLA, INFY.NS):", "AAPL")
period = st.selectbox("Select Historical Period:", ["1y", "3y", "5y", "10y"], index=2)

if st.button("Predict Now"):
    with st.spinner("Training models and forecasting..."):
        try:
            today_pred, tomorrow_pred, rmse, mae, r2, accuracy, preds = predict_today_and_tomorrow(ticker, period)

            # Display Results
            st.success(f"**Predicted Closing Price (Today):** ${today_pred:.2f}")
            st.success(f"**Predicted Closing Price (Tomorrow):** ${tomorrow_pred:.2f}")

            st.write("### Model Performance Metrics")
            st.metric("RMSE", f"{rmse:.2f}")
            st.metric("MAE", f"{mae:.2f}")
            st.metric("R²", f"{r2:.3f}")
            st.metric("Accuracy (%)", f"{accuracy:.2f}")

            # Plot Actual vs Predicted
            fig, ax = plt.subplots(figsize=(11, 6))
            ax.plot(preds.index, preds["actual"], label="Actual", color="black", linewidth=2)
            ax.plot(preds.index, preds["meta"], label="Predicted (Ensemble)", color="dodgerblue", linestyle="--", linewidth=2)
            ax.set_title(f"{ticker} - Actual vs Predicted Closing Prices", fontsize=13)
            ax.set_xlabel("Date")
            ax.set_ylabel("Price")
            ax.legend()
            ax.grid(True, alpha=0.4)
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Error: {e}")

st.markdown("---")
st.caption("Developed by team GARUDA")
