import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
from model import predict_today_and_tomorrow

# Set page configuration
st.set_page_config(page_title="📈 StockVision - BATCH-2", layout="wide")

# Function to set background image
def set_background(image_file):
    with open(image_file, "rb") as f:
        data = f.read()
    import base64
    encoded = base64.b64encode(data).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Apply background
set_background("light_stock_background.png")

# App title and description
st.title("📈 StockVision - Predict Today's & Tomorrow's Closing Prices")
st.markdown("""
This app predicts *today's* and *tomorrow's* stock closing prices using an ensemble of:
**Linear Regression**, **Random Forest**, **XGBoost**, and **LSTM** models.
""")

# User input
ticker = st.text_input("Enter Stock Ticker (e.g. AAPL, TSLA, INFY.NS):", "AAPL")
period = st.selectbox("Select Historical Period:", ["1y", "3y", "5y", "10y"], index=2)

# Prediction button
if st.button("Predict Now"):
    with st.spinner("Training models and forecasting..."):
        try:
            today_pred, tomorrow_pred, rmse, mae, r2, accuracy, preds = predict_today_and_tomorrow(ticker, period)

            # Display predictions
            st.success(f"**Predicted Closing Price (Today):** ${today_pred:.2f}")
            st.success(f"**Predicted Closing Price (Tomorrow):** ${tomorrow_pred:.2f}")

            # Display metrics side-by-side
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("RMSE", f"{rmse:.2f}")
            col2.metric("MAE", f"{mae:.2f}")
            col3.metric("R²", f"{r2:.3f}")
            col4.metric("Accuracy (%)", f"{accuracy:.2f}")

            # Plot actual vs predicted prices
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

# Footer
st.markdown("---")
st.caption("Developed by BATCH2")
                                 