import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from gru_model import StockPredictor  # import your GRU predictor class

# Streamlit UI
st.title("📈 Stock Price Prediction with GRU")
st.markdown("This app predicts stock prices using a Bidirectional GRU model.")

# User input
ticker = st.text_input("Enter stock ticker (e.g., AAPL, MSFT, TSLA):", "AAPL")

if st.button("Predict"):
    # Run prediction
    predictor = StockPredictor()
    predictor.run(ticker)

    # Show predictions
    st.subheader("Predicted Close Prices for Next 7 Days")
    st.dataframe(predictor.future_predictions[['Close']])

    # Plot results
    st.subheader("📊 Historical vs Predicted Prices")
    fig, ax = plt.subplots(figsize=(10, 5))
    predictor.df['Close'].tail(200).plot(ax=ax, label="Historical Close")
    predictor.future_predictions['Close'].plot(ax=ax, label="Predicted Close", color="green")
    ax.legend()
    st.pyplot(fig)
