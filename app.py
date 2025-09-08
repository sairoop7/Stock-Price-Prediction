import streamlit as st
import pandas as pd
from gru_model import StockPredictor

st.title("📈 Stock Price Prediction (Next 7 Days)")

st.write("""
Enter the ticker symbol of a company to get its 7-day stock price prediction.  
Here are some examples you can try:  
- Microsoft → `MSFT`  
- Apple → `AAPL`  
- Amazon → `AMZN`  
- Alphabet (Google) → `GOOGL` or `GOOG`  
- Meta Platforms → `META`  
- Broadcom → `AVGO`  
- Tesla → `TSLA`  
- Berkshire Hathaway → `BRK.A` or `BRK.B`  
""")

ticker = st.text_input("Enter Stock Ticker Symbol:", "MSFT")

if st.button("Predict"):
    predictor = StockPredictor(streamlit_mode=True)  # ✅ Run in streamlit mode
    predictor.run(ticker)

    if predictor.future_predictions is not None:
        st.subheader(f"📊 Next 7-Day Predictions for {ticker}")

        # Format with 2 decimals
        table = predictor.future_predictions[['Close']].rename(
            columns={"Close": "Predicted Closing Price"}
        ).round(2)
        st.write(table)

        # Add line chart
        st.line_chart(table)
