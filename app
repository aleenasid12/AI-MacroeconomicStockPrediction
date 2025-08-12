import streamlit as st
from model import StockMarketPredictor  

st.title("📈 Stock Market Predictor")
st.write("Predict if stocks will go up or down tomorrow!")

# Get user inputs
api_key = st.text_input("Enter your FRED API key:", type="password")
stock = st.selectbox("Pick a stock:", ["SPY", "AAPL", "MSFT"])

# Button to make prediction
if st.button("🔮 Make Prediction"):
    if api_key:
        
        predictor = StockMarketPredictor(api_key, stock, '2020-01-01')
        models, data = predictor.run_full_analysis()
        
        # Show the prediction
        direction, return_pred, prob = predictor.make_predictions()
        
        if direction == 1:
            st.success(f"📈 Prediction: Stock will go UP!")
        else:
            st.error(f"📉 Prediction: Stock will go DOWN!")
            
        st.write(f"Expected return: {return_pred:.2%}")
    else:
        st.warning("Please enter your API key first!")