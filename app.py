from cripto_app import predict_next_day,train_and_save_model
import streamlit as st


crypto_currencies = ['BTC-USD', 'LTC-USD', 'ETH-USD', 'XMR-USD']
selected_crypto = st.selectbox('Select Cryptocurrency', crypto_currencies)


if st.button('Predict Next Day Price'):
    # Predict the next day's price for the selected cryptocurrency
    st.write('Model Is train on Latest data wait for 3-4 mins')
    train_and_save_model(selected_crypto)
    predicted_price = predict_next_day(selected_crypto)
    
    if predicted_price:
        st.success(f"The predicted price for {selected_crypto} on the next day is: ${predicted_price:.2f}")