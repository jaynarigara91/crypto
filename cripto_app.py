import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.models import Sequential, load_model
import matplotlib.pyplot as plt
import os

# List of cryptocurrencies you want to predict
crypto_currencies = ['BTC-USD', 'LTC-USD', 'ETH-USD', 'XMR-USD']  # Add other crypto pairs here
prediction_day = 60
model_save_path = "crypto_model.h5"  # Path to save the model


# Function to train and save the model
def train_and_save_model(crypto_currency):
    start = dt.datetime(2014, 1, 1)
    end = dt.datetime.now()

    # Fetch data using yfinance
    data = yf.download(crypto_currency, start=start, end=end)
    print(f"\nTraining model for {crypto_currency}:")
    print(data.tail())  

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

    # Prepare training data
    X_train, Y_train = [], []
    for x in range(prediction_day, len(scaled_data)):
        X_train.append(scaled_data[x - prediction_day:x, 0])
        Y_train.append(scaled_data[x, 0])

    X_train, Y_train = np.array(X_train), np.array(Y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, Y_train, epochs=25, batch_size=32)

    # Save the trained model to disk
    model.save(model_save_path)
    print(f"Model saved for {crypto_currency} at {model_save_path}")



# Function to load the model and make predictions
def predict_next_day(crypto_currency):
    # Load the saved model
    model_path = model_save_path
    if os.path.exists(model_path):
        model = tensorflow.keras.models.load_model(model_path)
        print(f"Loaded model from {model_path}")
    else:
        print("Model not found. Train the model first.")
        return

    # Fetch the latest data for prediction (the most recent `prediction_day` days)
    start = dt.datetime(2020, 1, 1)
    end = dt.datetime.now()
    data = yf.download(crypto_currency, start=start, end=end)
    
    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

    # Prepare the latest `prediction_day` days of data
    model_inputs = scaled_data[-prediction_day:]
    model_inputs = model_inputs.reshape(1, -1, 1)  # Reshape to match model input shape

    # Predict the next day's price
    prediction = model.predict(model_inputs)
    
    # Inverse the scaling transformation to get the actual predicted price
    predicted_price = scaler.inverse_transform(prediction)
    return predicted_price[0][0]

