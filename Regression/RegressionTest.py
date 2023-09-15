import numpy as np
import pandas as pd
from datetime import date, timedelta
from sklearn.model_selection import train_test_split
from keras.models import load_model 
import os 
from tensorflow.keras.layers import Dense, Conv1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import yfinance as yf


# Get Apple Stock History
tkr = yf.Ticker('GOOGL')
hist = tkr.history(period="2y")
print(f"YFinance Columns Obtained from TKR.History: {hist.columns}")
hist = hist.tz_localize(None) # fix weird date format

# Get the start and end date
end = date(2023, 6, 1)
start = end - timedelta(days=84)
hist = hist.loc[start:end]

# filter to desired columns
hist = hist[['Close', 'Volume']]

# Calculate weekly trends
hist = hist.resample('W').mean()

# Create price rise and volume rise features in a new df
df = pd.DataFrame(index=hist.index)
df['priceRise'] = hist['Close'].diff()
df['volumeRise'] = hist['Volume'].diff()  

# Calculate the future price change of the next day
df['futurePriceChange'] = df['priceRise'].diff(periods=-1) * -1

# Remove any NaN
df = df.dropna()

print(df.info())
print(df)

# get the second to last row
prediction_features = df.iloc[0:len(df)-1]
priseRise = prediction_features['priceRise']
volumeRise = prediction_features['volumeRise']

print(f"\nPrice Rise: {priseRise}")

# Create Sequences for the LSTM
sequence_length = 10
x_df = df.iloc[:-1][['priceRise', 'volumeRise']]
X_sequences = []
for i in range(len(x_df) - sequence_length + 1):
    # Extract a sequence of length 'sequence_length'
    sequence = x_df[x_df.columns].iloc[i:i + sequence_length].values
    X_sequences.append(sequence)

# print(f"X Sequences: {X_sequences}")
# print(f"X Sequences shape: {np.array(X_sequences).shape}")

# Load the saved model 
""" 
The model expects a shape of (10, 10, 2) 
This means that the model expects 10 sequences of length 10, 
and 2 features (priceRise, volumeRise)
"""
load_model = load_model(os.path.abspath('savedModels/stock_rnn.h5'))
print(f"model shape: {load_model.input_shape}")

# Make a prediction
print(np.array([X_sequences[0]]))
prediction = load_model.predict(np.array([X_sequences[0]])).tolist()[0][0]
print(f"Future prise rise/fall Prediction in the next 7 days: {prediction}")
print(f"Actual price rise/fall: {df['futurePriceChange'][-1]}")