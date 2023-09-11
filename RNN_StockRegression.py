import yfinance as yf
import numpy as np
import pandas as pd
from datetime import date, timedelta
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import tensorflow as tf
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

"""
The goal of this program is to based on features of the target stock,
predict if the stock will rise or fall in the next week.
My choices will be -1, -.5, 0, .5, 1 which represent the magnitude of the rise or fall
"""

# Get Apple Stock History
tkr = yf.Ticker('AAPL')
hist = tkr.history(period="5y")
print(f"YFinance Columns Obtained from TKR.History: {hist.columns}")
hist = hist.tz_localize(None) # fix weird date format

# Get the start and end date
end = date.today()
start = end - timedelta(days=1825)

# filter to desired columns
hist = hist[['Close', 'Volume']]
print(f"Columns used: {hist.columns}")

# Save the dataframe to a CSV file
hist.to_csv('savedData/stock_close_volume_rnn.csv')

# Create a new dataframe that will be used in training
features = ['priceRise', 'volumeRise']
df = pd.DataFrame(index=hist.index, columns=features)

# Take the difference between two days and apply the log function
df['priceRise'] = np.log(hist['Close'] / hist['Close'].shift(1))
df['volumeRise'] = np.log(hist['Volume'] / hist['Volume'].shift(1))

# Refactor the df to only include engineered features
df = df[['priceRise', 'volumeRise']]

# Calculate the percent change in price
df['priceChange'] = df['priceRise'].diff().shift(-1)

# Remove any NaN 
df = df.dropna()

# Save the dataframe to a CSV file
df.to_csv('savedData/stock_normalized_rnn.csv')

# Display the dataframe with features and predictions
print(f"LENGTH OF DF: {len(df)}")
print(f"CLOSE AND VOLUME AVERAGE 7 DAYS WITH PREDICTIONS: \n{df.head(20)}")

# Finalize features
features = df[['priceRise', 'volumeRise']]
features = np.around(features, decimals=2)
features = features.astype('float32')

# Finalize predictions
predicted_changes = df['priceChange']
predicted_changes = np.around(predicted_changes, decimals=3)
predicted_changes = predicted_changes.astype('float32')

# Prepare for conversion to sequences
df_combined = pd.concat([features, predicted_changes], axis=1)
sequence_length = 10
X_sequences = []
y_sequences = []

# Iterate through the DataFrame to create sequences
for i in range(len(df_combined) - sequence_length + 1):
    # Extract a sequence of length 'sequence_length'
    sequence = df_combined[['priceRise', 'volumeRise']].iloc[i:i + sequence_length].values
    target = df_combined['priceChange'].iloc[i + sequence_length - 1]  # Target is the prediction value at the end of the sequence

    X_sequences.append(sequence)
    y_sequences.append(target)

# Convert lists to NumPy arrays
X_sequences = np.array(X_sequences)
y_sequences = np.array(y_sequences)

# Build an RNN model
model = Sequential()
model.add(LSTM(
    128, 
    input_shape=(10, 2),
    batch_size=10,
    dtype='float32')) 
model.add(Dense(10, activation='linear'))  # Softmax activation for multiclass classification
model.compile(loss='mean_squared_error', optimizer='adam')  # Use categorical crossentropy for classification
model.summary()


# Split data into training (80%) and validation (20%)
X_train, X_val, y_train, y_val = train_test_split(X_sequences, y_sequences, test_size=0.2, random_state=42)

# Train the model using X_train and y_train, and validate on X_val and y_val
model.fit(X_train, y_train, epochs=5, validation_data=(X_val, y_val))

# View means squared error
print(f"Mean Squared Error: {model.evaluate(X_sequences, y_sequences)}")
print(f"Mean Squared Error on Validation Set: {model.evaluate(X_val, y_val)}")

# Save the model
model.save('savedModels/stock_rnn.keras')

# Plot the model compared to actual price changes
