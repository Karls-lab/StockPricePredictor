import yfinance as yf
import numpy as np
import pandas as pd
from datetime import date, timedelta
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

"""
The goal of this program is to based on features of the target stock,
predict if the stock will rise or fall in the next week.
My choices will be -1, -.5, 0, .5, 1 which represent the magnitude of the rise or fall
"""

# Get Apple Stock History
tkr = yf.Ticker('AAPL')
hist = tkr.history(period="10y")
print(f"YFinance Columns Obtained from TKR.History: {hist.columns}")
hist = hist.tz_localize(None) # fix weird date format

# Get the start and end date
end = date(2023, 6, 1)
start = end - timedelta(days=3650)

# Filter down to weekly trends 
hist = hist.resample('W').mean()

# filter to desired columns
hist = hist[['Close', 'Volume']]
print(f"Columns used: {hist.columns}")

# Save the dataframe to a CSV file
# hist.to_csv('savedData/stock_close_volume_rnn.csv')

# Create a new dataframe that will be used in training
df = pd.DataFrame(index=hist.index)

# Take the difference between two days
df['priceRise'] = hist['Close'].diff()
df['volumeRise'] = hist['Volume'].diff()

# Create an average of the price rise 
df['averagePrice'] = df['priceRise'].rolling(14).mean()

# Define the feature columns
features_columns = ['priceRise', 'volumeRise', 'averagePrice']

# Calculate the future price change of the next day
df['futurePriceChange'] = df['priceRise'].diff(periods=-1) * -1

# Remove any NaN 
df = df.dropna()

# Save the dataframe to a CSV file
# df.to_csv('savedData/stock_predictions_rnn.csv')


"""Start the RNN Training Process"""
# Display the dataframe with features and predictions
print(f"LENGTH OF DF: {len(df)}")
print(f"CLOSE AND VOLUME AVERAGE 7 DAYS WITH PREDICTIONS: \n{df.head(10)}")

# Round the dataframe to 3 decimal places and specify the data type
df = np.around(df, decimals=3)
df = df.astype('float32')

# Split into features and predicted changes
features = df[features_columns]
predicted_changes = df['futurePriceChange']

# Prepare for conversion to sequences
df_combined = pd.concat([features, predicted_changes], axis=1)
sequence_length = 10
num_features = len(features.columns)
X_sequences = []
y_sequences = []

# Iterate through the DataFrame to create sequences
for i in range(len(df_combined) - sequence_length + 1):
    # Extract a sequence of length 'sequence_length'
    sequence = df_combined[features_columns].iloc[i:i + sequence_length].values
    target = df_combined['futurePriceChange'].iloc[i + sequence_length - 1]  # Target is the prediction value at the end of the sequence

    X_sequences.append(sequence)
    y_sequences.append(target)

# Convert lists to NumPy arrays
X_sequences = np.array(X_sequences)
y_sequences = np.array(y_sequences)

# Build an RNN model
model = Sequential()
model.add(LSTM(
    128, 
    input_shape=(sequence_length, num_features),
    batch_size=32,
    dtype='float32',
    )) 
model.add(Dense(sequence_length, activation='linear'))  # Softmax activation for multiclass classification
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
