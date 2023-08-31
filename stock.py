import yfinance as yf
import numpy as np
import pandas as pd
from datetime import date, timedelta
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

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

# Create a new dataframe that will be used in training
features = ['priceRise', 'volumeRise']
df = pd.DataFrame(index=hist.index, columns=features)

# Take the difference between two days and apply the log function
df['priceRise'] = np.log(hist['Close'] / hist['Close'].shift(1))
df['volumeRise'] = np.log(hist['Volume'] / hist['Volume'].shift(1))

# Average out the features over the last 7 days
df = df.resample('W').mean()

# Refactor the df to only include engineered features
df = df[['priceRise', 'volumeRise']]

# Calculate conditions based on the price rise/fall
conditions = [
    (df['priceRise'].shift(-1) <= -0.01), # -2 percent fall
    (df['priceRise'].shift(-1) <= -0.005) & (df['priceRise'].shift(-1) > -0.01), # -.5 percent fall
    (df['priceRise'].shift(-1) >  -0.005) & (df['priceRise'].shift(-1) <  0.005), # no change
    (df['priceRise'].shift(-1) >=  0.005) & (df['priceRise'].shift(-1) <  0.01), # .5 percent rise
    (df['priceRise'].shift(-1) >=  0.01)  # 2 percent rise
]

# Create a price prediction column based on percent price rise and fall
choices = [-2, -1, 0, 1, 2]
df['Pred'] = np.select(conditions, choices, default=0)

# Display the dataframe with features and predictions
print(f"CLOSE AND VOLUME AVERAGE 7 DAYS WITH PREDICTIONS: \n{df.head(20)}")

# Display the percentage of each prediction
print(f"Percentage of each prediction: \n{df['Pred'].value_counts(normalize=True)}")

# Finalize features and 
features = df[['priceRise', 'volumeRise']].to_numpy()
features = np.around(features, decimals=2)
predicted_changes = df['Pred'].to_numpy()

# Train my model
x_train, y_train, y_train, y_test = train_test_split(features, predicted_changes, test_size=0.2)
mlClassifier = MLPClassifier(
    hidden_layer_sizes=(100, 100, 5),
    activation='relu',
    solver='adam',
    max_iter=1000,
    random_state=1
)
mlClassifier.fit(x_train, y_train)
print(f"Score: {mlClassifier.score(x_train, y_train)}\n\n")

print("From the data posted in the past week, will the stock rise or fall?")
print(df.tail(1))