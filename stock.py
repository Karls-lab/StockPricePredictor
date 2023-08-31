import yfinance as yf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import pandas_datareader.data as pdr
from datetime import date, timedelta

"""
The goal of this program is to based on features of the target stock,
predict if the stock will rise or fall in the next week.
My choices will be -1, -.5, 0, .5, 1 which represent the magnitude of the rise or fall
Ideas:
    Average out the price and volume over the last 7 days and use that as a feature

"""

# Get Apple Stock History
tkr = yf.Ticker('AAPL')
hist = tkr.history(period="1y")
print(f"Columns: {hist.columns}")
hist = hist.tz_localize(None) # fix weird date format

# Get the start and end date
end = date.today()
start = end - timedelta(days=365)

df = hist

# filter to desired columns
df = df[['Close', 'Volume']]
print(f"Columns used: {df.columns}")

# Take the difference between two days and apply the log function
df['priceRise'] = np.log(df['Close'] / df['Close'].shift(1))
df['volumeRise'] = np.log(df['Volume'] / df['Volume'].shift(1))
df = df.dropna()

# Average out the features over the last 7 days
df = df.resample('W').mean()
print(f"CLOSE AND VOLUME AVERAGE 7 DAYS: \n{df}")

# Features for the Model
df = df[['priceRise', 'volumeRise']]

# Calculate conditions based on the price rise/fall
conditions = [
    (df['priceRise'].shift(-1) < -0.02), # -2 percent fall
    (df['priceRise'].shift(-1) < -0.01), # -1 percent fall
    (df['priceRise'].shift(-1) >= -0.01) & (df['priceRise'].shift(-1) <= 0.01), # no change
    (df['priceRise'].shift(-1) > 0.01), # 1 percent rise
    (df['priceRise'].shift(-1) > 0.02)  # 2 percent rise
]

choices = [-2, -1, 0, 1, 2]
df['Pred'] = np.select(conditions, choices, default=0)

features = df[['priceRise', 'volumeRise']].to_numpy()
features = np.around(features, decimals=2)
target = df['Pred'].to_numpy()
# print(features) # from this data
# print(target)   # determine if it will become -1, 0, or 1

# Train my model
x_train, y_train, y_train, y_test = train_test_split(features, target, test_size=0.2)
mlClassifier = MLPClassifier(
    hidden_layer_sizes=(100, 100),
    activation='relu',
    solver='adam',
    max_iter=1000,
    random_state=1
)
mlClassifier.fit(x_train, y_train)
print(f"Score: {mlClassifier.score(x_train, y_train)}")