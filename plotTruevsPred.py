import matplotlib.pyplot as plt

# Assuming 'actual_prices' and 'predicted_prices' are your arrays or DataFrames
# Make sure they have the same length and are in the same order.

# Create a time index (e.g., days, dates) if applicable
time_index = range(len(actual_prices))  # Replace with your time index data if available

# Plot actual prices in blue
plt.plot(time_index, actual_prices, label='Actual Prices', color='blue')

# Plot predicted prices in red
plt.plot(time_index, predicted_prices, label='Predicted Prices', color='red')

# Add labels, title, and legend
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.title('Stock Price Prediction vs Actual')
plt.legend()

# Show the plot
plt.show()
