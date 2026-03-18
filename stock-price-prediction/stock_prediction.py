import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Download stock data (Apple stock)
data = yf.download("AAPL", start="2020-01-01", end="2024-01-01")

# Select features
X = data[['Open','High','Low']]
y = data['Close']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict prices
predictions = model.predict(X_test)

# Plot result
plt.plot(y_test.values, label="Actual Price")
plt.plot(predictions, label="Predicted Price")

plt.title("Stock Price Prediction")
plt.legend()
plt.show()