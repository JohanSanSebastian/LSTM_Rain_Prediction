# Importing essential libraries - ML and data processing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from datetime import datetime, timedelta

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Loading data from CSV files
print("Reading local data")
df = pd.read_csv('humidity_data.csv')
assert {'temperature', 'humidity'}.issubset(df.columns), "CSV must have 'temperature' and 'humidity' columns"

data = df[['temperature', 'humidity']].values

# Normalizing data to (0,1) range
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Creating sequences of 10 elements for the LSTM
def create_sequences(arr, sequence_length=10):
    X, y = [], []
    for i in range(len(arr) - sequence_length):
        X.append(arr[i:i + sequence_length])
        y.append(arr[i + sequence_length])
    return np.array(X), np.array(y)

SEQUENCE_LENGTH = 10
X, y = create_sequences(scaled_data, SEQUENCE_LENGTH)

# Splitting the data as 80% training data and 20% testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# Building and compiling the model
model = Sequential([
    LSTM(50, activation='relu', return_sequences=True, input_shape=(SEQUENCE_LENGTH, 2)),
    Dropout(0.2),
    LSTM(50, activation='relu'),
    Dropout(0.2),
    Dense(2)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    epochs=50, batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)

# Predicting using the LSTM
predictions = model.predict(X_test, verbose=0)
predictions_actual = scaler.inverse_transform(predictions)
y_test_actual = scaler.inverse_transform(y_test)

# Fetching historical data from Open Mateo's API using a function
def fetch_historical_weather(lat, lon, start_date, end_date):
    # Uses the API endpoint to make an API call
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        'latitude': lat,
        'longitude': lon,
        'start_date': start_date,
        'end_date': end_date,
        'hourly': 'temperature_2m,relative_humidity_2m,precipitation',
        'timezone': 'auto'
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    hourly = data['hourly']
    df_hist = pd.DataFrame({
        'timestamp': pd.to_datetime(hourly['time']),
        'temperature': hourly['temperature_2m'],
        'humidity': hourly['relative_humidity_2m'],
        'precipitation': hourly['precipitation']
    })
    df_hist['rain'] = (df_hist['precipitation'] > 0).astype(int)
    return df_hist

# Setting up location and time constraints
LATITUDE = 28.6139
LONGITUDE = 77.2090

df['time'] = pd.to_datetime(df['time'])

# Extracting start and end dates
start_date = (df['time'].min() - pd.Timedelta(days=30)).strftime('%Y-%m-%d')
end_date = (df['time'].max()).strftime('%Y-%m-%d')

print("Fetching historical weather")
df_weather = fetch_historical_weather(LATITUDE, LONGITUDE, start_date, end_date)
print(f"Historical rows: {len(df_weather)}")

# Running logistic regression to predict rain

# Setting X with the value of hum and temp
# Setting Y with value of precipitation
X_rain = df_weather[['temperature', 'humidity']].values
y_rain = df_weather['rain'].values

# Splitting 80% of X
split_idx = int(0.8 * len(X_rain))

# Setting training vs testing data (for X and Y)
X_train_rain, X_test_rain = X_rain[:split_idx], X_rain[split_idx:]
y_train_rain, y_test_rain = y_rain[:split_idx], y_rain[split_idx:]

# Setting the model with the same X and Y values
rain_model = LogisticRegression(max_iter=1000)
rain_model.fit(X_train_rain, y_train_rain)
rain_probs_test = rain_model.predict_proba(X_test_rain)[:, 1]

# Running the forecast
rain_forecast = rain_model.predict_proba(predictions_actual)[:, 1]

# Plotting the data obtained
fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

# Temperature
axes[0].plot(y_test_actual[:, 0], label='Actual Temperature', linewidth=2)
axes[0].plot(predictions_actual[:, 0], label='Predicted Temperature', linewidth=2, alpha=0.7)
axes[0].set_ylabel('Temp (°C)')
axes[0].set_title('Temperature, Humidity, and Rain Probability')
axes[0].grid(True, alpha=0.3)
axes[0].legend()

# Humidity
axes[1].plot(y_test_actual[:, 1], label='Actual Humidity', linewidth=2)
axes[1].plot(predictions_actual[:, 1], label='Predicted Humidity', linewidth=2, alpha=0.7)
axes[1].set_ylabel('Humidity (%)')
axes[1].grid(True, alpha=0.3)
axes[1].legend()

# Rain probability
axes[2].plot(rain_forecast, label='Predicted Rain', linewidth=2, color='blue')
axes[2].fill_between(range(len(rain_forecast)), rain_forecast, alpha=0.25, color='blue')
axes[2].set_xlabel('Time Steps')
axes[2].set_ylabel('Probability')
axes[2].grid(True, alpha=0.3)
axes[2].legend()

fig.tight_layout()
fig.savefig('combined_forecasts.png', dpi=300)
plt.show()
