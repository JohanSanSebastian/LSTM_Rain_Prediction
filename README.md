
# Smart Rain Prediction on Farm21 Test Node

Made as part of the final project for my S4 course - **Python Programming and Application**, this project uses the ESP32-based integrated farm21 test board (https://github.com/JohanSanSebastian/farm21_v1_board) to log temperature and humidity, then trains an LSTM model to forecast future readings, and then uses logistic regression with historical weather data (obtained from a Open-Meteo API) to estimate the probability of rain at each time step. 

*Tested by running system out my hostel room window over a period of 10 days.*

## Features

- Automatic CSV logging of timestamped temperature and humidity readings from the microcontroller through serial interface.
- LSTM-based multivariate time‑series model to predict future temperature and humidity. 
- Historical weather fetch via Open‑Meteo API to build a rain/no‑rain dataset. 
- Logistic regression model that converts temperature and humidity into rain probabilities. 
- Combined visualization showing actual vs predicted temperature, actual vs predicted humidity, and predicted rain probability over time. 


## Data Pipeline

1. **Acquisition**  
   - ESP32 reads temperature and humidity via DHT22 and prints comma‑separated values over serial (e.g., `temp,hum`). 
   - `data_log.py` parses each serial line, attaches a current timestamp, and appends it as a new row to `humidity_data.csv` until the user stops the script.

2. **Pre‑processing**  
   - `main.py` loads `humidity_data.csv` and extracts temperature and humidity as a two‑feature array.
   - A `MinMaxScaler` normalizes all features to \[0, 1\] and a helper function creates sliding windows of length 10 steps (sequence length) to use as LSTM inputs.

3. **LSTM Training and Forecasting**  
   - The dataset is split into 80% training and 20% testing without shuffling to respect temporal order.
   - A Sequential Keras model with two LSTM layers, dropout regularization, and a dense output layer of size 2 (temperature and humidity) is trained using mean squared error loss and Adam optimizer, with early stopping on validation loss. 

4. **Historical Rain Modelling**  
   - Historical hourly weather data (temperature, humidity, precipitation) is fetched from Open‑Meteo for a window spanning roughly 30 days before the earliest local sample to the last sample timestamp. 
   - A binary rain label is created by checking whether the precipitation value is greater than zero, and a logistic regression model (`max_iter=1000`) is fitted on temperature and humidity to predict rain vs no rain. 

5. **Rain Probability Forecasting**  
   - The trained logistic regression model is evaluated on a held‑out portion of the historical dataset and then applied to the LSTM’s predictions. 
   - The `predict_proba` output gives the probability of rain at each step, which is plotted as a filled curve for easier visual interpretation of risk over time. 


## Extensions being worked on beyond coursework

- Integrate additional sensors like soil moisture and light intensity to improve rain prediction fidelity.
- Deploy the trained LSTM and logistic regression models to run directly on the ESP32 using TinyML approaches for on‑device inference.
- Add a simple Flask or FastAPI dashboard to serve live plots and rain risk alerts over a web interface.