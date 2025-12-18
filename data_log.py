# Importing libraries required to interact with serial communication
import serial
import csv
import time
from datetime import datetime

# Initializing the serial comm terminal with baud rate 9600
arduino = serial.Serial('/dev/ttyUSB0', 9600)
time.sleep(2)

filename = 'humidity_data.csv'

# Setting the header rows
with open(filename, 'a', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['time', 'temperature', 'humidity'])

print("Logging data")

# Until interrupted by user, the code runs
try:
    while True:
        line = arduino.readline().decode().strip()
        if line:
            try:
                # Logs the data into the CSV (Data is only sent once an hour as the Arduino has a sleep function of 3600000 ms)
                temp, hum = line.split(',')
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                with open(filename, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([timestamp, temp, hum])
            except:
                pass
except KeyboardInterrupt:
    print("\nLogging stopped.")
    arduino.close()
