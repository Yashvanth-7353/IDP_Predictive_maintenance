import firebase_admin
from firebase_admin import credentials, db
import pandas as pd
import time
from datetime import datetime

# 1. Firebase Setup (Keep your existing credentials)
cred = credentials.Certificate("D:\\2nd sem EL\\Predictive_maintenance_4th_sem\\predictive-maintanece-firebase-adminsdk-fbsvc-1b52ba2157.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://predictive-maintanece-default-rtdb.asia-southeast1.firebasedatabase.app/'
})

csv_file = 'golden_baseline.csv'

# 2. Initialize Clean DataFrame (No Labels)
try:
    data_df = pd.read_csv(csv_file)
except FileNotFoundError:
    data_df = pd.DataFrame(columns=[
        'timestamp', 'temperature_data', 'Humidity', 'Rpm', 'sound_dB', 'Amplitude', 'Frequency'
    ])

print("🚀 Starting Golden Baseline Collection. Do NOT introduce faults. Press Ctrl+C to stop.")

try:
    while True:
        ref = db.reference('sensor2')
        sensor_data = ref.get()

        if sensor_data:
            # 3. Pure Raw Data Logging
            new_entry = {
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'temperature_data': sensor_data.get('temperature_data', 0),
                'Humidity': sensor_data.get('Humidity', 0),
                'Rpm': sensor_data.get('Rpm', 0),
                'sound_dB': sensor_data.get('sound_dB', 0),
                'Amplitude': sensor_data.get('Amplitude', 0),
                'Frequency': sensor_data.get('Frequency', 0)
            }

            data_df = pd.concat([data_df, pd.DataFrame([new_entry])], ignore_index=True)
            data_df.to_csv(csv_file, index=False)
            
            # 4. Show the logged data in the terminal
            print(f"[{new_entry['timestamp']}] ✅ Logged | Temp: {new_entry['temperature_data']}°C | Hum: {new_entry['Humidity']}% | RPM: {new_entry['Rpm']} | Sound: {new_entry['sound_dB']}dB | Amp: {new_entry['Amplitude']} | Freq: {new_entry['Frequency']}Hz")

        time.sleep(2) # Faster sampling rate for better time-series data

except KeyboardInterrupt:
    print("\n🛑 Baseline collection complete.")