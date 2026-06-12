print("Script is alive! Loading libraries...")


import firebase_admin
from firebase_admin import credentials, db
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import time

# 1. Firebase Setup
cred = credentials.Certificate("predictive-maintanece-firebase-adminsdk-fbsvc-1b52ba2157.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://predictive-maintanece-default-rtdb.asia-southeast1.firebasedatabase.app/'
})

# 2. Load ML Assets
model = load_model('autoencoder.h5', compile=False)
scaler = joblib.load('scaler.pkl')
with open('threshold.txt', 'r') as f:
    THRESHOLD = float(f.read())

features = ['temperature_data', 'Humidity', 'Rpm', 'sound_dB', 'Amplitude', 'Frequency']

print(f"📡 Starting Live ML Inference... (Threshold: {THRESHOLD:.4f})")

while True:
    ref = db.reference('sensor2')
    data = ref.get()

    if data:
        # Extract live values
        live_values = [
            data.get('temperature_data', 0),
            data.get('Humidity', 0),
            data.get('Rpm', 0),
            data.get('sound_dB', 0),
            data.get('Amplitude', 0),
            data.get('Frequency', 0)
        ]
        
        # Scale and Predict
        live_array = np.array([live_values])
        scaled_live = scaler.transform(live_array)
        reconstruction = model.predict(scaled_live, verbose=0)
        
        # Calculate Error
        mse = np.mean(np.power(scaled_live - reconstruction, 2), axis=1)[0]
        
        # Determine Status
        is_anomaly = bool(mse > THRESHOLD)
        root_cause = "None"

        if is_anomaly:
            # Find which sensor deviated the most
            individual_errors = np.abs(scaled_live[0] - reconstruction[0])
            worst_sensor_index = np.argmax(individual_errors)
            root_cause = features[worst_sensor_index]

        # Push ML results back to Firebase for the UI to read
        ref.update({
            'ML_Anomaly': is_anomaly,
            'ML_Error': float(mse),
            'ML_Root_Cause': root_cause
        })
        
        status_icon = "🔴" if is_anomaly else "🟢"
        print(f"{status_icon} MSE: {mse:.4f} | Anomaly: {is_anomaly} | Root Cause: {root_cause}")

    time.sleep(2)