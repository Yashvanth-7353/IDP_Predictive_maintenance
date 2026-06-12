import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
import os

# 1. Load the Golden Baseline
csv_file = 'golden_baseline.csv'
data = pd.read_csv(csv_file)

# Select only the sensor columns
features = ['temperature_data', 'Humidity', 'Rpm', 'sound_dB', 'Amplitude', 'Frequency']
X = data[features].values

# 2. Scale the Data (Crucial for Neural Networks)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Build the Autoencoder
# It compresses 6 features down to 3, then attempts to reconstruct them back to 6.
model = Sequential([
    Input(shape=(6,)),
    Dense(6, activation='relu'),
    Dense(3, activation='relu'), # The "bottleneck"
    Dense(6, activation='linear')
])

model.compile(optimizer='adam', loss='mse')

print("🧠 Training the Autoencoder on normal data...")
model.fit(X_scaled, X_scaled, epochs=50, batch_size=32, validation_split=0.1, verbose=1)

# 4. Calculate the "Anomaly Threshold"
# We find the maximum error the model makes on normal data. Anything above this is an anomaly.
reconstructions = model.predict(X_scaled)
mse = np.mean(np.power(X_scaled - reconstructions, 2), axis=1)
threshold = np.max(mse) * 1.2 # Adding a 20% buffer to prevent false alarms

print(f"\n✅ Training Complete.")
print(f"📍 Calculated Anomaly Threshold: {threshold:.4f}")

# 5. Save the Model, Scaler, and Threshold
model.save('autoencoder.h5')
joblib.dump(scaler, 'scaler.pkl')
with open('threshold.txt', 'w') as f:
    f.write(str(threshold))