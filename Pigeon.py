import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Simulate or load data
# Replace this simulated data with actual pigeon race data
data = {
    'race_distance': np.random.uniform(5, 500, 1000),  # Distance in kilometers
    'speed': np.random.uniform(10, 60, 1000),         # Speed in km/h
    'weather_condition': np.random.choice([0, 1], 1000),  # 0 for bad, 1 for good
    'training_hours': np.random.uniform(10, 100, 1000),   # Training hours
    'is_lost': np.random.choice([0, 1], 1000)             # 0 for not lost, 1 for lost
}
df = pd.DataFrame(data)

# Step 2: Add Calculus-related Features (e.g., Position and Velocity)
df['time'] = df['race_distance'] / df['speed']  # Time = Distance / Speed
df['position'] = df['speed'] * df['time']       # Position approximation
df['acceleration'] = df['speed'] / df['time']  # Simplified acceleration estimate

# Step 3: Prepare data for training
features = ['race_distance', 'speed', 'weather_condition', 'training_hours', 'time', 'position', 'acceleration']
X = df[features]
y = df['is_lost']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train the Predictor Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 5: Make Predictions and Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Step 6: Feature Importance (Optional for understanding the model)
importances = model.feature_importances_
print("Feature Importances:")
for feature, importance in zip(features, importances):
    print(f"{feature}: {importance:.3f}")

# Save the Model for Future Use
import joblib
joblib.dump(model, 'pigeon_race_predictor.pkl')

# notes
# the data dictionary stimulates the necessary features (distance, speed, weather, training hours, etc.)
# Calculus features: time, position, and acceleration 
# Training model: RandomF