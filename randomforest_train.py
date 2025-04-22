import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset
df = pd.read_csv("gesture_data.csv", header=None)

# Separate features and labels
X = df.iloc[:, :-1].values  # All columns except the last
y = df.iloc[:, -1].values   # Last column (gesture label)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, "gesture_model.pkl")
print("Model trained and saved as 'gesture_model.pkl'")

# Test accuracy
accuracy = model.score(X_test, y_test)
print(f"Model Accuracy: {accuracy:.2f}")
