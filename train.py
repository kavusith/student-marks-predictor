# train.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import pickle

# Load dataset
df = pd.read_csv('Student_Marks.csv')

# Check columns
expected_features = ['number_courses', 'time_study']
target = 'Marks'

missing = set(expected_features + [target]) - set(df.columns)
if missing:
    raise ValueError(f"Missing columns in CSV: {missing}")

# Features and target
X = df[expected_features]
y = df[target]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)

# Train model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Evaluate
preds = model.predict(X_test)
mae = mean_absolute_error(y_test, preds)
print(f"Mean Absolute Error: {mae:.3f}")
print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

# Save model + metadata in Pickle
saved_data = {
    'model': model,
    'feature_cols': expected_features,
    'target_col': target
}

with open('student_marks.pkl', 'wb') as f:
    pickle.dump(saved_data, f)

print("Saved model to student_marks.pkl")
