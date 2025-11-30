# train.py -- robust version that auto-detects columns and prints helpful diagnostics
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import pickle
import sys

CSV = 'Student_Marks.csv'

# Try load CSV
try:
    df = pd.read_csv(CSV)
except FileNotFoundError:
    print(f"ERROR: Could not find '{CSV}' in the current folder.")
    sys.exit(1)
except Exception as e:
    print("ERROR reading CSV:", e)
    sys.exit(1)

print("Loaded CSV. Columns found:")
print(list(df.columns))
print("\nFirst 5 rows:")
print(df.head().to_string(index=False))

# Candidate names for the target column (case-insensitive)
target_candidates = ['marks', 'mark', 'score', 'total', 'grades', 'grade']

# find target column (case-insensitive match)
found_target = None
for col in df.columns:
    if col.strip().lower() in target_candidates:
        found_target = col
        break

# If we didn't find automatically, check for exact 'Marks'
if found_target is None and 'Marks' in df.columns:
    found_target = 'Marks'

# If still not found, prompt clear error with suggestions
if found_target is None:
    print("\nERROR: Could not automatically find the target column (Marks).")
    print("Please either:")
    print("  1) Rename the target column in your CSV to one of: Marks, marks, Mark, Score, Total, grade, etc.")
    print("  2) Or set 'target_col' variable below to the exact column name in your CSV.")
    print("\nDetected columns were:", list(df.columns))
    # helpful hint: show numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    print("Numeric columns detected (possible features or target):", numeric_cols)
    sys.exit(1)

target_col = found_target
print(f"\nUsing target column: '{target_col}'")

# Optionally you can explicitly set feature_cols here if you know them:
# feature_cols = ['Courses', 'StudyTime']   # <-- uncomment and edit if you want fixed names
feature_cols = None

# If feature_cols not set, auto-select numeric columns except the target
if not feature_cols:
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    feature_cols = [c for c in numeric_cols if c != target_col]
    if not feature_cols:
        # Try columns that look numeric but stored as object
        coerced = df.apply(pd.to_numeric, errors='coerce')
        numeric_guess = [c for c in coerced.columns if coerced[c].notna().sum() > 0]
        feature_cols = [c for c in numeric_guess if c != target_col]
    print(f"Auto-detected feature columns: {feature_cols}")

# Final validation
missing = set(feature_cols + [target_col]) - set(df.columns)
if missing:
    raise ValueError(f"Missing columns in CSV: {missing}. Update CSV column names or set feature_cols manually in this script.")

# Prepare data
X = df[feature_cols].apply(pd.to_numeric, errors='coerce')
y = pd.to_numeric(df[target_col], errors='coerce')

# Drop rows with NaNs
data = pd.concat([X, y], axis=1).dropna()
X = data[feature_cols]
y = data[target_col]

if X.shape[0] < 5:
    print("WARNING: After dropping NaN rows you have fewer than 5 samples. Model may not train well.")

# Split
x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)

# Train model
r = RandomForestRegressor(random_state=42)
r.fit(x_train, y_train)

# Evaluate
preds = r.predict(x_test)
mae = mean_absolute_error(y_test, preds)
print(f"\nDone training. Mean Absolute Error on test set: {mae:.3f}")
print(f"Training samples: {x_train.shape[0]}, Test samples: {x_test.shape[0]}")

# Save model and the feature order
out = {'model': r, 'feature_cols': feature_cols, 'target_col': target_col}
with open('student_marks.pkl', 'wb') as f:
    pickle.dump(out, f)

print("\nSaved model to student_marks.pkl")
print("You can now run your Streamlit app (app.py) which should use the same feature columns.")

