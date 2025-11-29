import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# 1. Load Data
try:
    df_orig = pd.read_csv('healthcare_dataset.csv')
    df_anon = pd.read_csv('anonymized_augmented_dataset.csv')
except FileNotFoundError:
    print("Error: Files not found. Please ensure CSV files are in the same directory.")
    exit()

# 2. Preprocess Original Dataset
# Calculate 'Length of Stay' to match the concept in the anonymized dataset
df_orig['Date of Admission'] = pd.to_datetime(df_orig['Date of Admission'], dayfirst=True, errors='coerce')
df_orig['Discharge Date'] = pd.to_datetime(df_orig['Discharge Date'], dayfirst=True, errors='coerce')
df_orig['Length of Stay'] = (df_orig['Discharge Date'] - df_orig['Date of Admission']).dt.days

# Select features relevant for comparison
features_orig = ['Age', 'Gender', 'Medical Condition', 'Billing Amount', 'Admission Type', 'Length of Stay']
X_orig = df_orig[features_orig].dropna()
y_orig = df_orig.loc[X_orig.index, 'Test Results']

# Encode categorical columns
for col in X_orig.select_dtypes(include='object'):
    X_orig[col] = LabelEncoder().fit_transform(X_orig[col])

# 3. Preprocess Anonymized Dataset
features_anon = ['Age', 'Gender', 'Medical Condition', 'Billing Amount', 'Admission Type', 'Length of Stay']
X_anon = df_anon[features_anon].fillna('Unknown') # Handle missing values as a category
y_anon = df_anon['Test Results']

# Encode all columns (since anonymized data is bucketed/categorical)
for col in X_anon.columns:
    X_anon[col] = LabelEncoder().fit_transform(X_anon[col].astype(str))

# 4. Define Evaluation Function
def train_and_evaluate(X, y, dataset_name):
    # Split the data into variables ONCE
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    # Predict on the TEST set
    preds = clf.predict(X_test)
    
    # Calculate accuracy
    acc = accuracy_score(y_test, preds)
    print(f"{dataset_name} Accuracy: {acc:.2%}")

# 5. Run Comparison
print("--- Model Effectiveness Report ---")
train_and_evaluate(X_orig, y_orig, "Original Data")
train_and_evaluate(X_anon, y_anon, "Anonymized Data")