import pandas as pd

# 1. Load Data
try:
    df_orig = pd.read_csv('healthcare_dataset.csv')
    df_anon = pd.read_csv('anonymized_augmented_dataset.csv')
except FileNotFoundError:
    print("Error: Files not found.")
    exit()

# 2. Preprocess Original (Calculate derived fields to match Anonymized schema)
# We use the raw, specific values for the original dataset to show vulnerability
df_orig['Date of Admission'] = pd.to_datetime(df_orig['Date of Admission'], dayfirst=True, errors='coerce')
df_orig['Discharge Date'] = pd.to_datetime(df_orig['Discharge Date'], dayfirst=True, errors='coerce')
df_orig['Length of Stay'] = (df_orig['Discharge Date'] - df_orig['Date of Admission']).dt.days

# Define Quasi-Identifiers (QIs) and Sensitive Attribute
# QIs: Attributes an attacker might know (Age, Gender, Billing, etc.)
qis_orig = ['Age', 'Gender', 'Insurance Provider', 'Billing Amount', 'Admission Type', 'Length of Stay']
qis_anon = ['Age', 'Gender', 'Insurance Provider', 'Billing Amount', 'Admission Type', 'Length of Stay']
sensitive_col = 'Medical Condition'

# 3. Privacy Evaluation Function
def get_privacy_metrics(df, qis, sensitive_col, name):
    # Group data by Quasi-Identifiers
    # This finds all rows that look identical to an attacker
    groups = df.groupby(qis)
    
    # Calculate Metrics
    group_sizes = groups.size()
    min_k = group_sizes.min()                 # Smallest group size (K-Anonymity)
    
    # Re-identification Risk:
    # Count how many groups have size 1. These people are uniquely identifiable.
    num_unique_individuals = (group_sizes == 1).sum()
    risk_percentage = (num_unique_individuals / len(df)) * 100
    
    # L-Diversity:
    # Check diversity of sensitive values within groups
    l_values = groups[sensitive_col].nunique()
    min_l = l_values.min()

    print(f"--- {name} ---")
    print(f"Min K-Anonymity: {min_k}")
    print(f"Min L-Diversity: {min_l}")
    print(f"Re-identification Risk: {num_unique_individuals} unique users ({risk_percentage:.2f}%)")
    print("-" * 30)

# 4. Run Comparison
# Note: For Original, we use the raw data. For Anonymized, we use the buckets.
print("SECURITY & PRIVACY REPORT\n")
get_privacy_metrics(df_orig, qis_orig, sensitive_col, "Original Dataset")
get_privacy_metrics(df_anon.fillna('Unknown'), qis_anon, sensitive_col, "Anonymized Dataset")