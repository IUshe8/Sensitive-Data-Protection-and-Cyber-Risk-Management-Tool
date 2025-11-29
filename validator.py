import pandas as pd

# --- CONFIGURATION ---
ANONYMIZED_FILE_PATH = "anonymized_augmented_dataset.csv"
# ---------------------------
TARGET_K = 5
TARGET_L = 2

# --- QI LIST  ---
QI_COLUMNS = [
    'Age',
    'Gender',
    'Admission Type',
    'Insurance Provider',
    'Billing Amount',
    'Length of Stay'
]
# -------------------------

SENSITIVE_COLUMNS = [
    'Medical Condition',
    'Medication',
    # 'Test Results'
]

# --- PII LIST ---
PII_COLUMNS_TO_CHECK = [
    'Name',
    'Doctor',
    'Hospital',
    'Room Number',
    'Blood Type',
    'Discharge Date',
    'Date of Admission',
    'SSN',
    'Phone',
    'Email'
]
# ---------------------------

# --- VALIDATION FUNCTIONS ---
def run_phase1_pii_check(df_anonymized):
    print_header("Phase 1: Direct Identifier (PII) Removal Test")
    found_pii = []
    
    for col in PII_COLUMNS_TO_CHECK:
        if col in df_anonymized.columns:
            found_pii.append(col)
            
    if found_pii:
        print(f"--- FAILED ---")
        print(f"Error: The following columns were found but should have been removed:")
        print(", ".join(found_pii))
        return False
    else:
        print("--- PASSED ---")
        print("No direct PII or removed QI columns were found.")
        return True

def run_phase2_k_anonymity_check(df_anonymized, qi_cols, target_k):
    print_header("Phase 2: k-Anonymity Test")
    
    missing_qi_cols = [col for col in qi_cols if col not in df_anonymized.columns]
    if missing_qi_cols:
        print(f"--- FAILED ---")
        print(f"Error: The following required QI columns are missing from the dataset:")
        print(", ".join(missing_qi_cols))
        return False
        
    df_qi_grouped = df_anonymized[qi_cols].astype(str).fillna('NA_VALUE')
    k_counts = df_qi_grouped.groupby(qi_cols).size()
    
    if k_counts.empty:
        print("--- FAILED ---")
        print("Error: No data groups found. The dataset might be empty or QIs are wrong.")
        return False
        
    min_k = k_counts.min()
    
    print(f"Target k-value: {target_k}")
    print(f"Minimum k-value found: {min_k}")
    
    if min_k < target_k:
        print(f"--- FAILED ---")
        failing_groups = k_counts[k_counts < target_k]
        print(f"Found {len(failing_groups)} groups that FAIL the k={target_k} test.")
        print("Example failing groups (first 5):")
        for index, value in failing_groups.head(5).items():
            print(f"  {index}  ->  k={value}")
        return False
    else:
        print("--- PASSED ---")
        print(f"All {len(k_counts)} unique groups meet the k={target_k} requirement.")
        return True

def run_phase3_l_diversity_check(df_anonymized, qi_cols, sensitive_cols, target_l):
    print_header("Phase 3: l-Diversity Test")
    all_passed = True
    
    df_qi_grouped = df_anonymized[qi_cols].astype(str).fillna('NA_VALUE')
    
    for sensitive_col in sensitive_cols:
        if sensitive_col not in df_anonymized.columns:
            print(f"Warning: Sensitive column '{sensitive_col}' not found. Skipping.")
            continue
            
        print(f"\nChecking l-diversity for: '{sensitive_col}'")
        
        df_l_check = df_anonymized[[sensitive_col]].copy()
        df_l_check['group_key'] = list(df_qi_grouped.itertuples(index=False, name=None))
        
        l_counts = df_l_check.groupby('group_key')[sensitive_col].nunique()
        
        if l_counts.empty:
            print(f"Error: Could not calculate l-diversity. Groups may be empty.")
            all_passed = False
            continue
            
        min_l = l_counts.min()
        
        print(f"Target l-value: {target_l}")
        print(f"Minimum l-value found: {min_l}")
        
        if min_l < target_l:
            print(f"--- FAILED (for '{sensitive_col}') ---")
            failing_groups = l_counts[l_counts < target_l]
            print(f"Found {len(failing_groups)} groups that FAIL the l={target_l} test.")
            all_passed = False
        else:
            print(f"--- PASSED (for '{sensitive_col}') ---")
            
    return all_passed

def print_header(title):
    print("\n" + "="*60)
    print(f" {title} ")
    print("="*60)

# --- MAIN EXECUTION ---
def main():
    print("Starting Anonymization Validation (V9 RULES)...") # <-- Updated print
    
    try:
        df = pd.read_csv(ANONYMIZED_FILE_PATH, dtype=str)
        print(f"Successfully loaded '{ANONYMIZED_FILE_PATH}'.")
    except FileNotFoundError:
        print(f"--- FATAL ERROR ---")
        print(f"File not found: '{ANONYMIZED_FILE_PATH}'")
        print("Please run the 'anonymizer_healthcare.py' script first to generate this file.")
        return
    except Exception as e:
        print(f"--- FATAL ERROR ---")
        print(f"An error occurred loading the file: {e}")
        return
        
    phase1_pass = run_phase1_pii_check(df)
    
    if not phase1_pass:
        print("\nHalting tests because Phase 1 FAILED.")
        phase2_pass = False
        phase3_pass = False
    else:
        phase2_pass = run_phase2_k_anonymity_check(df, QI_COLUMNS, TARGET_K)
        
        if not phase2_pass:
            print("\nSkipping Phase 3 because Phase 2 FAILED.")
            phase3_pass = False
        else:
            phase3_pass = run_phase3_l_diversity_check(df, QI_COLUMNS, SENSITIVE_COLUMNS, TARGET_L)
    
    print_header("Final Validation Summary")
    print(f"Phase 1 (PII Removal):   {'PASSED' if phase1_pass else 'FAILED'}")
    print(f"Phase 2 (k-Anonymity):   {'PASSED' if phase2_pass else 'FAILED'}")
    print(f"Phase 3 (l-Diversity):   {'PASSED' if phase3_pass else 'FAILED'}")
    print("="*60)
    
    if phase1_pass and phase2_pass and phase3_pass:
        print("\nOverall Result: PASSED. The data meets the configured privacy targets.")
    else:
        print("\nOverall Result: FAILED. Review the log above to identify privacy risks.")

if __name__ == "__main__":
    main()