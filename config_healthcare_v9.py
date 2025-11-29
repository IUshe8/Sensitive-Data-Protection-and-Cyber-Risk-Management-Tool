# config_healthcare_v9.py

"""
Configuration "recipe" for the healthcare_dataset.csv.
VERSION 9: This config is identical to V8.
The anonymizer script will now handle augmentation
as a final step.
"""

# --- Rule for Insurance ---
INSURANCE_MAP = {
    'Medicare': 'Public',
    'Aetna': 'Private',
    'Blue Cross': 'Private',
    'Cigna': 'Private',
    'UnitedHealthcare': 'Private'
}

# --- Rule for Admission Type ---
ADMISSION_TYPE_MAP = {
    'Emergency': 'Non-Elective',
    'Urgent': 'Non-Elective',
    'Elective': 'Elective'
}

ANONYMIZATION_CONFIG_HEALTHCARE = {
    # Direct Identifiers (PII) to be removed
    'Name': {'type': 'remove'},
    'Doctor': {'type': 'remove'},
    'Hospital': {'type': 'remove'},
    'Room Number': {'type': 'remove'},
    'Blood Type': {'type': 'remove'}, # Kept from V4
    
    # --- CALCULATE Length of Stay ---
    'Length of Stay': {
        'type': 'calculate_los_binned',
        'col_admission': 'Date of Admission',
        'col_discharge': 'Discharge Date',
        'bins': [0, 7, 30, float('inf')], # 0-6 days, 7-29 days, 30+ days
        'labels': ['0-6 Days (Short)', '7-29 Days (Medium)', '30+ Days (Long)'],
        'na_default_label': '0-6 Days (Short)' # Default missing to Short
    },
    
    # We must still remove the original date columns
    'Date of Admission': {'type': 'remove'},
    'Discharge Date': {'type': 'remove'},
    # -----------------------------------------------

    # --- Quasi-Identifiers (QIs) ---
    'Insurance Provider': {
        'type': 'generalize_simple_map',
        'mapping': INSURANCE_MAP,
        'default': 'Other'
    },
    'Billing Amount': {
        'type': 'generalize_bin',
        'bins': [0, 30000, float('inf')],
        'labels': ['0-30k', '30k+']
    },
    'Age': {
        'type': 'generalize_bin',
        'bins': [0, 18, 65, 120],
        'labels': ['0-17 (Child)', '18-64 (Adult)', '65+ (Senior)']
    },
    'Admission Type': {
        'type': 'generalize_simple_map',
        'mapping': ADMISSION_TYPE_MAP,
        'default': 'Elective'
    },
    'Gender': {'type': 'qi_exact'},

    # --- Sensitive columns ---
    'Medical Condition': {'type': 'sensitive'},
    'Medication': {'type': 'sensitive'},
    'Test Results': {'type': 'sensitive'}
}