import pandas as pd
import numpy as np
from pathlib import Path

# Import paths from config
from config import RAW_TRAIN_DATA, SCHEMA_DICTIONARY, FEATURE_WHITELIST

def inspect_dataset():
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)

    print("="*50)
    print("ROUND 1 - DATASET INSPECTION")
    print("="*50)

    # 1. Load dataset
    print(f"Loading data from: {RAW_TRAIN_DATA}")
    df = pd.read_csv(RAW_TRAIN_DATA, low_memory=False)

    # 2. Basic info
    print("\n--- 2. Dataset Shape & Basic Info ---")
    print(f"Shape: {df.shape}")
    print("\nColumn Names:")
    print(list(df.columns))
    print("\nData Types:")
    print(df.dtypes)
    
    print("\nFirst 5 rows:")
    print(df.head())
    
    print("\nLast 5 rows:")
    print(df.tail())

    # 3. Missing Values and duplicates
    print("\n--- 3. Missing Values & Duplicates ---")
    missing_counts = df.isnull().sum()
    missing_pct = (df.isnull().sum() / len(df)) * 100
    missing_df = pd.DataFrame({'Missing Values': missing_counts, 'Percentage': missing_pct})
    print("\nMissing Values (columns with missing values):")
    cols_with_missing = missing_df[missing_df['Missing Values'] > 0]
    if len(cols_with_missing) > 0:
        print(cols_with_missing)
    else:
        print("None. All columns have 0 missing values.")
    print(f"Total missing values across dataset: {missing_counts.sum()}")
    
    exact_duplicates = df.duplicated().sum()
    print(f"\nExact duplicate rows: {exact_duplicates}")
    
    if 'application_id' in df.columns:
        duplicate_ids = df['application_id'].duplicated().sum()
        print(f"Duplicate application IDs: {duplicate_ids}")

    # 4. Distributions
    print("\n--- 4. Distributions ---")
    if 'missed_target' in df.columns:
        print("\nmissed_target distribution:")
        print(df['missed_target'].value_counts(dropna=False))
        
    if 'certificate_type' in df.columns:
        print("\ncertificate_type distribution:")
        print(df['certificate_type'].value_counts(dropna=False))
        
    if 'current_stage' in df.columns:
        print("\ncurrent_stage distribution:")
        print(df['current_stage'].value_counts(dropna=False))
        
    if 'district' in df.columns:
        print("\ndistrict distribution:")
        print(df['district'].value_counts(dropna=False))

    # 5. Numerical Statistics
    print("\n--- 5. Numerical Descriptive Statistics ---")
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    if len(numerical_cols) > 0:
        print(df[numerical_cols].describe().T)
    else:
        print("No numerical columns found.")

    # 6. Categorical Statistics
    print("\n--- 6. Categorical Statistics ---")
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    if len(categorical_cols) > 0:
        print(df[categorical_cols].describe().T)
        print("\nCategorical Unique Counts:")
        for col in categorical_cols:
            print(f"{col}: {df[col].nunique()} unique values")
    else:
        print("No categorical columns found.")

    # 7. Required Target Columns
    print("\n--- 7. Target Columns Check ---")
    expected_targets = [
        'missed_target', 
        'actual_total_tat_days', 
        'actual_remaining_tat_days', 
        'delay_overrun_days'
    ]
    for target in expected_targets:
        status = "FOUND" if target in df.columns else "MISSING"
        print(f"{target}: {status}")

    # 8. Schema Dictionary Check
    print("\n--- 8. Schema Dictionary Verification ---")
    if not SCHEMA_DICTIONARY.exists():
        print(f"Schema dictionary not found at {SCHEMA_DICTIONARY}")
    else:
        schema_df = pd.read_csv(SCHEMA_DICTIONARY)

        expected_cols = schema_df["column"].dropna().tolist()

        missing_from_dataset = set(expected_cols) - set(df.columns)
        extra_in_dataset = set(df.columns) - set(expected_cols)

        if missing_from_dataset:
            print(f"Expected columns missing from dataset: {missing_from_dataset}")
        else:
            print("All expected columns from schema are present in the dataset.")

        if extra_in_dataset:
            print(f"Columns in dataset not in schema: {extra_in_dataset}")
        else:
            print("No extra columns found outside the schema.")

    # 9. Leakage Inventory
    print("\n--- 9. Leakage Inventory ---")
    leakage_cols = [
        'actual_initial_scrutiny_days',
        'actual_document_verification_days',
        'actual_department_review_days',
        'actual_approval_days',
        'actual_certificate_issuance_days',
        'actual_total_tat_days',
        'actual_remaining_tat_days',
        'missed_target',
        'delay_overrun_days',
        'actual_bottleneck_stage',
        'application_id'
    ]
    
    found_leakage = [col for col in leakage_cols if col in df.columns]
    print("Potential leakage/target columns present in raw data (must be excluded from model features):")
    for col in found_leakage:
        print(f"- {col}")

    print("\n" + "="*50)
    print("INSPECTION COMPLETE. NO MODIFICATIONS MADE.")
    print("="*50)


if __name__ == "__main__":
    inspect_dataset()
