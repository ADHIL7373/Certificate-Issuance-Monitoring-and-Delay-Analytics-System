# Round 1 — Data Cleaning Report

## Dataset Summary

- Original rows: 50,000
- Exact duplicate rows removed: 200
- Rows after exact deduplication: 49,800
- Final columns: 40

## Duplicate Application IDs

- Duplicate application IDs remaining: 0
- Non-identical duplicate IDs were preserved for investigation.

## Date Handling

- Remaining invalid dates: 0

## Cleaning Operations

- Removed exact duplicate rows.
- Preserved non-identical duplicate application IDs for investigation.
- Normalized categorical values.
- Parsed mixed submission-date formats.
- Converted invalid numeric tokens to NaN.
- Validated domain ranges.
- Validated non-negative operational counts.
- Checked document-count logical consistency.
- Imputed numeric values using certificate type + current stage medians.
- Used global median as fallback.
- Imputed categorical values using mode.
- Added data_quality_flag.

## Important Modeling Rules

- Negative days_to_target values were retained.
- Operationally extreme TAT values were not removed.
- Outcome/leakage columns must not be used as model features.

## Output

- Clean dataset: `D:\Datascience Hackathon\certificate-delay-intelligence\outputs\round1_cleaned_applications.csv`
- Duplicate-ID investigation: `D:\Datascience Hackathon\certificate-delay-intelligence\outputs\round1_duplicate_application_ids.csv`