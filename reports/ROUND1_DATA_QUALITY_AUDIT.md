# Round 1 — Data Quality Audit

## Dataset

- Raw rows: 50,000
- Columns: 39
- Unique applications: 49,800
- Exact duplicate rows: 200
- Duplicate application IDs: 200
- Total missing cells: 8,113
- Invalid dates: 0

## Leakage Columns

- `application_id`
- `actual_initial_scrutiny_days`
- `actual_document_verification_days`
- `actual_department_review_days`
- `actual_approval_days`
- `actual_certificate_issuance_days`
- `actual_total_tat_days`
- `actual_remaining_tat_days`
- `missed_target`
- `delay_overrun_days`
- `actual_bottleneck_stage`

These columns must be excluded from the predictive feature matrix.
