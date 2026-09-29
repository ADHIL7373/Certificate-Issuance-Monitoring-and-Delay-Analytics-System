# Round 1 — Outlier Analysis Report

## Approach

IQR-based statistical outlier detection was applied to numerical variables.
Additional domain-specific validation rules were applied to identify impossible or logically inconsistent values.

## Outlier Policy

- Statistical outliers were detected but not automatically removed.
- Negative `days_to_target` values were retained because they indicate that an application's target date has been crossed.
- Extreme turnaround times were retained when operationally plausible.
- Domain-invalid values are treated as data-quality issues.
- Severe delays are important to the project's prediction objective and should not be removed merely because they are statistically extreme.

## Generated Files

- `D:\Datascience Hackathon\certificate-delay-intelligence\outputs\round1_outlier_report.csv`
- `D:\Datascience Hackathon\certificate-delay-intelligence\outputs\round1_domain_outlier_checks.csv`
- `D:\Datascience Hackathon\certificate-delay-intelligence\outputs\round1_tat_extreme_analysis.csv`
- `D:\Datascience Hackathon\certificate-delay-intelligence\outputs\round1_top_20_tat_extremes.csv`