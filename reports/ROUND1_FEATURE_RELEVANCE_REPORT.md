# Round 1 — Feature Relevance Report

## Objective

Assess the relationship between candidate input features and the `missed_target` classification target before model training.

## Methods

- Point-biserial correlation for numerical features.
- Mann-Whitney U comparison for numerical features.
- Chi-square test for categorical features.
- Cramér's V for categorical association strength.
- Mutual information as supplementary nonlinear relevance evidence.

## Leakage Policy

Actual processing durations, actual total TAT, actual remaining TAT, delay overrun, actual bottleneck stage, application ID and target itself are excluded from predictive inputs.

## Important Note

Statistical association does not establish causation.
Final model feature selection will also consider prediction-time availability, domain meaning, redundancy and leakage risk.

## Outputs

- `D:\Datascience Hackathon\certificate-delay-intelligence\outputs\round1_feature_relevance.csv`
- `D:\Datascience Hackathon\certificate-delay-intelligence\outputs\round1_feature_candidates.csv`