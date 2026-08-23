# Agentic AutoML Advisor Report

**Created at:** 2026-08-23T18:27:29

**Dataset:** `data/sample/holdout_demo.csv`

**Target column:** `Approved`

**User objective:** Predict whether a loan application should be approved

---

# 1. Executive Summary

The experiment completed with overall reliability marked as **high**.

The final recommendation decision is: **recommend_candidate_model**.

A candidate model can be recommended based on the current experiment evidence.

# 2. Dataset Overview

- **Rows:** 500
- **Columns:** 7
- **Feature count:** 6
- **Target column:** Approved

## Numerical Features

- ApplicantIncome
- LoanAmount
- Credit_History
- Age

## Categorical Features

- Employment_Type
- Property_Area

# 3. Task Detection

- **Task type:** binary_classification
- **Base task:** classification
- **Reason:** The target contains two distinct categorical values.

## Target Summary

- **unique_values:** 2
- **classes:** ['No', 'Yes']

# 4. Data Quality Findings

No data quality issues were recorded.

# 5. Experiment Plan

No experiment plan was created.

# 6. Preprocessing Strategy

## Columns Dropped

- None

## Final Numerical Features

- ApplicantIncome
- LoanAmount
- Credit_History
- Age

## Final Categorical Features

- Employment_Type
- Property_Area

## Missing Value Strategy

- **Numerical:** median
- **Categorical:** most_frequent

## Leakage Control

The preprocessing pipeline is designed to be fitted inside cross-validation folds, not on the full dataset before evaluation.

# 7. Training Summary

- **Task type:** binary_classification
- **Validation method:** cross_validation_with_holdout
- **CV folds:** 5
- **Primary metric:** f1
- **Models attempted:** 4
- **Models completed:** 4
- **Models failed:** 0

**Leakage control:** Preprocessing was fitted inside each cross-validation fold.

## Validation Design

- **Strategy:** cross_validation_with_final_holdout
- **Cross-validation used:** True
- **Rows used for cross-validation:** 400
- **Final holdout used:** True
- **Holdout rows:** 100
- **Reason:** The dataset contained enough rows to reserve an untouched final holdout.

## Guarded Hyperparameter Tuning

- **Eligible:** True
- **Eligibility reason:** The experiment satisfied the guarded tuning policy.

- **Tuning status:** completed
- **Model:** Logistic Regression
- **Primary metric:** f1
- **Baseline CV score:** 0.8
- **Tuned CV score:** 0.82
- **Improvement:** 0.02
- **Accepted for final evaluation:** True
- **Successful trials:** 12
- **Duration seconds:** 2.5
- **Holdout used during tuning:** False

### Best Parameters

- **C:** 0.5
- **class_weight:** None

## Final Holdout Evaluation

- **Status:** completed
- **Holdout used:** True
- **Reason:** The selected useful candidate was fitted on the training portion and evaluated once on the untouched holdout.
- **Model:** Logistic Regression
- **Model ID:** logistic_regression
- **Training rows:** 400
- **Holdout rows:** 100
- **Primary metric:** f1
- **Cross-validation score:** 0.82
- **Holdout score:** 0.81
- **Passed holdout guardrail:** True
- **Guardrail explanation:** Holdout f1 was 0.8100. The minimum permitted value under the current guardrail was 0.7200.

| Holdout Metric | Value |
|---|---:|
| accuracy | 0.83 |
| balanced_accuracy | 0.82 |
| precision | 0.84 |
| recall | 0.79 |
| f1 | 0.81 |

# 8. Model Results

No model results were created.

# 9. Leaderboard and Comparison

- **Primary metric:** f1
- **Dummy baseline score:** 0.6
- **Best overall model:** Logistic Regression
- **Best useful model:** Logistic Regression
- **Recommendation status:** useful_model_selected

## Important Notes

- Logistic Regression clearly outperformed the dummy baseline.

## Leaderboard

| Rank | Model | Primary Score | Beats Dummy | Reliability | Overfit Gap |
|---:|---|---:|---|---|---:|
| 1 | Logistic Regression | 0.82 | True | high | 0.03 |
| 2 | Dummy Classifier | 0.6 | False | baseline | 0.0 |

## Leaderboard Reliability Flags

### Logistic Regression
- No reliability flags.

### Dummy Classifier
- No reliability flags.


# 10. Reliability Critic Report

- **Overall reliability:** high
- **Recommendation decision:** recommend_candidate_model
- **Candidate met tuning reliability gate:** True

## Selected Candidate

- **Model:** Logistic Regression
- **Model ID:** logistic_regression
- **Primary metric:** f1
- **Primary score:** 0.82
- **Reliability level:** high

## Critic Findings

### 1. Final holdout evidence passed [INFO]
**Evidence:** Logistic Regression was evaluated on 100 untouched rows.

**Recommendation:** Treat Logistic Regression as the current candidate.

## Next Actions

- Proceed with the selected candidate as the current recommended model.
- Save the critic-approved fitted pipeline and its metadata.

# 11. Model Artifact and Explainability

## Model Artifact Status

- **Status:** saved
- **Reason:** Final fitted pipeline was saved because the critic approved the candidate model.
- **Critic decision:** recommend_candidate_model
- **Artifact path:** `outputs/models/approved_logistic_regression_pipeline.joblib`
- **Metadata path:** `outputs/models/approved_logistic_regression_metadata.json`

## Feature Importance

- **Importance type:** linear_absolute_coefficient

| Rank | Feature | Importance |
|---:|---|---:|
| 1 | numerical__Credit_History | 1.42 |
| 2 | numerical__ApplicantIncome | 0.86 |

# 12. Tool Execution Timeline

1. **[STARTED]** final_report_tool - Started final report generation.
2. **[COMPLETED]** final_report_tool - Final report saved to D:\automl-tabular-advisor\outputspytest-temp-20260823-182713\test_approved_model_report_inc0\approved_run\final_report.md.

# 13. Reproducibility and Provenance

No reproducibility metadata was recorded for this run.

# 14. Final Conclusion

The controlled workflow completed through reliability review, model persistence assessment, and reporting. Guarded tuning status: **completed**. Final holdout status: **completed**. The critic approved the selected candidate based on the current evidence. Review feature importance, artifact compatibility, domain suitability, and deployment constraints before use.
