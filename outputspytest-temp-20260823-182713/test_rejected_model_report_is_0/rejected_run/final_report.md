# Agentic AutoML Advisor Report

**Created at:** 2026-08-23T18:27:29

**Dataset:** `data/sample/loan_sample.csv`

**Target column:** `Loan_Status`

**User objective:** Predict whether a loan application will be approved

---

# 1. Executive Summary

The experiment completed with overall reliability marked as **low**.

The final recommendation decision is: **do_not_recommend_model**.

No final model should be recommended from this run. The workflow executed successfully, but the evidence is not strong enough for real model selection.

# 2. Dataset Overview

- **Rows:** 10
- **Columns:** 8
- **Feature count:** 7
- **Target column:** Loan_Status

## Numerical Features

- ApplicantIncome
- LoanAmount
- Credit_History

## Categorical Features

- Loan_ID
- Gender
- Married
- Property_Area

# 3. Task Detection

- **Task type:** binary_classification
- **Base task:** classification
- **Reason:** The target contains two distinct categorical values.

## Target Summary

- **unique_values:** 2
- **classes:** ['N', 'Y']

# 4. Data Quality Findings

No data quality issues were recorded.

# 5. Experiment Plan

No experiment plan was created.

# 6. Preprocessing Strategy

## Columns Dropped

- Loan_ID

## Final Numerical Features

- ApplicantIncome
- LoanAmount
- Credit_History

## Final Categorical Features

- Gender
- Married
- Property_Area

## Missing Value Strategy

- **Numerical:** median
- **Categorical:** most_frequent

## Leakage Control

The preprocessing pipeline is designed to be fitted inside cross-validation folds, not on the full dataset before evaluation.

# 7. Training Summary

- **Task type:** binary_classification
- **Validation method:** cross_validation
- **CV folds:** 2
- **Primary metric:** f1
- **Models attempted:** 4
- **Models completed:** 4
- **Models failed:** 0

**Leakage control:** Preprocessing was fitted inside each cross-validation fold.

## Validation Design

- **Strategy:** cross_validation_only
- **Cross-validation used:** True
- **Rows used for cross-validation:** 10
- **Final holdout used:** False
- **Holdout rows:** 0
- **Reason:** The dataset was too small to reserve a final holdout.

## Guarded Hyperparameter Tuning

No guarded tuning decision was recorded.

## Final Holdout Evaluation

No final holdout evaluation result was recorded.

# 8. Model Results

No model results were created.

# 9. Leaderboard and Comparison

- **Primary metric:** f1
- **Dummy baseline score:** 0.67
- **Best overall model:** Dummy Classifier
- **Best useful model:** None
- **Recommendation status:** no_model_clearly_beats_dummy

## Important Notes

- No non-dummy candidate clearly outperformed the dummy baseline.

## Leaderboard

No leaderboard was created.

# 10. Reliability Critic Report

- **Overall reliability:** low
- **Recommendation decision:** do_not_recommend_model
- **Candidate met tuning reliability gate:** False

## Selected Candidate

- No model was selected.

## Critic Findings

### 1. No useful model clearly beats the dummy baseline [HIGH]
**Evidence:** The comparison summary did not identify a useful non-dummy candidate.

**Recommendation:** Do not recommend a final model from this run.

## Next Actions

- Do not select or persist a final model from this run.
- Use a larger and more representative dataset.

# 11. Model Artifact and Explainability

## Model Artifact Status

- **Status:** skipped
- **Reason:** No model was saved because the critic determined that the current evidence does not justify recommending a model.
- **Critic decision:** do_not_recommend_model

## Feature Importance

Feature importance status: **skipped**. Reason: Feature importance was skipped because no final model was saved.

# 12. Tool Execution Timeline

1. **[STARTED]** final_report_tool - Started final report generation.
2. **[COMPLETED]** final_report_tool - Final report saved to D:\automl-tabular-advisor\outputspytest-temp-20260823-182713\test_rejected_model_report_is_0\rejected_run\final_report.md.

# 13. Reproducibility and Provenance

No reproducibility metadata was recorded for this run.

# 14. Final Conclusion

The controlled workflow completed through reliability review, model persistence assessment, and reporting. Guarded tuning status: **not recorded**. Final holdout status: **not recorded**. The AutoML advisor should not recommend a final model from this run. Review the critic findings and improve the modelling evidence before rerunning the workflow.
