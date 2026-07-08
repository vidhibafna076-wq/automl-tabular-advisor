from typing import Any


def _add_issue(
    issues: list[dict[str, Any]],
    severity: str,
    issue: str,
    finding: str,
    why_it_matters: str,
    recommendation: str,
) -> None:
    """
    Helper function to add a structured data quality issue.

    We use this helper so every issue has the same format.
    """

    issues.append(
        {
            "severity": severity,
            "issue": issue,
            "finding": finding,
            "why_it_matters": why_it_matters,
            "recommendation": recommendation,
        }
    )


def _check_small_dataset(profile: dict[str, Any], issues: list[dict[str, Any]]) -> None:
    """
    Warn when the dataset is small.

    Small datasets can make model evaluation unstable because the train/test split
    may not represent the real population well.
    """

    rows = profile["rows"]

    if rows < 100:
        severity = "high"
        recommendation = (
            "Use cross-validation and treat model scores as highly uncertain. "
            "Collect more data if possible."
        )
    elif rows < 500:
        severity = "medium"
        recommendation = (
            "Use cross-validation and prefer simpler baseline models before "
            "using complex models."
        )
    else:
        return

    _add_issue(
        issues=issues,
        severity=severity,
        issue="Small dataset",
        finding=f"The dataset has only {rows} rows.",
        why_it_matters=(
            "Small datasets can produce unstable model evaluation results. "
            "A model may appear strong or weak simply because of the split."
        ),
        recommendation=recommendation,
    )


def _check_missing_values(profile: dict[str, Any], issues: list[dict[str, Any]]) -> None:
    """
    Detect columns with missing values.
    """

    missing_columns = [
        item
        for item in profile["missing_values"]
        if item["missing_count"] > 0
    ]

    if not missing_columns:
        return

    affected_columns = [
        f"{item['column']} ({item['missing_percentage']}%)"
        for item in missing_columns
    ]

    max_missing_percentage = max(
        item["missing_percentage"]
        for item in missing_columns
    )

    severity = "high" if max_missing_percentage >= 40 else "medium"

    _add_issue(
        issues=issues,
        severity=severity,
        issue="Missing values detected",
        finding="Missing values found in: " + ", ".join(affected_columns) + ".",
        why_it_matters=(
            "Most machine learning models cannot train directly on missing values. "
            "Missing data can also introduce bias if the missingness is not random."
        ),
        recommendation=(
            "Use imputation before model training. For the first version, use median "
            "imputation for numerical columns and most-frequent imputation for "
            "categorical columns."
        ),
    )


def _check_duplicate_rows(profile: dict[str, Any], issues: list[dict[str, Any]]) -> None:
    """
    Detect duplicate rows.
    """

    duplicate_rows = profile["duplicate_rows"]

    if duplicate_rows <= 0:
        return

    duplicate_percentage = round((duplicate_rows / profile["rows"]) * 100, 2)

    severity = "medium" if duplicate_percentage >= 5 else "low"

    _add_issue(
        issues=issues,
        severity=severity,
        issue="Duplicate rows detected",
        finding=(
            f"{duplicate_rows} duplicate rows were found "
            f"({duplicate_percentage}% of the dataset)."
        ),
        why_it_matters=(
            "Duplicate rows can make evaluation scores look better than they really are, "
            "especially if similar records appear in both training and testing data."
        ),
        recommendation="Consider removing duplicate rows before model training.",
    )


def _check_constant_columns(profile: dict[str, Any], issues: list[dict[str, Any]]) -> None:
    """
    Detect columns with only one unique value.
    """

    constant_columns = profile["constant_columns"]

    if not constant_columns:
        return

    _add_issue(
        issues=issues,
        severity="low",
        issue="Constant columns detected",
        finding="Constant columns found: " + ", ".join(constant_columns) + ".",
        why_it_matters=(
            "Columns with only one value do not provide useful information for prediction."
        ),
        recommendation="Drop constant columns before model training.",
    )


def _check_possible_id_columns(profile: dict[str, Any], issues: list[dict[str, Any]]) -> None:
    """
    Detect possible identifier columns.
    """

    possible_id_columns = profile["possible_id_columns"]

    if not possible_id_columns:
        return

    _add_issue(
        issues=issues,
        severity="medium",
        issue="Possible ID columns detected",
        finding="Possible identifier columns found: " + ", ".join(possible_id_columns) + ".",
        why_it_matters=(
            "ID columns usually do not generalise to new data. They can cause the model "
            "to memorise records instead of learning real patterns."
        ),
        recommendation=(
            "Review these columns. In most cases, ID-like columns should be removed "
            "before model training."
        ),
    )


def _check_high_cardinality_columns(
    profile: dict[str, Any],
    issues: list[dict[str, Any]],
) -> None:
    """
    Detect categorical columns with too many unique values.
    """

    high_cardinality_columns = profile["high_cardinality_columns"]

    if not high_cardinality_columns:
        return

    affected = [
        f"{item['column']} ({item['unique_count']} unique values)"
        for item in high_cardinality_columns
    ]

    _add_issue(
        issues=issues,
        severity="medium",
        issue="High-cardinality categorical columns detected",
        finding="High-cardinality columns found: " + ", ".join(affected) + ".",
        why_it_matters=(
            "One-hot encoding high-cardinality columns can create too many features, "
            "increase training time, and cause overfitting."
        ),
        recommendation=(
            "Consider dropping these columns, grouping rare categories, or using a "
            "controlled encoding strategy later."
        ),
    )


def _check_outliers(profile: dict[str, Any], issues: list[dict[str, Any]]) -> None:
    """
    Detect numerical columns with possible outliers.
    """

    numeric_summary = profile["numeric_summary"]

    outlier_columns = []

    for column, summary in numeric_summary.items():
        if summary["outlier_count"] > 0:
            outlier_columns.append(
                f"{column} ({summary['outlier_count']} outliers, "
                f"{summary['outlier_percentage']}%)"
            )

    if not outlier_columns:
        return

    _add_issue(
        issues=issues,
        severity="low",
        issue="Possible outliers detected",
        finding="Possible outliers found in: " + ", ".join(outlier_columns) + ".",
        why_it_matters=(
            "Outliers can strongly affect some models, especially linear models. "
            "Tree-based models are usually less sensitive."
        ),
        recommendation=(
            "Do not remove outliers automatically. Review them first. Later, compare "
            "model performance with and without outlier handling if needed."
        ),
    )


def _check_class_imbalance(
    task_info: Any,
    issues: list[dict[str, Any]],
) -> None:
    """
    Detect class imbalance for classification tasks.

    Class imbalance means one class is much more common than another.
    """

    if task_info.base_task != "classification":
        return

    class_percentages = task_info.target_summary.get("class_percentages", {})

    if not class_percentages:
        return

    largest_class = max(class_percentages, key=class_percentages.get)
    largest_percentage = class_percentages[largest_class]

    if largest_percentage >= 90:
        severity = "high"
    elif largest_percentage >= 75:
        severity = "medium"
    else:
        return

    _add_issue(
        issues=issues,
        severity=severity,
        issue="Class imbalance detected",
        finding=(
            f"The largest class is '{largest_class}' and represents "
            f"{largest_percentage}% of the target values."
        ),
        why_it_matters=(
            "Accuracy may be misleading on imbalanced datasets. A model can get high "
            "accuracy by mostly predicting the majority class."
        ),
        recommendation=(
            "Use metrics such as F1-score, recall, balanced accuracy, ROC-AUC, "
            "or PR-AUC instead of accuracy alone."
        ),
    )


def _check_regression_target_size(
    task_info: Any,
    issues: list[dict[str, Any]],
) -> None:
    """
    Add simple regression-specific warning if the target has very low variation.
    """

    if task_info.base_task != "regression":
        return

    target_summary = task_info.target_summary
    std = target_summary.get("std", 0.0)

    if std == 0:
        _add_issue(
            issues=issues,
            severity="high",
            issue="Regression target has no variation",
            finding="The regression target has a standard deviation of 0.",
            why_it_matters=(
                "If the target does not vary, the model has no meaningful pattern to learn."
            ),
            recommendation="Check whether the selected target column is correct.",
        )


def generate_quality_report(
    profile: dict[str, Any],
    task_info: Any,
) -> list[dict[str, Any]]:
    """
    Generate a structured data quality report.

    This function takes:
    - dataset profile from data_profiler.py
    - task information from task_detector.py

    It returns a list of warnings and recommendations.
    """

    issues: list[dict[str, Any]] = []

    _check_small_dataset(profile, issues)
    _check_missing_values(profile, issues)
    _check_duplicate_rows(profile, issues)
    _check_constant_columns(profile, issues)
    _check_possible_id_columns(profile, issues)
    _check_high_cardinality_columns(profile, issues)
    _check_outliers(profile, issues)
    _check_class_imbalance(task_info, issues)
    _check_regression_target_size(task_info, issues)

    if not issues:
        _add_issue(
            issues=issues,
            severity="info",
            issue="No major data quality issues detected",
            finding="The initial data quality scan did not find major structural issues.",
            why_it_matters=(
                "This means the dataset is ready for basic preprocessing and baseline "
                "model training."
            ),
            recommendation="Proceed to preprocessing and baseline model training.",
        )

    return issues