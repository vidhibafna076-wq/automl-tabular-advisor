def print_dataset_summary(metadata):
    """
    Print basic dataset loading information.
    """

    print("\nDataset loaded successfully.")
    print("-" * 60)
    print(f"File name: {metadata['file_name']}")
    print(f"Rows: {metadata['rows']}")
    print(f"Columns: {metadata['columns']}")
    print(f"Feature columns: {metadata['feature_count']}")
    print(f"Target column: {metadata['target_column']}")
    print(f"Target missing values: {metadata['target_missing_count']}")

    print("\nColumns:")
    for column in metadata["column_names"]:
        print(f"- {column}")


def print_task_summary(task_info):
    """
    Print the detected ML task information.
    """

    print("\nTask detection completed.")
    print("-" * 60)
    print(f"Detected task type: {task_info['task_type']}")
    print(f"Base task: {task_info['base_task']}")
    print(f"Reason: {task_info['reason']}")
    print(f"Unique target values: {task_info['unique_count']}")
    print(f"Missing target values: {task_info['missing_count']}")

    print("\nTarget preview:")
    for value in task_info["unique_values_preview"]:
        print(f"- {value}")

    print("\nTarget summary:")

    if task_info["base_task"] == "classification":
        print("Class counts:")
        for label, count in task_info["target_summary"]["class_counts"].items():
            print(f"- {label}: {count}")

        print("\nClass percentages:")
        for label, percentage in task_info["target_summary"]["class_percentages"].items():
            print(f"- {label}: {percentage}%")

    else:
        for key, value in task_info["target_summary"].items():
            print(f"- {key}: {value}")


def print_profile_summary(profile):
    """
    Print the dataset profile in a readable format.
    """

    print("\nDataset profiling completed.")
    print("-" * 60)

    print(f"Rows: {profile['rows']}")
    print(f"Columns: {profile['columns']}")
    print(f"Feature count: {profile['feature_count']}")

    print("\nNumerical columns:")
    if profile["numerical_columns"]:
        for column in profile["numerical_columns"]:
            print(f"- {column}")
    else:
        print("- None")

    print("\nCategorical columns:")
    if profile["categorical_columns"]:
        for column in profile["categorical_columns"]:
            print(f"- {column}")
    else:
        print("- None")

    print("\nColumns with missing values:")
    missing_columns = [
        item
        for item in profile["missing_values"]
        if item["missing_count"] > 0
    ]

    if missing_columns:
        for item in missing_columns:
            print(
                f"- {item['column']}: "
                f"{item['missing_count']} missing "
                f"({item['missing_percentage']}%)"
            )
    else:
        print("- None")

    print(f"\nDuplicate rows: {profile['duplicate_rows']}")

    print("\nPossible ID columns:")
    if profile["possible_id_columns"]:
        for column in profile["possible_id_columns"]:
            print(f"- {column}")
    else:
        print("- None")

    print("\nHigh-cardinality categorical columns:")
    if profile["high_cardinality_columns"]:
        for item in profile["high_cardinality_columns"]:
            print(
                f"- {item['column']}: "
                f"{item['unique_count']} unique values "
                f"(ratio: {item['unique_ratio']})"
            )
    else:
        print("- None")

    print("\nNumeric summary:")
    if profile["numeric_summary"]:
        for column, summary in profile["numeric_summary"].items():
            print(f"\n{column}:")
            print(f"  min: {summary['min']}")
            print(f"  max: {summary['max']}")
            print(
                "  mean: "
                f"{round(summary['mean'], 2) if summary['mean'] is not None else None}"
            )
            print(f"  median: {summary['median']}")
            print(
                f"  outliers: {summary['outlier_count']} "
                f"({summary['outlier_percentage']}%)"
            )
    else:
        print("- None")


def print_quality_report(quality_issues):
    """
    Print data quality warnings and recommendations.
    """

    print("\nData quality report completed.")
    print("-" * 60)

    for index, issue in enumerate(quality_issues, start=1):
        print(f"\n{index}. {issue['issue']} [{issue['severity'].upper()}]")
        print(f"Finding: {issue['finding']}")
        print(f"Why it matters: {issue['why_it_matters']}")
        print(f"Recommendation: {issue['recommendation']}")


def print_tool_history(tool_history):
    """
    Print a simple factual audit timeline of tool actions.
    """

    print("\nTool execution timeline.")
    print("-" * 60)

    if not tool_history:
        print("- No tool history recorded.")
        return

    for index, event in enumerate(tool_history, start=1):
        print(
            f"{index}. [{event['status'].upper()}] "
            f"{event['tool_name']} - {event['message']}"
        )


def print_system_warnings(warnings):
    """
    Print system-level warnings.
    """

    if not warnings:
        return

    print("\nSystem warnings:")
    for warning in warnings:
        print(f"- {warning}")

def print_experiment_plan(plan):
    """
    Print the experiment plan created by the planner agent.
    """

    print("\nExperiment plan created.")
    print("-" * 60)

    if not plan:
        print("- No plan was created.")
        return

    for index, step in enumerate(plan, start=1):
        approval_text = "Yes" if step["requires_user_approval"] else "No"

        print(f"\n{index}. {step['title']} [{step['priority'].upper()}]")
        print(f"Action: {step['action']}")
        print(f"Reason: {step['reason']}")
        print(f"Requires user approval: {approval_text}")
        print(f"Status: {step['status']}")


def print_preprocessing_config(config, pending_approvals):
    """
    Print the preprocessing configuration created by the preprocessing plan tool.
    """

    print("\nPreprocessing configuration created.")
    print("-" * 60)

    if not config:
        print("- No preprocessing configuration available.")
        return

    print(f"Target column: {config['target_column']}")

    print("\nColumns to drop:")
    if config["columns_to_drop"]:
        for column in config["columns_to_drop"]:
            reason = config["drop_reasons"].get(column, "not specified")
            print(f"- {column} ({reason})")
    else:
        print("- None")

    print("\nFinal numerical features:")
    if config["numerical_features"]:
        for column in config["numerical_features"]:
            print(f"- {column}")
    else:
        print("- None")

    print("\nFinal categorical features:")
    if config["categorical_features"]:
        for column in config["categorical_features"]:
            print(f"- {column}")
    else:
        print("- None")

    print("\nMissing value strategy:")
    print(
        "- Numerical: "
        f"{config['missing_value_strategy']['numerical']['strategy']} "
        f"for {config['missing_value_strategy']['numerical']['columns_with_missing_values']}"
    )
    print(
        "- Categorical: "
        f"{config['missing_value_strategy']['categorical']['strategy']} "
        f"for {config['missing_value_strategy']['categorical']['columns_with_missing_values']}"
    )

    print("\nScaling strategy:")
    print(f"- Scaler: {config['scaling_strategy']['scaler']}")
    print(f"- Reason: {config['scaling_strategy']['reason']}")

    print("\nEncoding strategy:")
    print(f"- Encoder: {config['encoding_strategy']['categorical_encoder']}")
    print(f"- Unknown categories: {config['encoding_strategy']['handle_unknown']}")
    print(f"- Reason: {config['encoding_strategy']['reason']}")

    print("\nValidation strategy:")
    for key, value in config["validation_strategy"].items():
        print(f"- {key}: {value}")

    print("\nMetric strategy:")
    print(f"- Primary metric: {config['metric_strategy']['primary_metric']}")
    print(f"- Secondary metrics: {config['metric_strategy']['secondary_metrics']}")
    print(f"- Reason: {config['metric_strategy']['reason']}")

    if pending_approvals:
        print("\nPending approvals:")
        for approval in pending_approvals:
            print(f"- {approval['title']}: {approval['columns']}")
            print(f"  Reason: {approval['reason']}")
    else:
        print("\nPending approvals:")
        print("- None")

def print_preprocessing_pipeline_summary(summary):
    """
    Print the preprocessing pipeline summary.
    """

    print("\nPreprocessing pipeline created.")
    print("-" * 60)

    if not summary:
        print("- No preprocessing pipeline summary available.")
        return

    print(f"Pipeline type: {summary['pipeline_type']}")
    print(f"Is fitted: {summary['is_fitted']}")
    print(f"Rows after target cleaning: {summary['rows_after_target_cleaning']}")
    print(f"Target column: {summary['target_column']}")

    print("\nColumns dropped:")
    if summary["columns_to_drop"]:
        for column in summary["columns_to_drop"]:
            print(f"- {column}")
    else:
        print("- None")

    print("\nNumerical pipeline:")
    print("Features:")
    if summary["numerical_pipeline"]["features"]:
        for column in summary["numerical_pipeline"]["features"]:
            print(f"- {column}")
    else:
        print("- None")

    print("Steps:")
    if summary["numerical_pipeline"]["steps"]:
        for step in summary["numerical_pipeline"]["steps"]:
            print(f"- {step}")
    else:
        print("- None")

    print("\nCategorical pipeline:")
    print("Features:")
    if summary["categorical_pipeline"]["features"]:
        for column in summary["categorical_pipeline"]["features"]:
            print(f"- {column}")
    else:
        print("- None")

    print("Steps:")
    if summary["categorical_pipeline"]["steps"]:
        for step in summary["categorical_pipeline"]["steps"]:
            print(f"- {step}")
    else:
        print("- None")

    print("\nLeakage note:")
    print(f"- {summary['leakage_note']}")

def print_model_registry_summary(model_registry_summary):
    """
    Print the approved model registry summary.
    """

    print("\nModel registry created.")
    print("-" * 60)

    if not model_registry_summary:
        print("- No candidate models available.")
        return

    for index, model in enumerate(model_registry_summary, start=1):
        baseline_text = "Yes" if model["is_dummy_baseline"] else "No"

        print(f"\n{index}. {model['display_name']}")
        print(f"Model ID: {model['model_id']}")
        print(f"Family: {model['family']}")
        print(f"Complexity: {model['complexity']}")
        print(f"Interpretability: {model['interpretability']}")
        print(f"Dummy baseline: {baseline_text}")
        print(f"Supports predict_proba: {model['supports_predict_proba']}")

        print("Strengths:")
        for item in model["strengths"]:
            print(f"- {item}")

        print("Limitations:")
        for item in model["limitations"]:
            print(f"- {item}")

def print_training_results(model_results, training_summary):
    """
    Print baseline model training results.
    """

    print("\nBaseline model training completed.")
    print("-" * 60)

    if not training_summary:
        print("- No training summary available.")
        return

    print(f"Task type: {training_summary['task_type']}")
    print(f"Validation method: {training_summary['validation_method']}")
    print(f"CV folds: {training_summary['cv_folds']}")
    print(f"Primary metric: {training_summary['primary_metric']}")
    print(f"Metric direction: {training_summary['primary_metric_direction']}")
    print(f"Models attempted: {training_summary['models_attempted']}")
    print(f"Models completed: {training_summary['models_completed']}")
    print(f"Models failed: {training_summary['models_failed']}")

    if training_summary["unsupported_metrics_in_first_trainer"]:
        print("\nMetrics planned but not yet implemented in this trainer:")
        for metric in training_summary["unsupported_metrics_in_first_trainer"]:
            print(f"- {metric}")

    print("\nLeakage control:")
    print(f"- {training_summary['leakage_control']}")

    print("\nModel results:")
    if not model_results:
        print("- No model results available.")
        return

    for result in model_results:
        print("\n" + result["display_name"])
        print("-" * 40)
        print(f"Model ID: {result['model_id']}")
        print(f"Family: {result['family']}")
        print(f"Status: {result['status']}")
        print(f"Primary metric: {result['primary_metric']}")
        print(f"Primary score: {result['primary_score']}")
        print(f"Fit time mean: {result['fit_time_mean']}")
        print(f"Score time mean: {result['score_time_mean']}")

        if result["error"]:
            print(f"Error: {result['error']}")
            continue

        print("Metrics:")

        for metric_name, metric_values in result["metrics"].items():
            print(
                f"- {metric_name}: "
                f"cv_mean={metric_values['cv_mean']}, "
                f"cv_std={metric_values['cv_std']}, "
                f"train_mean={metric_values['train_mean']}"
            )

            diagnostics = result.get("classification_diagnostics", {})

            if diagnostics:
                print("\nClassification diagnostics:")

                if diagnostics.get("positive_label") is not None:
                    print(f"Positive label: {diagnostics['positive_label']}")

                if diagnostics.get("out_of_fold_roc_auc") is not None:
                    print(f"Out-of-fold ROC-AUC: {diagnostics['out_of_fold_roc_auc']}")

                if diagnostics.get("out_of_fold_pr_auc") is not None:
                    print(f"Out-of-fold PR-AUC: {diagnostics['out_of_fold_pr_auc']}")

                print("Labels:")
                for label in diagnostics.get("labels", []):
                    print(f"- {label}")

                print("Confusion matrix:")
                matrix = diagnostics.get("confusion_matrix", [])

                if matrix:
                    for row in matrix:
                        print(f"- {row}")
                else:
                    print("- Not available")

                print("Class-level metrics:")
                for label, values in diagnostics.get("class_level_metrics", {}).items():
                    print(
                        f"- {label}: "
                        f"precision={values['precision']}, "
                        f"recall={values['recall']}, "
                        f"f1={values['f1_score']}, "
                        f"support={values['support']}"
                )

def print_model_leaderboard(leaderboard, comparison_summary):
    """
    Print model leaderboard and comparison summary.
    """

    print("\nModel comparison and leaderboard completed.")
    print("-" * 60)

    if not leaderboard:
        print("- No leaderboard available.")
        return

    print(f"Primary metric: {comparison_summary['primary_metric']}")
    print(f"Metric direction: {comparison_summary['primary_metric_direction']}")
    print(f"Dummy baseline score: {comparison_summary['dummy_baseline_score']}")
    print(f"Models ranked: {comparison_summary['models_ranked']}")
    print(f"Recommendation status: {comparison_summary['recommendation_status']}")

    print("\nImportant notes:")
    if comparison_summary["important_notes"]:
        for note in comparison_summary["important_notes"]:
            print(f"- {note}")
    else:
        print("- None")

    print("\nLeaderboard:")

    for item in leaderboard:
        print("\n" + f"Rank {item['rank']}: {item['display_name']}")
        print("-" * 40)
        print(f"Model ID: {item['model_id']}")
        print(f"Family: {item['family']}")
        print(f"Primary score: {item['primary_score']}")
        print(f"CV std: {item['primary_cv_std']}")
        print(f"Train score: {item['primary_train_score']}")
        print(f"Overfit gap: {item['overfit_gap']}")
        print(f"Reliability level: {item['reliability_level']}")
        print(f"Dummy baseline: {item['is_dummy_baseline']}")
        print(f"Beats dummy: {item['improvement_over_dummy']['beats_dummy']}")
        print(f"Ties dummy: {item['improvement_over_dummy']['ties_dummy']}")
        print(f"Raw improvement over dummy: {item['improvement_over_dummy']['raw_improvement']}")

        print("Reliability flags:")
        if item["reliability_flags"]:
            for flag in item["reliability_flags"]:
                print(
                    f"- [{flag['severity'].upper()}] "
                    f"{flag['issue']}: {flag['finding']}"
                )
                print(f"  Recommendation: {flag['recommendation']}")
        else:
            print("- None")

def print_critic_report(critic_report):
    """
    Print the reliability critic report.
    """

    print("\nReliability critic report completed.")
    print("-" * 60)

    if not critic_report:
        print("- No critic report available.")
        return

    print(f"Overall reliability: {critic_report['overall_reliability']}")
    print(f"Recommendation decision: {critic_report['recommendation_decision']}")
    print(f"Can proceed to tuning: {critic_report['can_proceed_to_tuning']}")

    print("\nSelected candidate:")
    if critic_report["selected_candidate"]:
        candidate = critic_report["selected_candidate"]
        print(f"- Model: {candidate['display_name']}")
        print(f"- Model ID: {candidate['model_id']}")
        print(f"- Primary metric: {candidate['primary_metric']}")
        print(f"- Primary score: {candidate['primary_score']}")
        print(f"- Reliability level: {candidate['reliability_level']}")
    else:
        print("- None")

    print("\nFinding counts:")
    for severity, count in critic_report["finding_counts"].items():
        print(f"- {severity}: {count}")

    print("\nFindings:")
    if critic_report["findings"]:
        for index, finding in enumerate(critic_report["findings"], start=1):
            print(f"\n{index}. [{finding['severity'].upper()}] {finding['issue']}")
            print(f"Evidence: {finding['evidence']}")
            print(f"Recommendation: {finding['recommendation']}")
    else:
        print("- None")

    print("\nNext actions:")
    for action in critic_report["next_actions"]:
        print(f"- {action}")

def print_final_report_summary(final_report_summary, final_report_path):
    """
    Print final report generation summary.
    """

    print("\nFinal report created.")
    print("-" * 60)

    if not final_report_summary:
        print("- No final report summary available.")
        return

    print(f"Report path: {final_report_path}")
    print(f"Report type: {final_report_summary['report_type']}")
    print(f"Dataset path: {final_report_summary['dataset_path']}")
    print(f"Target column: {final_report_summary['target_column']}")
    print(f"Task type: {final_report_summary['task_type']}")
    print(f"Overall reliability: {final_report_summary['overall_reliability']}")
    print(f"Recommendation decision: {final_report_summary['recommendation_decision']}")
    print(f"Can proceed to tuning: {final_report_summary['can_proceed_to_tuning']}")
    print(f"Best overall model: {final_report_summary['best_overall_model']}")
    print(f"Best useful model: {final_report_summary['best_useful_model']}")
    print(f"Models trained: {final_report_summary['models_trained']}")

def print_orchestrator_summary(orchestrator_result):
    """
    Print the orchestrator execution summary.
    """

    print("\nOrchestrator summary.")
    print("-" * 60)

    print(f"Success: {orchestrator_result['success']}")
    print(f"Stop reason: {orchestrator_result['stop_reason']}")

    if orchestrator_result.get("error"):
        print(f"Error: {orchestrator_result['error']}")

    print("\nActions taken:")

    for item in orchestrator_result["actions_taken"]:
        print(
            f"- Step {item['step_number']}: "
            f"{item['selected_action']} "
            f"({item['state_status_before']} -> "
            f"{item.get('state_status_after', 'not run')})"
        )