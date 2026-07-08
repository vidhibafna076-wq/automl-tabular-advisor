from dataclasses import asdict
from datetime import datetime
from typing import Any

from src.data_loader import load_dataset
from src.data_profiler import profile_dataset
from src.data_quality import generate_quality_report
from src.state import ExperimentState
from src.task_detector import detect_task_type


def _add_tool_event(
    state: ExperimentState,
    tool_name: str,
    status: str,
    message: str,
) -> None:
    """
    Add a factual event to the experiment's tool history.

    This is useful for showing an audit timeline later.
    """

    state.tool_history.append(
        {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "tool_name": tool_name,
            "status": status,
            "message": message,
        }
    )


def inspect_dataset_tool(state: ExperimentState) -> dict[str, Any]:
    """
    Approved tool for the dataset observation stage.

    This tool performs the first major AutoML inspection workflow:
    1. Load the dataset
    2. Detect the task type
    3. Profile the dataset
    4. Generate data quality warnings
    5. Update the shared experiment state

    The tool does not train models.
    The tool does not modify the dataset.
    The tool only observes and records evidence.
    """

    tool_name = "inspect_dataset_tool"
    state.iteration += 1

    _add_tool_event(
        state=state,
        tool_name=tool_name,
        status="started",
        message="Started dataset inspection.",
    )

    try:
        # Step 1: Load dataset
        dataset = load_dataset(
            file_path=state.dataset_path,
            target_column=state.target_column,
        )

        state.status = "dataset_loaded"
        state.completed_steps.append("loaded_dataset")

        _add_tool_event(
            state=state,
            tool_name=tool_name,
            status="success",
            message=(
                f"Loaded dataset with {dataset.metadata['rows']} rows and "
                f"{dataset.metadata['columns']} columns."
            ),
        )

        if dataset.metadata["target_missing_count"] > 0:
            state.warnings.append(
                f"Target column has {dataset.metadata['target_missing_count']} "
                "missing values. Rows with missing target values should be "
                "excluded before model training."
            )

        if dataset.metadata["removed_empty_columns"]:
            state.warnings.append(
                "Removed completely empty columns: "
                + ", ".join(dataset.metadata["removed_empty_columns"])
            )

        # Step 2: Detect task type
        task_info = detect_task_type(dataset.y)

        state.task_type = task_info.task_type
        state.base_task = task_info.base_task
        state.task_reason = task_info.reason
        state.target_unique_count = task_info.unique_count
        state.target_missing_count = task_info.missing_count
        state.target_summary = task_info.target_summary
        state.status = "task_detected"
        state.completed_steps.append("detected_task_type")

        _add_tool_event(
            state=state,
            tool_name=tool_name,
            status="success",
            message=f"Detected task type: {task_info.task_type}.",
        )

        # Step 3: Profile dataset
        profile = profile_dataset(
            df=dataset.df,
            target_column=state.target_column,
        )

        state.profile = profile
        state.status = "profile_completed"
        state.completed_steps.append("profiled_dataset")

        _add_tool_event(
            state=state,
            tool_name=tool_name,
            status="success",
            message="Generated dataset profile.",
        )

        # Step 4: Generate quality report
        quality_issues = generate_quality_report(
            profile=profile,
            task_info=task_info,
        )

        state.quality_issues = quality_issues
        state.status = "quality_report_completed"
        state.completed_steps.append("generated_quality_report")

        _add_tool_event(
            state=state,
            tool_name=tool_name,
            status="success",
            message=f"Generated data quality report with {len(quality_issues)} issue(s).",
        )

        # Step 5: Final status for this tool
        state.status = "dataset_inspection_completed"

        _add_tool_event(
            state=state,
            tool_name=tool_name,
            status="completed",
            message="Dataset inspection completed successfully.",
        )

        return {
            "success": True,
            "state": state,
            "dataset_metadata": dataset.metadata,
            "task_info": asdict(task_info),
            "profile": profile,
            "quality_issues": quality_issues,
            "error": None,
        }

    except Exception as error:
        state.status = "dataset_inspection_failed"
        state.warnings.append(str(error))
        state.completed_steps.append("failed_dataset_inspection")

        _add_tool_event(
            state=state,
            tool_name=tool_name,
            status="failed",
            message=str(error),
        )

        return {
            "success": False,
            "state": state,
            "dataset_metadata": None,
            "task_info": None,
            "profile": None,
            "quality_issues": [],
            "error": str(error),
        }