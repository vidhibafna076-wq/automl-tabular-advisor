from datetime import datetime
from typing import Any

from src.reporting import (
    create_final_report_markdown,
    create_final_report_summary,
    save_final_report,
)
from src.state import ExperimentState


def _add_tool_event(
    state: ExperimentState,
    tool_name: str,
    status: str,
    message: str,
) -> None:
    """
    Add a factual event to the experiment tool history.
    """

    state.tool_history.append(
        {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "tool_name": tool_name,
            "status": status,
            "message": message,
        }
    )


def final_report_tool(
    state: ExperimentState,
    output_path: str = "outputs/reports/final_report.md",
) -> dict[str, Any]:
    """
    Approved tool for creating the final AutoML advisor report.

    This tool:
    - reads the completed experiment state
    - creates a Markdown report
    - saves it to outputs/reports/final_report.md
    - stores report metadata in state
    """

    tool_name = "final_report_tool"
    state.iteration += 1

    _add_tool_event(
        state=state,
        tool_name=tool_name,
        status="started",
        message="Started final report generation.",
    )

    try:
        if state.status != "reliability_critique_completed":
            raise ValueError(
                "Cannot create final report before reliability critique is completed."
            )

        if not state.critic_report:
            raise ValueError("No critic report found in experiment state.")

        report_markdown = create_final_report_markdown(state)
        saved_path = save_final_report(
            markdown_text=report_markdown,
            output_path=output_path,
        )

        report_summary = create_final_report_summary(state)

        state.final_report_path = saved_path
        state.final_report_summary = report_summary
        state.status = "final_report_created"
        state.completed_steps.append("created_final_report")

        _add_tool_event(
            state=state,
            tool_name=tool_name,
            status="completed",
            message=f"Final report saved to {saved_path}.",
        )

        return {
            "success": True,
            "state": state,
            "final_report_path": saved_path,
            "final_report_summary": report_summary,
            "report_markdown": report_markdown,
            "error": None,
        }

    except Exception as error:
        state.status = "final_report_failed"
        state.warnings.append(str(error))
        state.completed_steps.append("failed_final_report")

        _add_tool_event(
            state=state,
            tool_name=tool_name,
            status="failed",
            message=str(error),
        )

        return {
            "success": False,
            "state": state,
            "final_report_path": None,
            "final_report_summary": {},
            "report_markdown": "",
            "error": str(error),
        }