import argparse

from src.agents.orchestrator_agent import run_orchestrator
from src.display import (
    print_final_report_summary,
    print_orchestrator_summary,
    print_system_warnings,
    print_tool_history,
)
from src.state import ExperimentState, save_state


def main() -> int:
    """
    Run the Agentic AutoML Advisor CLI.

    Exit codes:
    - 0: The workflow completed successfully, including valid outcomes where
         the critic recommends caution or refuses to recommend a model.
    - 1: The workflow encountered an internal execution failure.
    - 2: Invalid command-line arguments, handled automatically by argparse.
    """

    parser = argparse.ArgumentParser(
        description="Agentic AutoML Advisor - Orchestrated Workflow"
    )

    parser.add_argument(
        "--file",
        required=True,
        help="Path to the CSV dataset.",
    )

    parser.add_argument(
        "--target",
        required=True,
        help="Name of the target column.",
    )

    parser.add_argument(
        "--objective",
        required=False,
        default=None,
        help=(
            "Optional user objective, for example: "
            "'Predict customer churn'."
        ),
    )

    parser.add_argument(
        "--approve-drop-id-columns",
        action="store_true",
        help=(
            "Approve dropping possible ID columns detected "
            "during inspection."
        ),
    )

    args = parser.parse_args()

    state = ExperimentState(
        dataset_path=args.file,
        target_column=args.target,
        user_objective=args.objective,
    )

    state.completed_steps.append(
        "created_experiment_state"
    )

    orchestrator_result = run_orchestrator(
        state=state,
        approve_drop_id_columns=(
            args.approve_drop_id_columns
        ),
    )

    state = orchestrator_result["state"]

    save_state(
        state,
        "outputs/reports/experiment_state.json",
    )

    print_orchestrator_summary(
        orchestrator_result
    )

    if state.final_report_summary:
        print_final_report_summary(
            state.final_report_summary,
            state.final_report_path,
        )

    print_tool_history(
        state.tool_history
    )
    print_system_warnings(
        state.warnings
    )

    print(
        "\nState updated and saved to "
        "outputs/reports/experiment_state.json"
    )
    print(
        "Current status:",
        state.status,
    )

    if orchestrator_result.get(
        "success",
        False,
    ):
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())