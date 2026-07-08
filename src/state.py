from dataclasses import dataclass, field, asdict
from typing import Any
import json
from pathlib import Path


@dataclass
class ExperimentState:
    """
    Stores the current state of one AutoML experiment.

    In an agentic system, state is the shared memory.
    Every tool and agent reads from and updates this object.
    """

    dataset_path: str
    target_column: str
    user_objective: str | None = None

    # Dataset understanding
    # Dataset understanding
    task_type: str | None = None
    base_task: str | None = None
    task_reason: str | None = None
    target_unique_count: int | None = None
    target_missing_count: int | None = None
    target_summary: dict[str, Any] = field(default_factory=dict)

    # Data inspection outputs
    profile: dict[str, Any] = field(default_factory=dict)
    quality_issues: list[dict[str, Any]] = field(default_factory=list)

    # Agent planning and execution
    approved_actions: list[str] = field(default_factory=list)
    pending_approvals: list[dict[str, Any]] = field(default_factory=list)
    experiment_plan: list[dict[str, Any]] = field(default_factory=list)
    preprocessing_config: dict[str, Any] = field(default_factory=dict)
    preprocessing_pipeline_summary: dict[str, Any] = field(default_factory=dict)
    completed_steps: list[str] = field(default_factory=list)
    tool_history: list[dict[str, Any]] = field(default_factory=list)

    # Model experiment outputs
    model_registry_summary: list[dict[str, Any]] = field(default_factory=list)
    model_results: list[dict[str, Any]] = field(default_factory=list)
    training_summary: dict[str, Any] = field(default_factory=dict)
    leaderboard: list[dict[str, Any]] = field(default_factory=list)
    comparison_summary: dict[str, Any] = field(default_factory=dict)
    critic_report: dict[str, Any] = field(default_factory=dict)
    final_report_path: str | None = None
    final_report_summary: dict[str, Any] = field(default_factory=dict)

    # System messages
    warnings: list[str] = field(default_factory=list)

    # Lifecycle tracking
    iteration: int = 0
    status: str = "created"


def state_to_dict(state: ExperimentState) -> dict[str, Any]:
    """
    Convert ExperimentState into a normal dictionary.
    This helps us save it as JSON later.
    """
    return asdict(state)


def save_state(state: ExperimentState, output_path: str) -> None:
    """
    Save the current experiment state as a JSON file.

    This makes the experiment reproducible because we can see:
    - what dataset was used
    - what target was selected
    - what was detected
    - what warnings were produced
    - which steps were completed
    """

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as file:
        json.dump(state_to_dict(state), file, indent=4)


def load_state(input_path: str) -> ExperimentState:
    """
    Load a previously saved experiment state from JSON.
    """

    with open(input_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    return ExperimentState(**data)