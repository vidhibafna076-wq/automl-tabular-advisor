from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
from src.state import ExperimentState


@dataclass(frozen=True)
class RunPaths:
    """
    Stores the output locations belonging to one experiment run.

    Every report, state file and approved model artifact created during
    the run must remain under the same run directory.
    """

    run_id: str
    run_directory: Path
    state_path: Path
    report_path: Path
    model_directory: Path

    def to_dict(self) -> dict[str, str]:
        """
        Convert run paths into JSON-safe strings.
        """

        return {
            "run_id": self.run_id,
            "run_directory": str(self.run_directory),
            "state_path": str(self.state_path),
            "report_path": str(self.report_path),
            "model_directory": str(self.model_directory),
        }


def _normalise_unique_suffix(value: str) -> str:
    """
    Convert a supplied suffix into a filesystem-safe identifier.
    """

    cleaned_value = "".join(
        character.lower()
        for character in value
        if character.isalnum()
    )

    if not cleaned_value:
        raise ValueError(
            "Run suffix must contain at least one letter or number."
        )

    return cleaned_value[:16]


def create_run_paths(
    output_root: str | Path = "outputs/runs",
    created_at: datetime | None = None,
    unique_suffix: str | None = None,
) -> RunPaths:
    """
    Create an isolated output directory for one experiment.

    Example:

    outputs/runs/20260726T071530_123456Z_a1b2c3d4/
        experiment_state.json
        final_report.md
        models/

    The directory is created with exist_ok=False so an existing run can
    never be silently overwritten.
    """

    run_time = created_at or datetime.now(timezone.utc)

    if run_time.tzinfo is None:
        run_time = run_time.replace(
            tzinfo=timezone.utc,
        )

    utc_run_time = run_time.astimezone(
        timezone.utc,
    )

    timestamp = utc_run_time.strftime(
        "%Y%m%dT%H%M%S_%fZ"
    )

    suffix = _normalise_unique_suffix(
        unique_suffix or uuid4().hex[:8]
    )

    run_id = f"{timestamp}_{suffix}"

    run_directory = Path(output_root) / run_id
    model_directory = run_directory / "models"

    # Never reuse an existing experiment directory.
    run_directory.mkdir(
        parents=True,
        exist_ok=False,
    )

    model_directory.mkdir(
        parents=False,
        exist_ok=False,
    )

    return RunPaths(
        run_id=run_id,
        run_directory=run_directory,
        state_path=(
            run_directory
            / "experiment_state.json"
        ),
        report_path=(
            run_directory
            / "final_report.md"
        ),
        model_directory=model_directory,
    )

def apply_run_paths_to_state(
    state: ExperimentState,
    run_paths: RunPaths,
) -> ExperimentState:
    """
    Attach one isolated run's output paths to ExperimentState.

    Path objects are converted to strings so the complete state remains
    JSON serializable.
    """

    path_metadata = run_paths.to_dict()

    state.run_id = path_metadata["run_id"]
    state.run_directory = path_metadata[
        "run_directory"
    ]
    state.state_path = path_metadata[
        "state_path"
    ]
    state.report_path = path_metadata[
        "report_path"
    ]
    state.model_directory = path_metadata[
        "model_directory"
    ]

    return state