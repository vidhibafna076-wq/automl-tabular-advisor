from pathlib import Path
import subprocess
import sys

import json
import main as main_module


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAIN_SCRIPT = PROJECT_ROOT / "main.py"
LOAN_SAMPLE = PROJECT_ROOT / "data" / "sample" / "loan_sample.csv"


def run_cli(
    *arguments: str,
    timeout_seconds: int = 180,
) -> subprocess.CompletedProcess[str]:
    """
    Run the project's CLI using the same Python interpreter as pytest.

    Using sys.executable ensures the test runs inside the active virtual
    environment rather than accidentally using another Python installation.
    """

    return subprocess.run(
        [
            sys.executable,
            str(MAIN_SCRIPT),
            *arguments,
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout_seconds,
        check=False,
    )


def combined_output(
    result: subprocess.CompletedProcess[str],
) -> str:
    """
    Combine stdout and stderr so assertion failures show all CLI evidence.
    """

    return f"{result.stdout}\n{result.stderr}"


def test_rejected_model_is_successful_cli_outcome() -> None:
    """
    A critic refusal is a valid modelling result, not a software failure.
    """

    result = run_cli(
        "--file",
        str(LOAN_SAMPLE),
        "--target",
        "Loan_Status",
        "--objective",
        "Predict whether a loan application will be approved",
        "--approve-drop-id-columns",
    )

    output = combined_output(result)

    assert result.returncode == 0, output
    assert "Success: True" in output
    assert (
        "Recommendation decision: do_not_recommend_model"
        in output
    )
    assert "Current status: final_report_created" in output
    assert "No model was saved because" in output


def test_missing_target_column_returns_exit_code_1() -> None:
    """
    An internal workflow failure must return exit code 1.
    """

    result = run_cli(
        "--file",
        str(LOAN_SAMPLE),
        "--target",
        "Missing_Target",
        "--objective",
        "Test CLI failure handling",
    )

    output = combined_output(result)

    assert result.returncode == 1, output
    assert "Success: False" in output
    assert "Stop reason: dataset_inspection_failed" in output
    assert "Current status: dataset_inspection_failed" in output
    assert "Target column 'Missing_Target' was not found" in output


def test_missing_required_argument_returns_exit_code_2() -> None:
    """
    argparse must return exit code 2 when a required argument is absent.
    """

    result = run_cli(
        "--file",
        str(LOAN_SAMPLE),
    )

    output = combined_output(result)

    assert result.returncode == 2, output
    assert (
        "the following arguments are required: --target"
        in output
    )

def test_main_creates_and_saves_run_specific_state(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """
    Every CLI invocation must save its state inside a newly created isolated
    run directory rather than the shared legacy reports directory.
    """

    dataset_path = tmp_path / "input.csv"

    dataset_path.write_text(
        "Feature,Target\n1,Yes\n2,No\n",
        encoding="utf-8",
    )

    isolated_output_root = tmp_path / "runs"

    real_create_run_paths = (
        main_module.create_run_paths
    )

    captured = {}

    def fake_create_run_paths(
        output_root="outputs/runs",
    ):
        """
        Match create_run_paths() exactly while redirecting the test output into
        pytest's temporary directory.
        """

        assert output_root == "outputs/runs"

        run_paths = real_create_run_paths(
            output_root=isolated_output_root,
            unique_suffix="cli-test",
        )

        captured["run_paths"] = run_paths

        return run_paths

    def fake_run_orchestrator(
        state,
        approve_drop_id_columns=False,
    ):
        captured["orchestrator_state"] = state
        captured["approval"] = (
            approve_drop_id_columns
        )

        state.status = "final_report_created"

        return {
            "success": True,
            "state": state,
            "message": "Synthetic successful run",
            "error": None,
        }

    monkeypatch.setattr(
        main_module,
        "create_run_paths",
        fake_create_run_paths,
    )

    monkeypatch.setattr(
        main_module,
        "run_orchestrator",
        fake_run_orchestrator,
    )

    monkeypatch.setattr(
        main_module,
        "print_orchestrator_summary",
        lambda result: None,
    )

    monkeypatch.setattr(
        main_module,
        "print_final_report_summary",
        lambda summary, path: None,
    )

    monkeypatch.setattr(
        main_module,
        "print_tool_history",
        lambda history: None,
    )

    monkeypatch.setattr(
        main_module,
        "print_system_warnings",
        lambda warnings: None,
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            "--file",
            str(dataset_path),
            "--target",
            "Target",
            "--objective",
            "Predict the target",
            "--approve-drop-id-columns",
        ],
    )

    exit_code = main_module.main()

    run_paths = captured["run_paths"]

    assert exit_code == 0

    assert (
        captured["orchestrator_state"].run_id
        == run_paths.run_id
    )

    assert (
        captured["orchestrator_state"].report_path
        == str(run_paths.report_path)
    )

    assert (
        captured[
            "orchestrator_state"
        ].model_directory
        == str(run_paths.model_directory)
    )

    assert captured["approval"] is True

    assert run_paths.state_path.is_file()

    saved_data = json.loads(
        run_paths.state_path.read_text(
            encoding="utf-8",
        )
    )

    assert saved_data["run_id"] == run_paths.run_id

    assert saved_data["run_directory"] == str(
        run_paths.run_directory
    )

    assert saved_data["state_path"] == str(
        run_paths.state_path
    )

    assert saved_data["report_path"] == str(
        run_paths.report_path
    )

    assert saved_data["model_directory"] == str(
        run_paths.model_directory
    )

    legacy_state_path = (
        tmp_path
        / "outputs"
        / "reports"
        / "experiment_state.json"
    )

    assert legacy_state_path.exists() is False