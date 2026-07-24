from pathlib import Path
import subprocess
import sys


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