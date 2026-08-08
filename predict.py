import argparse
import json
import sys

from src.inference import predict_csv


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments for batch prediction.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Generate predictions using a saved Agentic AutoML "
            "Advisor pipeline."
        )
    )

    parser.add_argument(
        "--model",
        required=True,
        help="Path to the saved *_pipeline.joblib file.",
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to the CSV file containing new prediction rows.",
    )

    parser.add_argument(
        "--metadata",
        default=None,
        help=(
            "Optional path to the model metadata JSON. "
            "When omitted, it is inferred from the model filename."
        ),
    )

    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Optional prediction output CSV path. "
            "A timestamped path is generated when omitted."
        ),
    )

    return parser.parse_args()


def main() -> None:
    """
    Run safe batch prediction and print a concise execution summary.
    """

    arguments = parse_arguments()

    try:
        result = predict_csv(
            model_path=arguments.model,
            input_path=arguments.input,
            metadata_path=arguments.metadata,
            output_path=arguments.output,
        )

        print("")
        print("Prediction summary.")
        print("-" * 60)
        print(
            f"Success: {result['success']}"
        )
        print(
            f"Model: {result['display_name']}"
        )
        print(
            f"Model ID: {result['model_id']}"
        )
        print(
            f"Task type: {result['task_type']}"
        )
        print(
            f"Rows predicted: {result['rows_predicted']}"
        )
        print(
            f"Tuning applied: {result['tuning_applied']}"
        )
        print(
            f"Output path: {result['output_path']}"
        )

        compatibility = result.get(
            "artifact_compatibility",
            {},
        )
        print(
            "Artifact compatibility: "
            f"{compatibility.get('status', 'unknown')}"
        )

        for compatibility_warning in compatibility.get("warnings", []):
            print(f"Compatibility warning: {compatibility_warning}")

        ignored_columns = result.get(
            "ignored_input_columns",
            [],
        )

        if ignored_columns:
            print(
                "Ignored extra columns: "
                + ", ".join(ignored_columns)
            )

        probability_columns = result.get(
            "probability_columns",
            [],
        )

        if probability_columns:
            print(
                "Probability columns: "
                + ", ".join(probability_columns)
            )

        print("")
        print("Full prediction metadata.")
        print("-" * 60)
        print(
            json.dumps(
                result,
                indent=4,
            )
        )

    except Exception as error:
        print("")
        print("Prediction failed.")
        print("-" * 60)
        print(str(error))
        sys.exit(1)


if __name__ == "__main__":
    main()
