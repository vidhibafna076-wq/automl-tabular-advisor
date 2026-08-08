# Agentic AutoML Advisor

A transparent, controlled AutoML advisor for structured CSV datasets.

The application inspects tabular data, detects the prediction task, plans preprocessing and validation, trains defensible baseline models, compares them with a dummy baseline, performs guarded tuning and holdout evaluation, and reviews the strength of the evidence before recommending or saving a model.

The system prioritises caution over false confidence. A model is not recommended or persisted merely because it ranks first; the reliability review must find the available evidence strong enough.

## Current workflow

```text
Inspect -> Plan -> Prepare -> Pipeline -> Registry -> Train
        -> Compare -> Tune -> Holdout -> Critique -> Persist -> Report
```

This is a 12-stage controlled evaluation workflow. The orchestrator chooses the next permitted action from the experiment state and invokes a fixed set of tools. It does not allow an autonomous agent to execute arbitrary operations.

## User experience

The Streamlit application provides:

- CSV upload and a built-in sample dataset
- objective, target-column and task-type controls
- dataset preview and target-distribution summary
- background experiment execution with progress updates
- per-session and per-run artifact isolation
- configurable runtime limits and whole-run termination
- explicit approval or rejection of proposed preprocessing decisions
- resumable experiments after approval
- evidence-first results covering data, models, tuning, holdout evaluation, reliability and reporting
- reviewed-model discovery and batch CSV prediction
- downloadable reports and prediction results
- a responsive interface for desktop and smaller screens

## Current capabilities

- CSV datasets
- binary classification
- multiclass classification
- regression
- automatic task detection with a user override
- data profiling and quality checks
- leakage warnings
- preprocessing planning and pipeline construction
- cross-validated baseline training
- dummy-baseline comparison
- guarded Optuna hyperparameter tuning
- final holdout evaluation
- reliability criticism and cautious model-selection decisions
- conditional model persistence
- saved-model runtime compatibility validation
- feature-importance metadata when supported
- Markdown reports and JSON experiment state
- dataset fingerprints, software versions, Git revision and resumed-run lineage
- batch inference from reviewed saved pipelines
- Streamlit and command-line interfaces

## Installation

Create and activate a virtual environment before installing the dependencies.

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run the Streamlit interface

```powershell
streamlit run app.py
```

Upload a CSV or select the sample dataset, choose the target column, describe the objective, review the inferred task, and start the experiment.

Streamlit experiments run in spawned child processes. The browser remains responsive while a run is active and can terminate the complete experiment process. Termination does not currently skip an individual model while allowing the remainder of the run to continue.

## Run from the command line

```powershell
python main.py `
  --file data/sample/loan_sample.csv `
  --target Loan_Status `
  --objective "Predict whether a loan application will be approved" `
  --approve-drop-id-columns
```

The CLI creates an isolated directory under `outputs/runs/` and prints the paths to the run directory and saved experiment state.

## Generate predictions

```powershell
python predict.py `
  --model outputs/runs/<run-id>/models/<model>_pipeline.joblib `
  --input data/sample/inference_demo.csv
```

Only load model artifacts from trusted sources. Joblib model files can execute code during deserialization.

## Generated outputs

CLI experiments create isolated output directories using this layout:

```text
outputs/runs/<run-id>/
|-- experiment_state.json
|-- final_report.md
`-- models/
    |-- <model>_pipeline.joblib
    `-- <model>_metadata.json
```

Streamlit experiments use session-scoped storage:

```text
outputs/sessions/<session-id>/runs/<run-id>/
```

A run might not contain a saved model. Model persistence occurs only when the reliability critic approves the available evidence.

## Project structure

```text
app.py                         Streamlit entry point
main.py                        Command-line experiment runner
predict.py                     Command-line batch inference
data/sample/                   Small demonstration datasets
src/agents/                    Planner, critic and orchestrator
src/tools/                     Controlled workflow tools
src/ui/                        Streamlit interface and run-time management
src/data_loader.py             CSV loading and validation
src/data_profiler.py           Dataset profiling
src/data_quality.py            Data-quality analysis
src/preprocessing.py           Preprocessing construction
src/model_registry.py          Candidate model selection
src/training.py                Cross-validated model training
src/comparison.py              Leaderboard and dummy comparison
src/inference.py               Saved-pipeline inference
src/reporting.py               Report generation
src/run_context.py             Isolated run paths
src/state.py                   Shared experiment state
tests/                         Regression and workflow tests
outputs/                       Generated run artifacts
```

## Testing

```powershell
pip install -r requirements-dev.txt
python -m pytest -q
```

Pytest stores its temporary files and cache under `outputs/`, which keeps the
test suite usable in environments where the system temporary directory is
restricted.

## Important limitations

This is a learning-focused and portfolio-focused prototype, not a production AutoML platform.

- There is no authentication or account-level authorization.
- Session tokens provide run isolation but are not a security boundary.
- Artifacts and uploaded data are stored on the local filesystem.
- There is no durable job queue, external worker service or database-backed run registry.
- Whole runs can be terminated, but individual model fits cannot be cooperatively skipped.
- The workflow currently focuses on structured CSV data.
- Validation defaults cannot account for unknown time, group or causal structure unless that context is explicitly modelled.
- A recommendation is a baseline modelling decision, not evidence that a model is ready for production deployment.
- Saved model artifacts must only be loaded from trusted sources.

Before a public multi-user deployment, add authenticated identities, durable private storage, retention controls, resource quotas, monitoring and a server-side job queue.
