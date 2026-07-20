# Agentic AutoML Advisor

A transparent, controlled AutoML advisor for structured CSV datasets.

The project inspects tabular data, detects the prediction task, creates a preprocessing and validation plan, trains defensible baseline models, compares them with a dummy baseline, critiques the reliability of the evidence, and produces a final experiment report.

The system prioritises caution over false confidence. It can refuse to recommend or save a model when the available evidence is weak.

## User experience

The Streamlit application uses a compact agentic command-centre interface:

- central CSV upload or built-in sample dataset
- objective and target-column selection
- compact dataset summary with an optional preview
- real progress updates from the ten orchestrator steps
- separate results screen so the upload page does not become excessively long
- decision-first results with supporting Data, Models, Reliability and Report tabs
- downloadable Markdown report
- responsive dark interface for desktop and smaller screens

## Controlled agent workflow

```text
Inspect -> Plan -> Prepare -> Pipeline -> Registry
        -> Train -> Compare -> Critique -> Persist -> Report
```

The orchestrator selects the next permitted action from the experiment state. It does not allow an uncontrolled agent to execute arbitrary tools.

## Current capabilities

- CSV datasets
- binary classification
- multiclass classification
- regression
- automatic task detection
- data profiling and quality checks
- leakage warnings
- preprocessing configuration and pipeline creation
- cross-validated baseline training
- dummy-baseline comparison
- reliability criticism
- conditional model persistence
- final Markdown report and JSON experiment state
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

The application opens in your default browser. Upload a CSV, select the column you want to predict, describe the objective, and start the experiment.

## Run from the command line

```powershell
python main.py `
  --file data/sample/loan_sample.csv `
  --target Loan_Status `
  --objective "Predict whether a loan application will be approved" `
  --approve-drop-id-columns
```

## Generated outputs

Completed runs can create:

- `outputs/reports/experiment_state.json`
- `outputs/reports/final_report.md`
- an approved model artifact under `outputs/models/`
- model metadata and feature-importance information when persistence is approved

A model is not saved automatically merely because it ranks first. The reliability critic must approve the evidence first.

## Project structure

```text
app.py                         Streamlit interface
main.py                        Command-line interface
data/sample/                   Small demonstration dataset
src/agents/                    Planner, critic and orchestrator
src/tools/                     Controlled workflow tools
src/data_loader.py             CSV loading and validation
src/data_profiler.py           Dataset profiling
src/data_quality.py            Quality analysis
src/preprocessing.py           Preprocessing construction
src/model_registry.py          Candidate model selection
src/training.py                Cross-validated baseline training
src/comparison.py              Leaderboard and dummy comparison
src/reporting.py               Report generation
src/state.py                   Shared experiment state
outputs/                       Generated reports and approved artifacts
```

## Important limitations

This is a learning-focused and portfolio-focused prototype, not a production AutoML platform.

- Long-running jobs still run inside the Streamlit process.
- A browser Stop button cannot safely interrupt model fitting yet.
- There is no authentication, multi-user isolation or cloud job queue.
- The current workflow focuses on structured tabular data.
- A recommendation is a baseline modelling decision, not a production-deployment guarantee.

The next production-oriented step would be to move experiment execution into a background worker or API so runs can be cancelled, monitored and resumed independently of the browser sessi