# Agentic AutoML Advisor — Trust and Workflow Revision

## Recap

The previous UI was visually polished, but it could overstate the system's
capabilities. It described workflow stages as ten agents, omitted important
tuning and holdout evidence, paused without a usable approval flow, could show
prediction output from an older uploaded file, stored artifacts in shared
directories, and ran the orchestrator synchronously.

This revision fixes the UI and execution layer while preserving the existing
backend entry points:

- `run_orchestrator(...)`
- `ExperimentState`
- `save_state(...)`
- `predict_csv(...)`

## Files in this revision

| File | Responsibility |
|---|---|
| `app.py` | Streamlit configuration and entry point |
| `src/ui/app_shell.py` | Navigation, setup, target preview, history, and active runs |
| `src/ui/components.py` | One accessible design system and reusable components |
| `src/ui/constants.py` | Twelve truthful workflow stages and marker compatibility |
| `src/ui/helpers.py` | Safe formatting, CSV preview cache, and artifact extraction |
| `src/ui/runtime.py` | Session/run isolation, background worker, progress, and termination |
| `src/ui/results.py` | Approval UI and evidence-first result dashboard |
| `src/ui/prediction.py` | Reviewed-model metadata, schema checks, stale-state protection, and inference |

## What changed and why

1. **Workflow truth**

   The interface now says “12-stage controlled evaluation workflow,” not “10
   agents.” Tuning and final holdout have dedicated stages. Legacy and current
   completion-marker names are both recognised.

2. **Human approval and resume**

   Every pending preprocessing decision receives an explicit approve/reject
   control. No decision is selected by default. The choices are written into
   the preprocessing artifact and the paused `ExperimentState` is resumed in a
   new run linked to the original run ID.

3. **Responsive execution and termination**

   Training runs in a spawned child process. Streamlit remains responsive,
   displays elapsed time and reported events, supports a user-selected runtime
   limit, and can terminate the exact process attached to the session.

4. **Session and run isolation**

   Each browser session receives a random token and each experiment receives a
   unique run ID. Relative backend output paths resolve inside:

   `outputs/sessions/<session-id>/runs/<run-id>/`

   The child process changes only its own working directory, so backend writes
   such as `outputs/models` and `outputs/reports` become run-specific without
   changing global process state.

5. **Efficient setup**

   Setup parses at most 5,001 rows and caches the result by file bytes. The full
   CSV is read once by the experiment worker. Uploads are capped at 150 MB and
   target distributions clearly state when they use a preview.

6. **Consequential controls**

   Identifier dropping is off by default. The user can confirm or override the
   inferred task type and sees the target distribution before training.

7. **Complete evidence story**

   Results show:

   - Cross-validation score and variation
   - Dummy-baseline comparison
   - Whether tuning occurred
   - Baseline versus tuned score
   - Selected hyperparameters
   - Final holdout score and row count
   - CV-to-holdout change
   - Reliability decision and findings
   - Class-level diagnostics
   - Expected raw prediction features

8. **Prediction correctness**

   Model choices are linked to run-specific critic metadata. Legacy pipelines
   are hidden unless the user explicitly reveals them. The selected model and
   uploaded file bytes are fingerprinted together; changing either immediately
   removes the old prediction result. Downloads have unique keys.

9. **Safer language and failures**

   The UI calls models candidates and reviews evidence instead of claiming that
   they are trusted or production-ready. Raw exceptions are written to the
   run's local error log and replaced with a short user-facing message and run
   reference.

10. **Maintainability and accessibility**

    The two competing CSS systems are replaced by one system. Essential labels
    and evidence text are at least 14 px. The sidebar is discoverable, the
    Streamlit toolbar is not forcibly hidden, and the former monolithic
    `app.py` is split by responsibility.

## Backend artifact contract

The UI accepts several legacy field names, but the clearest backend artifacts
are:

```python
state.tuning_result = {
    "tuning_applied": True,
    "baseline_score": 0.78,
    "tuned_score": 0.81,
    "best_params": {"model__C": 1.0},
    "improvement": 0.03,
}

state.holdout_evaluation = {
    "evaluated": True,
    "primary_metric": "f1",
    "primary_score": 0.79,
    "rows": 250,
}

state.model_persistence = {
    "model_path": "outputs/models/logistic_regression_pipeline.joblib",
    "tuning_applied": True,
    "tuned_parameters": {"model__C": 1.0},
}
```

If your existing backend stores these values under one of the alternative
names already supported by `helpers.py`, no backend change is required.

## Integration

1. Back up the current `app.py`.
2. Replace it with the new `app.py`.
3. Copy the complete `src/ui/` folder into the project.
4. Do not replace the existing `src/state.py`, `src/inference.py`, or
   `src/agents/` backend files.
5. From the activated virtual environment, run:

   ```powershell
   python -m compileall app.py src/ui
   streamlit run app.py
   ```

No new third-party dependency is introduced beyond packages already used by
the original application.

## Production boundary

This revision prevents accidental cross-run overwrites and stops one browser
session from listing another session's artifacts. The random URL token is not a
replacement for authentication. A public multi-user deployment should still
add authenticated user identities, durable private object storage, retention
rules, and a server-side job queue.

The supplied `run_orchestrator(...)` interface can terminate the complete child
process, but it cannot skip only one model while allowing the remaining models
to continue. That finer control requires a cooperative cancellation/skip token
inside `train_baseline_models` and is therefore a backend change, not an
`app.py` change. The current revision provides whole-run termination and an
automatic runtime limit without falsely presenting per-model cancellation.

## Manual verification checklist

1. Open the sample dataset and confirm the target distribution appears.
2. Keep identifier dropping off and run a dataset with a likely ID column.
3. If the workflow pauses, approve or reject every item and resume.
4. Confirm the progress page updates while the child worker runs.
5. Start a deliberately slow run and use **Terminate this run**.
6. Confirm completed runs appear under **This session's runs**.
7. Verify the Tuning & holdout tab matches the final report.
8. Open prediction from the decision tab.
9. Upload File A and generate predictions.
10. Upload File B without generating. Confirm File A's result disappears.
11. Switch the saved model. Confirm the previous result remains hidden.
12. Upload a file missing one expected feature and confirm prediction is blocked.

## Commit point

Commit only after the syntax check and all twelve manual checks pass:

```powershell
git add app.py src/ui UI_UPGRADE_GUIDE.md
git commit -m "fix: make AutoML workflow evidence-first and session-isolated"
```
