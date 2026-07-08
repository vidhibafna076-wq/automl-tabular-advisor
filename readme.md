# Agentic AutoML Advisor

A transparent, controlled, agentic AutoML advisor for structured CSV datasets.

This project is a learning-focused and portfolio-focused AutoML prototype. It helps users inspect a tabular dataset, detect the machine learning task, build a safe baseline modelling workflow, compare models against a dummy baseline, critique the reliability of the results, and generate a final experiment report.

It is not a production AutoML platform. The current system is best understood as a cautious baseline ML advisor that explains what it is doing and refuses to recommend a model when the evidence is weak.

---

## Problem Statement

Many AutoML tools focus on producing a "best model" quickly, but they often hide important modelling decisions from the user.

This project takes a different approach.

The goal is to build an AutoML-style advisor that helps users understand:

- what kind of prediction task they have
- what data quality issues exist
- what preprocessing decisions were made
- which baseline models were tested
- whether any model actually beats a dummy baseline
- whether the results are reliable enough to trust
- why the system recommends or refuses to recommend a model

The project prioritises transparency and caution over false confidence.

---

## Who This Helps

This project is designed for:

- students learning applied machine learning workflows
- early data analysts or data scientists who want guidance on tabular modelling
- users who want an explainable baseline before trying advanced tuning
- portfolio reviewers who want to see ML engineering discipline
- non-experts who need a readable report explaining model reliability

---

## What "Agentic" Means In This Project

This project does not use an uncontrolled LLM agent that randomly decides what to do.

Here, "agentic" means a controlled workflow:

```text
Observe -> Plan -> Run approved tools -> Inspect results -> Critique -> Report