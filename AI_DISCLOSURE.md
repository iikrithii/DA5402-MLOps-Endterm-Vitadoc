# AI Tool Usage Disclosure

**Project:** VitaDoc - Blood Test Analysis Platform (DA5402 MLOps End-Term)  
Krithi Shailya, DA25S009

This document provides a detailed account of every AI tool used in this
project, what specifically it was used for. AI tools were used, in a limited and specific capacity, and
this document accounts for that honestly.

---

## Tools Used

| Tool | Model / Version | Role |
|---|---|---|
| Claude (Anthropic) | Claude Sonnet | Documentation drafting, README structure, boilerplate implementation and debugging|
| ChatGPT (OpenAI) | GPT-4o | Logic scaffolding and debugging |
| GitHub Copilot | GPT-4 based | Inline code completions and comment generation while typing |

---

## Guiding Principle

The rule I applied throughout this project was simple: **AI can write
what I have already decided.** It was not used to make a design
decision. 

In practice, this meant AI was useful for about the same things a google search find and paste from stack overflow/online forums would be: turning a clear idea into syntactically correct code, filling in docstrings once a function was already written, and debugging integration errors. 

---

## Detailed Usage

### 1. GitHub Copilot Inline Completions

Copilot runs continuously in the editor and offers single-line
or short-block completions as you type. The way I used it:

**Comments:** When I typed a `#` comment describing what the next line
does, Copilot would suggest the line. I accepted these when they matched
my intent and ignored them when they did not. This is not meaningfully
different from autocomplete the decision of what to write was already
made.

**Repetitive patterns:** In places like the Prometheus metric
registrations in `backend/main.py`, once I had written the first two or
three `Counter(...)` / `Gauge(...)` declarations myself, Copilot would
correctly infer the pattern for the next one. I reviewed each suggestion
before accepting.

**Function signatures:** When defining new functions in `src/features/`
and `src/models/`, Copilot would sometimes correctly suggest the
argument list after I had typed the function name. These were accepted
when correct, overridden when not.

---

### 2. Claude Documentation and README

Claude and Claude Code was used to help produce the boilerplate and the architecture documents, which I then manually changed by myself, after getting the overall layout. I also used it while debugging docker compose errors due to hardware limitations/version mismatches that were not available online. 

---

### 3. ChatGPT Occasional Logic Scaffolding

ChatGPT was used in a number of cases where I had a clear
algorithmic idea but wanted a starting-point implementation to
react to rather than writing from scratch. 


## What AI Was Not Used For

### System Architecture
The decision to integrate FastAPI + MLflow Registry + DVC + Airflow +
Prometheus + Grafana as a unified stack and specifically how each
service communicates with every other was entirely my own. The
architecture diagram reflects decisions I made iteratively over the
course of the project. No AI tool was consulted for architecture.

### Docker & Container Configuration
Every `Dockerfile`, `docker-compose.yml`, `docker-compose.airflow.yml`,
`docker-compose.mlflow.yml`, and `docker-compose.override.yml` was
configured by hand with a little completion help from templates and Copilot. 

### DVC Pipeline Design
The `dvc.yaml` stage graph is entirely my own. The decisions about
which stages to separate (e.g., `preprocess_ckd` and
`preprocess_thyroid` as independent stages to enable parallelism),
which outputs to cache vs not cache, how `params.yaml` feeds into
stage commands, and how `dvc.lock` was used for reproducibility were
all made by me. DVC's behaviour especially around cache invalidation
and the interaction with Git required careful reading of
documentation. 

### Airflow DAG Architecture
The stages, logics and pool integration of all dags was designed by me. AI was used to write down some of the functions and optimize the code. The `vitadoc_pool` slot design,
the `TASK_SLOTS` dynamic mapping approach in the data pipeline DAG,
the `_run()` subprocess isolation pattern, and the two-condition
trigger logic in the retraining DAG (drift threshold AND feedback
rate, checked independently) are all my design. The retraining DAG and chaining `dvc repro` into the Airflow task ran into a lot of integration and permission errors, whcih were resolved through documentation and AI. 

### MLflow Integration
Deciding what to log beyond autolog was entirely my own judgement.
Per-class F1 scores, SMOTE resampling ratios, dataset version tags,
and git commit hashes were logged because I decided they mattered for
reproducibility. The `@champion` alias
pattern and how the backend loads from the registry at startup were
implemented by reading MLflow documentation.

### Prometheus Instrumentation
The choice of which metrics to expose (`drift_score` by feature,
prediction counters by condition and class, inference histograms),
the alert rule thresholds, and the Alertmanager routing configuration were all assisted by templates and also my own decisions. AI was asked for some creative help for Grafana Dashboards. The Grafana dashboard JSON was built
manually in the Grafana UI and then exported.

---

