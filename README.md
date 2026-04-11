---
title: AppLensOpenEnv
emoji: 🔍
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
license: apache-2.0
---
# AppLens OpenEnv

An OpenEnv-compliant environment that analyses any public Git repository for language, LOC, dependencies, complexity, security vulnerabilities, and modernization recommendations.

## How It Works

1. Provide a public Git repo URL (GitHub, Azure Repos, GitLab, Bitbucket).
2. The environment shallow-clones the repo, walks its source tree, and parses real manifest files.
3. Seven deterministic analysis actions run in sequence via the OpenEnv `reset()` / `step()` / `state()` interface.
4. You get a full assessment report with no external APIs — everything runs locally.

## Project Structure

```
app_lens_openenv/
├── env.py                          # OpenEnv environment (reset / step / state)
├── models.py                       # Pydantic schemas: Action, Observation, Reward
├── openenv.yaml                    # OpenEnv metadata declaration
├── requirements.txt
├── Dockerfile
├── README.md
│
├── scanner/
│   ├── repo_scanner.py             # Clones repo, walks source tree, builds app metadata
│   └── dependency_extractor.py     # Parses manifest files (requirements.txt, pom.xml, package.json, csproj, etc.)
│
├── engine/
│   ├── action_router.py            # Dispatches action names to analyzer functions
│   ├── state_manager.py            # Tracks episode state
│   └── reward_engine.py            # Incremental reward / penalty logic
│
├── analysis/
│   ├── code_analyzer.py            # detect_language, calculate_loc, generate_report
│   ├── dependency_analyzer.py      # parse_dependencies
│   ├── complexity.py               # compute_complexity
│   ├── security.py                 # security_scan (matches deps against vulnerability DB)
│   └── modernization.py            # recommend_modernization
│
├── data/
│   └── vulnerabilities.json        # Known vulnerable dependency catalog
│
└── baseline/
    └── run_analysis.py             # CLI entry point — analyses a repo by URL
```

## Analysis Actions

| # | Action | What It Does |
|---|--------|--------------|
| 1 | `detect_language` | Identifies primary language by LOC distribution across file extensions |
| 2 | `calculate_loc` | Counts non-blank lines of code per source file |
| 3 | `parse_dependencies` | Extracts dependencies from real manifest files |
| 4 | `compute_complexity` | Scores complexity from LOC, dependency count, legacy flag |
| 5 | `security_scan` | Matches dependencies against `data/vulnerabilities.json` |
| 6 | `recommend_modernization` | Produces priority, target stack, effort, and recommendations |
| 7 | `generate_report` | Consolidates all results into a single report |

## Supported Manifest Files

| Ecosystem | Files Parsed |
|-----------|-------------|
| Python | `requirements.txt`, `requirements-*.txt`, `setup.cfg`, `pyproject.toml` |
| JavaScript / TypeScript | `package.json` |
| Java / Kotlin | `pom.xml`, `build.gradle`, `build.gradle.kts` |
| .NET (C# / VB / F#) | `.csproj`, `.vbproj`, `.fsproj` |
| Go | `go.mod` |
| Ruby | `Gemfile` |
| PHP | `composer.json` |

## Task Descriptions & Expected Difficulty

AppLensOpenEnv supports 3 distinct tasks via the `TASK_NAME` environment variable:

1. **`easy`**: 
   - **Target**: A simple framework (`pallets/itsdangerous`)
   - **Difficulty**: Low
   - **Actions Required**: `detect_language`, `calculate_loc`
   - **Goal**: Just execute the most basic parsing steps.

2. **`medium`**:
   - **Target**: A mid-sized repo (`pallets/click`)
   - **Difficulty**: Medium
   - **Actions Required**: `detect_language`, `calculate_loc`, `parse_dependencies`, `compute_complexity`
   - **Goal**: Read dependencies and gauge project complexity.

3. **`hard`**:
   - **Target**: A full-sized framework (`pallets/flask`)
   - **Difficulty**: High
   - **Actions Required**: All 7 actions, including `security_scan` and `recommend_modernization`
   - **Goal**: Execute the full modernization and security pipeline without repeating or using invalid actions.

## Baseline Scores

Running `python inference.py` with Qwen/Qwen2.5-72B-Instruct evaluates the baseline agent:

- **Easy**: 1.00 (Success)
- **Medium**: 1.00 (Success)
- **Hard**: 1.00 (Success)

## OpenEnv Schemas

### Action

```python
class Action(BaseModel):
    action: str
    parameters: Dict[str, Any] = {}
```

### Observation

```python
class Observation(BaseModel):
    task_id: str
    task_level: str
    app_id: str
    step_count: int
    max_steps: int
    required_actions: List[str]
    completed_actions: List[str]
    available_actions: List[str]
    results: Dict[str, Any] = {}
```

### Reward

```python
class Reward(BaseModel):
    delta: float
    total: float
    reason: str
    penalties: List[str] = []
```

## Reward Logic

- Correct required action: `+1 / 7` (~0.1429)
- Repeated action penalty: `-0.08`
- Invalid action penalty: `-0.15`
- Cumulative reward is floored at `0.0`

## Setup

### Prerequisites

- Python 3.11+
- `git` on PATH

### Local

```bash
pip install -r requirements.txt
python baseline/run_analysis.py https://github.com/pallets/flask.git
```

### Web UI (Email + On-Page Data)

Run a simple UI that takes an email address and repository URL, then displays analysis results in the browser.

```bash
pip install -r requirements.txt
python web_ui.py
```

Then open:

```text
http://127.0.0.1:5000
```

Notes:
- Keep `Send generated report to this email` checked to email the generated report (requires SMTP vars in `.env`).
- Uncheck it if you only want to view data in the UI without sending email.

## Staged Workflow (Analyze -> Document -> Email)

Use this workflow when you want to run analysis now and send the report later.

Create a local `.env` file in the project root and add your SMTP details:

```env
GMAIL_SMTP_SERVER=smtp.gmail.com
GMAIL_SMTP_PORT=587
GMAIL_SMTP_USER=your@gmail.com
GMAIL_SMTP_APP_PASSWORD=your-16-char-app-password
```

```bash
# 1) Analyze repo and save JSON artifact
python analysis/workflow.py analyze https://github.com/pallets/flask.git

# 2) Generate polished PDF document from latest analysis
python analysis/workflow.py document

# 3) Send latest generated PDF through Gmail SMTP
# Credentials are loaded automatically from .env
python analysis/workflow.py send --to someone@example.com
```

Optional one-shot execution:

```bash
python analysis/workflow.py all https://github.com/pallets/flask.git --to someone@example.com
```

## Single Command (All-in-One)

Run analysis, create a polished PDF report, and send email in one command:

```bash
python analysis/run_all.py https://github.com/pallets/flask.git someone@example.com
```
eg: python analysis/workflow.py all https://github.com/pallets/flask.git --to dhiran@gmail.com

You can copy `.env.example` to `.env` and fill in your Gmail details.

Optional environment variables:

```bash
GMAIL_SMTP_USER=<your-gmail>
GMAIL_SMTP_APP_PASSWORD=<your-app-password>
```

PowerShell example:

```powershell
$env:GMAIL_SMTP_USER="your@gmail.com"
$env:GMAIL_SMTP_APP_PASSWORD="your-app-password"
python analysis/workflow.py all https://github.com/pallets/flask.git --to someone@example.com
```

### Docker

```bash
docker build -t app-lens-openenv .
docker run --rm app-lens-openenv python baseline/run_analysis.py https://github.com/pallets/flask.git
```

## RL Agent (PPO + HuggingFace)

This project now includes an optional reinforcement-learning agent stack under `agent/`.

```bash
# 1) Train PPO agent on fast mock env (uses DistilBERT features)
python agent/train.py --timesteps 30000 --save-path agent/models/ppo_applens

# 2) Run trained policy on real repo analysis
python agent/run_agent.py https://github.com/pallets/flask.git --model agent/models/ppo_applens
```

Notes:
- The HuggingFace model (`distilbert-base-uncased`) is downloaded on first run.
- For faster downloads/rate limits, set `HF_TOKEN` in your environment.
- A short smoke train (`--timesteps 2000`) validates plumbing but is not enough for good policy quality.

## Example Output

```
python baseline/run_analysis.py https://github.com/pallets/flask.git

  AppLens OpenEnv — Live Repo Analysis
  URL: https://github.com/pallets/flask.git

  Cloning and scanning repository...

  App ID       : flask
  Required     : detect_language, calculate_loc, parse_dependencies, ...
  Max steps    : 12

  step   1  detect_language               reward_delta=+0.1429  OK
  step   2  calculate_loc                 reward_delta=+0.1429  OK
  step   3  parse_dependencies            reward_delta=+0.1429  OK
  step   4  compute_complexity            reward_delta=+0.1429  OK
  step   5  security_scan                 reward_delta=+0.1429  OK
  step   6  recommend_modernization       reward_delta=+0.1429  OK
  step   7  generate_report               reward_delta=+0.1429  OK

  Language Detection    → python
  Lines of Code         → 14117 LOC across 83 files
  Dependencies          → 21 packages
  Complexity            → score 100, level high
  Security Scan         → 2 vulnerabilities (flask, jinja2)
  Modernization         → priority medium, target Python 3.11 + FastAPI
  Report                → consolidated summary

  SUMMARY
  Total Steps     : 7
  Reward Total    : 1.0003
  Actions Done    : detect_language, calculate_loc, parse_dependencies,
                    compute_complexity, security_scan, recommend_modernization,
                    generate_report
```
