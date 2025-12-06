# How the Judge Gets Paper Data and Uses Docker

## Overview

The paperbench judge is part of the `paperbench` package and handles the evaluation/grading phase. Here's exactly how it works:

## 1. How the Judge Gets Paper Data

### Source: Paperbench Data Directory (Git LFS)

The judge **does NOT** get paper data from HuggingFace. Instead, it uses the **paperbench data directory** which is stored using **Git LFS** in the frontier-evals repository.

**IMPORTANT**: When you install paperbench via `pip install git+...`, the LFS files are **NOT** automatically fetched. You must manually set up the data directory.

### Data Directory Structure

```
paperbench/data/papers/{paper_id}/
├── config.yaml              # Paper configuration
├── paper.md                 # Paper content (markdown)
├── paper.pdf                # Paper PDF
├── rubric.json              # Evaluation rubric (LFS)
├── addendum.md              # Additional information
├── judge.addendum.md        # Judge-specific addendum
├── blacklist.txt            # Blacklisted websites
└── assets/                  # Paper-specific assets (LFS)
    ├── image1.png
    └── data.csv
```

### Required Setup

Before running evaluation, you **must** set up the data directory:

#### Option 1: Clone and fetch LFS data (Recommended)

```bash
# Clone the frontier-evals repository
git clone https://github.com/leandermaben/frontier-evals.git --filter=blob:none
cd frontier-evals

# Fetch LFS data for paperbench
git lfs fetch --include "project/paperbench/data/**"
git lfs checkout project/paperbench/data

# Set environment variable
export PAPERBENCH_DATA_DIR="$(pwd)/project/paperbench/data"
```

#### Option 2: Use setup script

```bash
# Run the provided setup script
cd benchmarks/paperbench
./scripts/setup_data.sh
```

### How It's Accessed

When you call `grade_submission()`:

```python
from paperbench.grade import grade_submission

judge_output = grade_submission(
    submission_path="/path/to/submission",
    paper_id="paper_001",          # ← Used to look up paper data
    judge_type="default",
    grader_upload_path="/path/to/output.json",
    run_group_id="eval_run",
    runs_dir="/path/to/runs",
    run_id="instance_001",
    code_only=False,
    completer_config=None,
)
```

**Internally, the judge:**

1. **Loads paper registry**: `from paperbench.paper_registry import PaperRegistry`
2. **Gets data directory**: Uses `PAPERBENCH_DATA_DIR` environment variable or falls back to package default
3. **Fetches paper metadata**: `paper = registry.get_paper(paper_id)` reads `{data_dir}/papers/{paper_id}/config.yaml`
4. **Loads rubric**: Reads from `{data_dir}/papers/{paper_id}/rubric.json` (LFS file)
5. **Creates TaskNode tree**: Converts rubric JSON to hierarchical TaskNode structure

**Environment Variable:**
- `PAPERBENCH_DATA_DIR`: Points to the paperbench data directory (e.g., `/path/to/frontier-evals/project/paperbench/data`)
- If not set, paperbench will try to use the package installation directory (which won't have LFS files)

### Data Structure

```python
# What the judge receives from paper registry
paper = {
    "paper_id": "paper_001",
    "title": "Paper Title",
    "rubric": TaskNode(  # Hierarchical tree
        id="root",
        requirements="...",
        weight=100,
        task_category="Overall",
        sub_tasks=[
            TaskNode(id="code", ...),
            TaskNode(id="experiments", ...),
            # ... more tasks
        ]
    ),
    "paper_path": "paperbench/data/papers/paper_001/paper.md",
    "assets": ["image1.png", "data.csv"],
}
```

## 2. How the Judge Uses Docker

### Two Separate Docker Stages

#### Stage 1: Reproduction (Executes reproduce.sh)

**Container:** `reproducer.Dockerfile`

**What it does:**
1. Creates fresh Ubuntu 24.04 container
2. Mounts submission directory
3. Runs `reproduce.sh` script
4. Captures outputs and logs

**Command (simplified):**
```bash
docker run --rm \
  --gpus all \                              # GPU access
  -v /path/to/submission:/submission \      # Mount submission
  -v /path/to/output:/output \              # Mount output dir
  -w /submission \
  paperbench-reproducer:latest \
  bash reproduce.sh > /output/reproduce.log 2>&1
```

**Dockerfile (reproducer.Dockerfile):**
```dockerfile
FROM ubuntu:24.04

# Install Python 3.11 and 3.12
RUN add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y \
    python3.11 python3.11-venv python3.11-dev \
    python3.12 python3.12-venv python3.12-dev \
    build-essential git cmake libopenblas-dev

# ML libraries pre-installed
RUN apt-get install -y \
    libatlas-base-dev libblas-dev liblapack-dev

WORKDIR /submission
CMD ["/bin/bash"]
```

**Key Points:**
- Clean environment (no prior state)
- GPU access via `--gpus all`
- Agent's code executes here
- Timeout: 7 days maximum
- Pre-commit: Runs `git clean -fd` to remove untracked files

#### Stage 2: Grading (Evaluates Results)

**Container:** Can run in same or different container

**What it does:**
1. Loads paper rubric
2. Compares reproduction outputs to expected results
3. Evaluates each leaf node task
4. Aggregates scores using weights

**Not always containerized** - can run directly:
```python
# Judge evaluation (may or may not use container)
judge = create_judge(
    paper_id=paper_id,
    judge_type="default",
    config=completer_config,
)

# Evaluate the executed submission
graded_tree = judge.evaluate(
    submission_dir="/output",
    executed_submission_dir="/output/executed",
)

# Calculate final score
score = calculate_score(graded_tree)
```

### Docker Images Used

**For Inference (OpenHands):**
- Image: `paperbench-agent:latest` (or custom)
- Purpose: Run OpenHands agent
- Contains: OpenHands SDK + paperbench package
- Entrypoint: `openhands.agent_server`

**For Reproduction (Paperbench):**
- Image: Built from `reproducer.Dockerfile`
- Purpose: Execute `reproduce.sh` in clean environment
- Contains: Ubuntu 24.04 + Python + ML libraries
- Entrypoint: `/bin/bash`

**For Grading (Paperbench):**
- May use base image or run on host
- Purpose: Compare outputs to rubric
- Contains: paperbench judge code
- Access to: Original paper data + reproduction outputs

## 3. Complete Workflow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│ 1. INFERENCE PHASE (OpenHands)                              │
│    Docker: paperbench-agent:latest                          │
│    Input: Paper content (NO rubric by default)              │
│    Output: submission/{paper_id}/reproduce.sh + code        │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. REPRODUCTION PHASE (Paperbench)                          │
│    Docker: reproducer.Dockerfile                            │
│    Input: submission/{paper_id}/                            │
│    Run: bash reproduce.sh                                   │
│    Output: executed_submission/ (results, logs, artifacts)  │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. GRADING PHASE (Paperbench)                               │
│    Docker: Optional                                         │
│    Input: - Paper data from paperbench registry             │
│           - Rubric (internal to paperbench package)         │
│           - Executed submission outputs                     │
│    Process: Judge evaluates each rubric task               │
│    Output: JudgeOutput (score, graded_task_tree)           │
└─────────────────────────────────────────────────────────────┘
```

## 4. Key Code Locations in Paperbench Package

```python
# Paper registry (gets paper data)
paperbench/paper_registry.py:
    def get_paper(paper_id: str) -> Paper:
        """Load paper from internal dataset."""
        paper_dir = Path(__file__).parent / "data" / "papers" / paper_id
        rubric = json.load(open(paper_dir / "rubric.json"))
        return Paper(paper_id=paper_id, rubric=TaskNode(**rubric), ...)

# Grading function (main entry point)
paperbench/grade.py:
    def grade_submission(
        submission_path: str,
        paper_id: str,        # ← Uses this to get paper from registry
        judge_type: str,
        ...
    ) -> JudgeOutput:
        # 1. Get paper data from registry
        paper = get_paper(paper_id)

        # 2. Run reproduction in Docker
        executed_submission = run_reproducer(
            submission_path=submission_path,
            docker_image="paperbench-reproducer",
        )

        # 3. Grade using judge
        judge = create_judge(paper, judge_type)
        graded_tree = judge.grade(executed_submission)

        # 4. Return results
        return JudgeOutput(
            score=graded_tree.score,
            graded_task_tree=graded_tree,
            ...
        )

# Judge implementation
paperbench/judge/:
    - scaffold.py: Judge base classes
    - code_only.py: Code-Dev variant judge
    - evaluator.py: Result comparison logic
```

## 5. Data Flow

```
HuggingFace Dataset         Paperbench Package
(leandermaben/paperbench)  (github: frontier-evals)
        │                           │
        ↓                           ↓
┌──────────────┐           ┌──────────────┐
│ Paper content│           │ Rubric data  │
│ (for agent)  │           │ (for judge)  │
└──────────────┘           └──────────────┘
        │                           │
        ↓                           ↓
  INFERENCE PHASE            GRADING PHASE
  (run_infer.py)             (run_eval.py)
        │                           │
        ↓                           ↓
  Agent creates code         Judge compares outputs
  in Docker container        to rubric requirements
        │                           │
        └─────────┬─────────────────┘
                  ↓
            Final Score
```

## 6. Important Notes

### Paper Data is Separate

- **Inference**: Uses HuggingFace dataset (has paper content)
- **Evaluation**: Uses paperbench package dataset (has rubrics + paper data)
- These are **two different sources**
- The paperbench package has the authoritative rubrics

### Docker Images are Separate

- **Inference Docker**: Has OpenHands SDK for agent
- **Reproduction Docker**: Clean Ubuntu for running reproduce.sh
- **Grading**: May or may not use Docker

### Judge is Deterministic

- Given same submission outputs, judge gives same score
- Rubric is fixed (from paperbench package)
- Weights and requirements don't change
- Reproducible evaluation

## 7. How to Verify

You can inspect the judge's behavior:

```python
# Load paperbench
from paperbench.paper_registry import get_paper
from paperbench.grade import run_judge

# Get paper (shows where data comes from)
paper = get_paper("paper_001")
print(paper.rubric)  # Shows the TaskNode tree

# Run judge manually
graded_tree = run_judge(
    submission_dir="/path/to/submission",
    paper_id="paper_001",
)
print(graded_tree.score)
```

This shows exactly what the judge sees and how it scores.
