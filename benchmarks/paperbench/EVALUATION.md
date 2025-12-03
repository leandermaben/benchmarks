# Paperbench Evaluation Methodology

## Overview

This document explains how the OpenHands Paperbench evaluation implementation matches the original Paperbench methodology.

## Evaluation Architecture

### Original Paperbench Three-Stage Pipeline

Paperbench uses a three-stage evaluation pipeline:

1. **Agent Rollout** - Agent generates code to reproduce the paper
2. **Reproduction** - Code executes in a clean container with GPU access
3. **Grading** - Judge evaluates outputs against paper-specific rubrics

### OpenHands Implementation

Our implementation maintains the same three-stage pipeline but separates inference from evaluation:

**Inference Phase** (`run_infer.py`):
- Corresponds to "Agent Rollout" stage
- Agent receives paper and rubric information
- Agent generates reproduction code and `reproduce.sh` script
- Outputs saved to workspace

**Evaluation Phase** (`run_eval.py`):
- Handles both "Reproduction" and "Grading" stages
- Uses `paperbench.grade.grade_submission()` directly
- Maintains exact grading methodology

## How Evaluation Matches Paperbench

### 1. Using Paperbench's Grading Infrastructure

**Key Code (run_eval.py:78-88):**
```python
from paperbench.grade import grade_submission

judge_output = grade_submission(
    submission_path=str(submission_path),
    paper_id=paper_id,
    judge_type=judge_type,
    grader_upload_path=grader_output_path,
    run_group_id="openhands_eval",
    runs_dir=str(output_dir),
    run_id=paper_id,
    code_only=code_only,
    completer_config=None,
)
```

This is the **exact same function** that paperbench uses internally for grading. It handles:

### 2. Reproduction Stage

**What `grade_submission()` does internally:**

1. **Extract submission** - Unpacks the submission archive/directory
2. **Clean workspace** - Runs `git clean -fd` to remove untracked files
3. **Launch reproducer container** - Uses paperbench's `reproducer.Dockerfile`
4. **Execute `reproduce.sh`** - Runs the reproduction script
5. **Capture outputs** - Saves execution logs and generated results
6. **Timeout handling** - Enforces max runtime (default: 7 days)

**Container used:** The paperbench reproducer container (Ubuntu 24.04, Python 3.11/3.12, ML libraries)

**Execution:** Exact same Docker run command as paperbench:
```bash
docker run --gpus all -v /submission:/submission reproducer bash reproduce.sh
```

### 3. Grading Stage

**What `grade_submission()` does for grading:**

1. **Load paper rubric** - Gets the hierarchical task tree for this paper
2. **Initialize judge** - Sets up the paperbench judge with specified type
3. **Evaluate outputs** - Compares reproduction results to rubric requirements
4. **Score calculation** - Computes weighted scores based on task completion
5. **Generate report** - Creates `JudgeOutput` with detailed results

**Judge types supported:**
- `"default"` - Full evaluation (execution + results matching)
- `"code_only"` - Code-Dev variant (skip execution, evaluate code quality)
- `"dummy"` - Testing judge

**Rubric evaluation:**
- Uses `TaskNode.get_leaf_nodes()` to get fine-grained tasks
- Evaluates each leaf node separately
- Weights applied according to rubric specifications
- Hierarchical scoring aggregated from leaf to root

### 4. Output Format

**JudgeOutput structure (matches paperbench):**
```python
{
    "judge_type": "default",
    "score": 0.75,                    # Overall score (0-1)
    "num_leaf_nodes": 15,             # Total evaluation tasks
    "num_invalid_leaf_nodes": 3,      # Failed tasks
    "graded_at": "2024-01-15T...",    # Timestamp
    "graded_task_tree": {...},        # Detailed per-task results
    "success": True                    # All nodes valid
}
```

This is identical to paperbench's `JudgeOutput` dataclass.

## Differences from Original Paperbench

### What We Changed

1. **Separation of inference and evaluation**
   - Original: Combined in single script
   - Ours: Separate `run_infer.py` and `run_eval.py`
   - Why: Better modularity, can evaluate multiple agent runs

2. **Agent framework**
   - Original: Uses paperbench's custom agent implementations
   - Ours: Uses OpenHands SDK agents
   - Why: Integrate with OpenHands ecosystem

3. **Prompt engineering**
   - Original: Minimal prompting, relies on agent design
   - Ours: Structured prompts with leaf node tasks
   - Why: Better guidance for general-purpose agents

### What We Preserved

1. **Evaluation methodology** - 100% identical via `grade_submission()`
2. **Rubric structure** - Same hierarchical TaskNode trees
3. **Scoring algorithm** - Same weighted leaf node scoring
4. **Container environments** - Same Dockerfiles
5. **Judge types** - Same judge variants
6. **Output format** - Same JudgeOutput structure
7. **Reproduction workflow** - Same `reproduce.sh` execution

## Verification

To verify that evaluation matches paperbench:

1. **Check the import**: We import `paperbench.grade.grade_submission`
2. **Trace the call**: All evaluation goes through this function
3. **Inspect outputs**: `JudgeOutput` format matches paperbench
4. **Compare scores**: Same submission should get same score

## Example Usage

```bash
# Run inference (agent generates code)
paperbench-infer .llm_config/gpt-4o.json \
  --output-dir eval_outputs/paperbench/run1

# Run evaluation (grade submissions using paperbench)
paperbench-eval \
  eval_outputs/paperbench/run1/output.jsonl \
  eval_outputs/paperbench/run1/submissions \
  --output-dir eval_outputs/paperbench/run1/grading \
  --judge-type default

# Use Code-Dev variant (skip execution)
paperbench-eval \
  eval_outputs/paperbench/run1/output.jsonl \
  eval_outputs/paperbench/run1/submissions \
  --output-dir eval_outputs/paperbench/run1/grading_code_only \
  --code-only
```

## Summary

**The evaluation is identical to paperbench** because:

1. ✅ We use `paperbench.grade.grade_submission()` directly
2. ✅ This function handles reproduction + grading exactly as paperbench does
3. ✅ Same containers, same rubrics, same scoring
4. ✅ Same judge infrastructure and output format
5. ✅ Only difference is how agent code is generated (inference phase)

**Bottom line:** The evaluation phase is not re-implemented—it's the **exact same code** from the paperbench package.
