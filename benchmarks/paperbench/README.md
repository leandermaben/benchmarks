# Paperbench Benchmark

Paperbench evaluates AI agents' capabilities in replicating academic research by having them reproduce results from ICML 2024 Spotlight and Oral papers from scratch.

## Overview

The benchmark consists of 23 papers from ICML 2024. For each paper, an AI agent must:
1. Read and understand the paper
2. Implement the methods described
3. Run experiments to reproduce the key results
4. Create a `reproduce.sh` script that can be executed to regenerate the results

## Setup

### Prerequisites

- Python 3.12+
- Docker (for containerized evaluation)
- Git with Git LFS support
- NVIDIA Container Toolkit (optional, for GPU support)
- OpenHands Agent SDK

### Installation

#### 1. Install System Dependencies

```bash
# Install Git LFS (if not already installed)
make build

# For GPU support (optional but recommended)
# Follow instructions at: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
```

#### 2. Install Python Dependencies

The benchmark dependencies are managed through the main repository's `pyproject.toml`. From the repository root:

```bash
# Install all dependencies
uv pip install -e .

# Or if using pip
pip install -e .
```

#### 3. Install Paperbench Package

The paperbench evaluation package needs to be installed separately:

```bash
uv pip install "git+https://github.com/leandermaben/frontier-evals.git#subdirectory=project/paperbench"
```

#### 4. Build the Docker Image

Build the Paperbench Docker image with the OpenHands SDK:

```bash
cd benchmarks/paperbench
docker build -t ghcr.io/openhands/paperbench:latest .
```

Alternatively, if a pre-built image is available:

```bash
docker pull ghcr.io/openhands/paperbench:latest
```

### Dataset

The Paperbench dataset is hosted on HuggingFace: [`leandermaben/paperbench`](https://huggingface.co/datasets/leandermaben/paperbench)

The dataset includes:
- **23 papers** from ICML 2024
- Full paper content in markdown format
- Paper-specific evaluation rubrics
- Associated assets (images, data files)
- Blacklisted repositories (implementations that should not be copied)

The dataset will be automatically downloaded when running the inference script.

## Usage

### Running Inference

The inference phase runs the AI agent to reproduce papers:

```bash
paperbench-infer .llm_config/gpt-4o.json \
  --dataset leandermaben/paperbench \
  --split train \
  --max-iterations 100 \
  --num-workers 1 \
  --output-dir eval_outputs/paperbench/gpt-4o \
  --workspace-type docker \
  --server-image ghcr.io/openhands/paperbench:latest
```

#### Key Arguments

- `llm_config`: Path to LLM configuration JSON file (see LLM Configuration below)
- `--dataset`: HuggingFace dataset name (default: `leandermaben/paperbench`)
- `--split`: Dataset split to use (default: `train`)
- `--max-iterations`: Maximum agent iterations per paper (default: 100)
- `--num-workers`: Number of parallel workers (default: 1)
- `--output-dir`: Directory for output files
- `--n-limit`: Limit number of papers to evaluate (0 = all)
- `--workspace-type`: Use `docker` for local or `remote` for cloud-based execution
- `--server-image`: Docker image for the agent server
- `--note`: Optional note to add to output directory name

#### Output Structure

```
eval_outputs/paperbench/gpt-4o/
├── metadata.json           # Evaluation configuration
├── output.jsonl           # Results for all papers
└── logs/
    ├── paper_id_1.log     # Agent logs per paper
    └── ...
```

Each line in `output.jsonl` contains:
```json
{
  "instance_id": "paper_id",
  "test_result": {
    "submission_ready": true,
    "submission_path": "/workspace",
    "file_count": 42
  },
  "instruction": "...",
  "error": null,
  "history": [...],
  "metrics": {...},
  "instance": {...}
}
```

### Running Evaluation

After inference, evaluate the submissions against the paper rubrics:

```bash
paperbench-eval \
  eval_outputs/paperbench/gpt-4o/output.jsonl \
  eval_outputs/paperbench/gpt-4o/submissions \
  --output-dir eval_outputs/paperbench/gpt-4o/evaluation \
  --num-workers 1 \
  --eval-image ghcr.io/openhends/paperbench:latest
```

#### Evaluation Arguments

- `inference_output`: Path to the inference `output.jsonl` file
- `submissions_dir`: Directory containing submission subdirectories (one per paper)
- `--output-dir`: Directory for evaluation results
- `--num-workers`: Number of parallel evaluation workers
- `--no-gpu`: Disable GPU support (enabled by default)
- `--eval-image`: Docker image for evaluation

#### Evaluation Process

For each submission, the evaluator:
1. **Cleans the workspace**: Runs `git clean -fd` to remove untracked files
2. **Runs reproduction**: Executes `reproduce.sh` in a fresh container (up to 7 days)
3. **Grades results**: Compares outputs against the paper-specific rubric
4. **Generates scores**: Produces evaluation metrics and overall score

#### Evaluation Output

```
eval_outputs/paperbench/gpt-4o/evaluation/
├── evaluation_summary.json   # Aggregate results
└── paper_id/
    ├── evaluation.json       # Detailed evaluation for this paper
    ├── reproduce.log         # Output from reproduce.sh
    └── grade.json           # Grading results
```

## LLM Configuration

Create a JSON file with your LLM configuration:

### OpenAI (GPT-4)

`.llm_config/gpt-4o.json`:
```json
{
  "model": "gpt-4o",
  "api_key": "YOUR_OPENAI_API_KEY"
}
```

### Anthropic (Claude)

`.llm_config/claude.json`:
```json
{
  "model": "claude-sonnet-4",
  "api_key": "YOUR_ANTHROPIC_API_KEY"
}
```

### LiteLLM Proxy

`.llm_config/litellm.json`:
```json
{
  "model": "litellm_proxy/anthropic/claude-sonnet-4",
  "base_url": "https://your-proxy.example.com",
  "api_key": "YOUR_API_KEY"
}
```

## Customization

### Custom Prompts

Modify the prompt template at `benchmarks/paperbench/prompts/default.j2` to customize the instructions given to the agent.

### Docker Image Customization

Edit `benchmarks/paperbench/Dockerfile` to:
- Add additional system dependencies
- Include pre-installed Python packages
- Configure GPU settings
- Set up custom environments

### Evaluation Rubrics

Rubrics are paper-specific and loaded from the HuggingFace dataset. They define:
- Required components to reproduce
- Evaluation criteria and weights
- Success thresholds

## Development

### Project Structure

```
benchmarks/paperbench/
├── Dockerfile              # Docker image for agent server and evaluation
├── README.md              # This file
├── run_infer.py           # Inference script
├── run_eval.py            # Evaluation script
└── prompts/
    └── default.j2         # Default prompt template
```

### Adding Dependencies

To add dependencies to the Docker image:

1. Edit `Dockerfile` and add to the `RUN pip install` commands
2. Rebuild the image: `docker build -t ghcr.io/openhands/paperbench:latest .`

### Running Tests

Run a limited evaluation to test the setup:

```bash
# Test with first 2 papers only
paperbench-infer .llm_config/gpt-4o.json \
  --n-limit 2 \
  --output-dir test_outputs/paperbench
```

## Troubleshooting

### Git LFS Issues

If you encounter issues with Git LFS during evaluation:

```bash
# Install Git LFS in the Docker container
docker run -it ghcr.io/openhands/paperbench:latest bash
> git lfs install
> git lfs pull
```

### GPU Not Detected

Ensure NVIDIA Container Toolkit is installed:

```bash
# Test GPU access
docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi

# If this fails, install NVIDIA Container Toolkit
# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
```

### Timeout Issues

If papers take too long to reproduce:

1. Increase `--max-iterations` for inference
2. Adjust the timeout in `run_eval.py` (default: 7 days)
3. Use more powerful hardware (e.g., GPU instances)

### Memory Issues

For memory-intensive papers:

1. Run with `--num-workers 1` to avoid parallel execution
2. Increase Docker memory limits: `docker run --memory=16g ...`
3. Use remote workspace instead of Docker

## Citation

If you use Paperbench in your research, please cite:

```bibtex
@misc{paperbench2024,
  title={Paperbench: Evaluating AI Agents on Academic Research Reproduction},
  author={Leander Maben and contributors},
  year={2024},
  url={https://github.com/leandermaben/frontier-evals}
}
```

## Resources

- **Dataset**: https://huggingface.co/datasets/leandermaben/paperbench
- **Original Paperbench**: https://github.com/leandermaben/frontier-evals/tree/main/project/paperbench
- **OpenHands SDK**: https://github.com/OpenHands/agent-sdk
- **OpenHands PR**: https://github.com/OpenHands/OpenHands/pull/11656

## Support

For issues or questions:
- Open an issue in this repository
- Check the original Paperbench repository
- Contact the maintainers
