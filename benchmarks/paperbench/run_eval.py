"""
Paperbench Evaluation Script

This script evaluates agent submissions from the inference phase by:
1. Running the reproduce.sh script in a fresh container
2. Grading the results against the paper-specific rubric
3. Generating evaluation scores
"""

import argparse
import json
import logging
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Optional

from datasets import load_dataset
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)

# Default Docker image for evaluation (with GPU support)
DEFAULT_EVAL_IMAGE = "ghcr.io/openhands/paperbench:latest"


class PaperBenchEvaluator:
    """Evaluator for Paperbench submissions."""

    def __init__(
        self,
        dataset_name: str = "leandermaben/paperbench",
        eval_image: str = DEFAULT_EVAL_IMAGE,
        use_gpu: bool = True,
        timeout: int = 7 * 24 * 3600,  # 7 days in seconds
    ):
        """
        Initialize the evaluator.

        Args:
            dataset_name: HuggingFace dataset name for paperbench.
            eval_image: Docker image to use for evaluation.
            use_gpu: Whether to enable GPU support (requires NVIDIA Container Toolkit).
            timeout: Maximum time for reproduction in seconds (default: 7 days).
        """
        self.dataset_name = dataset_name
        self.eval_image = eval_image
        self.use_gpu = use_gpu
        self.timeout = timeout
        self.rubrics = self._load_rubrics()

    def _load_rubrics(self) -> Dict[str, Any]:
        """
        Load rubrics from the dataset.

        Returns:
            Dictionary mapping paper_id to rubric.
        """
        logger.info(f"Loading rubrics from {self.dataset_name}")
        dataset = load_dataset(self.dataset_name, split="train")

        rubrics = {}
        for item in dataset:
            rubrics[item["paper_id"]] = item["rubric"]

        logger.info(f"Loaded {len(rubrics)} rubrics")
        return rubrics

    def evaluate_submission(
        self, paper_id: str, submission_dir: Path, output_dir: Path
    ) -> Dict[str, Any]:
        """
        Evaluate a single submission.

        Args:
            paper_id: The paper identifier.
            submission_dir: Directory containing the submission (with reproduce.sh).
            output_dir: Directory to write evaluation results.

        Returns:
            Dictionary with evaluation results.
        """
        logger.info(f"Evaluating submission for paper: {paper_id}")

        # Check if reproduce.sh exists
        reproduce_script = submission_dir / "reproduce.sh"
        if not reproduce_script.exists():
            logger.error(f"reproduce.sh not found in {submission_dir}")
            return {
                "paper_id": paper_id,
                "success": False,
                "error": "reproduce.sh not found",
                "score": 0.0,
            }

        # Create output directory for this paper
        paper_output_dir = output_dir / paper_id
        paper_output_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Step 1: Run reproduction in container
            reproduction_result = self._run_reproduction(
                paper_id, submission_dir, paper_output_dir
            )

            if not reproduction_result["success"]:
                return {
                    "paper_id": paper_id,
                    "success": False,
                    "error": reproduction_result["error"],
                    "score": 0.0,
                }

            # Step 2: Run grading/judging
            grading_result = self._run_grading(
                paper_id, paper_output_dir
            )

            # Combine results
            result = {
                "paper_id": paper_id,
                "success": True,
                "reproduction": reproduction_result,
                "grading": grading_result,
                "score": grading_result.get("score", 0.0),
            }

            # Save detailed results
            result_file = paper_output_dir / "evaluation.json"
            with open(result_file, "w") as f:
                json.dump(result, f, indent=2)

            logger.info(
                f"Evaluation complete for {paper_id}: score={result['score']:.2f}"
            )
            return result

        except Exception as e:
            logger.error(f"Error evaluating {paper_id}: {e}", exc_info=True)
            return {
                "paper_id": paper_id,
                "success": False,
                "error": str(e),
                "score": 0.0,
            }

    def _run_reproduction(
        self, paper_id: str, submission_dir: Path, output_dir: Path
    ) -> Dict[str, Any]:
        """
        Run the reproduce.sh script in a fresh container.

        Args:
            paper_id: The paper identifier.
            submission_dir: Directory containing the submission.
            output_dir: Directory to write reproduction outputs.

        Returns:
            Dictionary with reproduction results.
        """
        logger.info(f"Running reproduction for {paper_id}")

        # Clean up the submission directory (remove untracked files)
        clean_cmd = ["git", "-C", str(submission_dir), "clean", "-fd"]
        subprocess.run(clean_cmd, capture_output=True)

        # Build Docker run command
        docker_cmd = [
            "docker", "run",
            "--rm",
            "-v", f"{submission_dir.absolute()}:/home/submission",
            "-v", f"{output_dir.absolute()}:/home/output",
            "-w", "/home/submission",
        ]

        # Add GPU support if enabled
        if self.use_gpu:
            docker_cmd.extend(["--gpus", "all"])

        docker_cmd.extend([
            self.eval_image,
            "/bin/bash", "-c",
            "bash reproduce.sh > /home/output/reproduce.log 2>&1"
        ])

        try:
            result = subprocess.run(
                docker_cmd,
                timeout=self.timeout,
                capture_output=True,
                text=True,
            )

            return {
                "success": result.returncode == 0,
                "returncode": result.returncode,
                "error": result.stderr if result.returncode != 0 else None,
            }

        except subprocess.TimeoutExpired:
            logger.error(f"Reproduction timeout for {paper_id}")
            return {
                "success": False,
                "error": f"Timeout after {self.timeout} seconds",
            }
        except Exception as e:
            logger.error(f"Error running reproduction for {paper_id}: {e}")
            return {
                "success": False,
                "error": str(e),
            }

    def _run_grading(
        self, paper_id: str, output_dir: Path
    ) -> Dict[str, Any]:
        """
        Run grading/judging on the reproduction results.

        Args:
            paper_id: The paper identifier.
            output_dir: Directory containing reproduction outputs.

        Returns:
            Dictionary with grading results.
        """
        logger.info(f"Grading results for {paper_id}")

        # Get the rubric for this paper
        rubric = self.rubrics.get(paper_id)
        if not rubric:
            logger.error(f"No rubric found for {paper_id}")
            return {
                "success": False,
                "error": "Rubric not found",
                "score": 0.0,
            }

        # Build Docker run command for grading
        docker_cmd = [
            "docker", "run",
            "--rm",
            "-v", f"{output_dir.absolute()}:/home/output",
            "-w", "/home/output",
            self.eval_image,
            "python", "-m", "paperbench.judge",
            "--paper-id", paper_id,
            "--output-dir", "/home/output",
        ]

        try:
            result = subprocess.run(
                docker_cmd,
                timeout=3600,  # 1 hour timeout for grading
                capture_output=True,
                text=True,
            )

            if result.returncode == 0:
                # Parse grading results
                grade_file = output_dir / "grade.json"
                if grade_file.exists():
                    with open(grade_file, "r") as f:
                        grade_data = json.load(f)
                    return {
                        "success": True,
                        "score": grade_data.get("score", 0.0),
                        "details": grade_data,
                    }
                else:
                    logger.warning(f"Grade file not found for {paper_id}")
                    return {
                        "success": False,
                        "error": "Grade file not generated",
                        "score": 0.0,
                    }
            else:
                logger.error(f"Grading failed for {paper_id}: {result.stderr}")
                return {
                    "success": False,
                    "error": result.stderr,
                    "score": 0.0,
                }

        except subprocess.TimeoutExpired:
            logger.error(f"Grading timeout for {paper_id}")
            return {
                "success": False,
                "error": "Grading timeout",
                "score": 0.0,
            }
        except Exception as e:
            logger.error(f"Error grading {paper_id}: {e}")
            return {
                "success": False,
                "error": str(e),
                "score": 0.0,
            }


def evaluate_from_inference_output(
    inference_output: Path,
    submissions_dir: Path,
    output_dir: Path,
    num_workers: int = 1,
    use_gpu: bool = True,
    eval_image: str = DEFAULT_EVAL_IMAGE,
) -> None:
    """
    Evaluate submissions from inference output.

    Args:
        inference_output: Path to inference output.jsonl file.
        submissions_dir: Directory containing submission subdirectories.
        output_dir: Directory to write evaluation results.
        num_workers: Number of parallel workers.
        use_gpu: Whether to enable GPU support.
        eval_image: Docker image to use for evaluation.
    """
    # Load inference results
    inference_results = []
    with open(inference_output, "r") as f:
        for line in f:
            inference_results.append(json.loads(line))

    logger.info(f"Loaded {len(inference_results)} inference results")

    # Create evaluator
    evaluator = PaperBenchEvaluator(
        eval_image=eval_image,
        use_gpu=use_gpu,
    )

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Evaluate submissions
    all_results = []

    if num_workers > 1:
        # Parallel evaluation
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            for result in inference_results:
                paper_id = result["instance_id"]
                submission_dir = submissions_dir / paper_id

                if not submission_dir.exists():
                    logger.warning(f"Submission directory not found: {submission_dir}")
                    continue

                future = executor.submit(
                    evaluator.evaluate_submission,
                    paper_id,
                    submission_dir,
                    output_dir,
                )
                futures.append((paper_id, future))

            for paper_id, future in tqdm(futures, desc="Evaluating"):
                try:
                    result = future.result()
                    all_results.append(result)
                except Exception as e:
                    logger.error(f"Error evaluating {paper_id}: {e}")
                    all_results.append({
                        "paper_id": paper_id,
                        "success": False,
                        "error": str(e),
                        "score": 0.0,
                    })
    else:
        # Sequential evaluation
        for result in tqdm(inference_results, desc="Evaluating"):
            paper_id = result["instance_id"]
            submission_dir = submissions_dir / paper_id

            if not submission_dir.exists():
                logger.warning(f"Submission directory not found: {submission_dir}")
                continue

            eval_result = evaluator.evaluate_submission(
                paper_id, submission_dir, output_dir
            )
            all_results.append(eval_result)

    # Write aggregate results
    summary_file = output_dir / "evaluation_summary.json"
    with open(summary_file, "w") as f:
        json.dump({
            "total": len(all_results),
            "successful": sum(1 for r in all_results if r["success"]),
            "average_score": sum(r["score"] for r in all_results) / len(all_results) if all_results else 0.0,
            "results": all_results,
        }, f, indent=2)

    logger.info(f"Evaluation complete! Results written to {summary_file}")


def main():
    """Main entry point for Paperbench evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate Paperbench submissions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "inference_output",
        type=Path,
        help="Path to inference output.jsonl file",
    )
    parser.add_argument(
        "submissions_dir",
        type=Path,
        help="Directory containing submission subdirectories (one per paper)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("eval_outputs/paperbench/evaluation"),
        help="Output directory for evaluation results",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1)",
    )
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Disable GPU support",
    )
    parser.add_argument(
        "--eval-image",
        type=str,
        default=DEFAULT_EVAL_IMAGE,
        help=f"Docker image to use for evaluation (default: {DEFAULT_EVAL_IMAGE})",
    )

    args = parser.parse_args()

    evaluate_from_inference_output(
        inference_output=args.inference_output,
        submissions_dir=args.submissions_dir,
        output_dir=args.output_dir,
        num_workers=args.num_workers,
        use_gpu=not args.no_gpu,
        eval_image=args.eval_image,
    )


if __name__ == "__main__":
    main()
