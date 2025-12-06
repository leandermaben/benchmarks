"""
Paperbench Evaluation Script

This script evaluates agent submissions from the inference phase by using the
paperbench grading infrastructure to run reproduce.sh and grade against rubrics.
"""

import argparse
import json
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List

from tqdm import tqdm

# Import paperbench grading functionality
try:
    from paperbench.grade import grade_submission, run_judge
    PAPERBENCH_AVAILABLE = True
except ImportError:
    PAPERBENCH_AVAILABLE = False
    logging.warning(
        "paperbench package not available. Please install: "
        "pip install 'git+https://github.com/leandermaben/frontier-evals.git#subdirectory=project/paperbench'"
    )

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


def validate_paperbench_data() -> bool:
    """
    Validate that paperbench data directory is set up correctly.

    Returns:
        True if data is available, False otherwise.
    """
    data_dir = os.environ.get("PAPERBENCH_DATA_DIR")

    if not data_dir:
        logger.error(
            "PAPERBENCH_DATA_DIR environment variable not set!\n\n"
            "The paperbench data directory uses Git LFS and is NOT fetched during pip install.\n"
            "You must manually set up the data directory:\n\n"
            "1. Clone the frontier-evals repository:\n"
            "   cd /path/to/your/data/location\n"
            "   git clone https://github.com/leandermaben/frontier-evals.git --filter=blob:none\n"
            "   cd frontier-evals\n\n"
            "2. Fetch LFS data:\n"
            "   git lfs fetch --include 'project/paperbench/data/**'\n"
            "   git lfs checkout project/paperbench/data\n\n"
            "3. Set environment variable:\n"
            "   export PAPERBENCH_DATA_DIR=\"$(pwd)/project/paperbench/data\"\n\n"
            "Or use the setup script:\n"
            "   cd benchmarks/paperbench\n"
            "   ./scripts/setup_data.sh\n\n"
            "See README.md for more details."
        )
        return False

    data_path = Path(data_dir)
    if not data_path.exists():
        logger.error(
            f"PAPERBENCH_DATA_DIR points to non-existent directory: {data_dir}\n"
            f"Please verify the path is correct."
        )
        return False

    papers_dir = data_path / "papers"
    if not papers_dir.exists():
        logger.error(
            f"Papers directory not found at: {papers_dir}\n"
            f"The data directory structure may be incorrect.\n"
            f"Expected: {data_dir}/papers/\n"
            f"Please verify the data was cloned and LFS files were fetched correctly."
        )
        return False

    # Check if there are any paper directories
    paper_dirs = list(papers_dir.glob("*"))
    if not paper_dirs:
        logger.error(
            f"No paper directories found in: {papers_dir}\n"
            f"The LFS files may not have been fetched.\n"
            f"Run: git lfs checkout project/paperbench/data"
        )
        return False

    # Check a sample paper directory for rubric.json
    sample_paper = paper_dirs[0]
    rubric_file = sample_paper / "rubric.json"
    if not rubric_file.exists():
        logger.error(
            f"rubric.json not found in sample paper: {sample_paper}\n"
            f"The LFS files may not have been fetched.\n"
            f"Run: git lfs checkout project/paperbench/data"
        )
        return False

    logger.info(f"✓ Paperbench data directory validated: {data_dir}")
    logger.info(f"✓ Found {len(paper_dirs)} paper directories")
    return True


def evaluate_one_submission(
    paper_id: str,
    submission_path: Path,
    output_dir: Path,
    judge_type: str = "default",
    code_only: bool = False,
) -> Dict:
    """
    Evaluate a single submission using paperbench grading infrastructure.

    Args:
        paper_id: The paper identifier.
        submission_path: Path to submission archive or directory.
        output_dir: Directory to write evaluation results.
        judge_type: Type of judge to use for grading.
        code_only: If True, skip execution and only evaluate code.

    Returns:
        Dictionary with evaluation results.
    """
    if not PAPERBENCH_AVAILABLE:
        return {
            "paper_id": paper_id,
            "success": False,
            "error": "paperbench package not available",
            "score": 0.0,
        }

    logger.info(f"Evaluating submission for paper: {paper_id}")

    try:
        # Create output directory for this paper
        paper_output_dir = output_dir / paper_id
        paper_output_dir.mkdir(parents=True, exist_ok=True)

        grader_output_path = str(paper_output_dir / "grader_output.json")

        # Use paperbench's grade_submission function
        # This handles:
        # 1. Extracting submission
        # 2. Running reproduce.sh in reproducer container
        # 3. Grading against rubric
        judge_output = grade_submission(
            submission_path=str(submission_path),
            paper_id=paper_id,
            judge_type=judge_type,
            grader_upload_path=grader_output_path,
            run_group_id="openhands_eval",
            runs_dir=str(output_dir),
            run_id=paper_id,
            code_only=code_only,
            completer_config=None,  # Use default completer config
        )

        # Extract results
        result = {
            "paper_id": paper_id,
            "success": judge_output.success if judge_output else False,
            "score": judge_output.score if judge_output else 0.0,
            "num_leaf_nodes": judge_output.num_leaf_nodes if judge_output else 0,
            "num_invalid_leaf_nodes": judge_output.num_invalid_leaf_nodes if judge_output else 0,
            "judge_type": judge_type,
            "graded_at": judge_output.graded_at if judge_output else None,
        }

        # Save detailed results
        result_file = paper_output_dir / "evaluation.json"
        with open(result_file, "w") as f:
            json.dump(result, f, indent=2)

        logger.info(
            f"Evaluation complete for {paper_id}: "
            f"score={result['score']:.2f}, "
            f"success={result['success']}"
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


def evaluate_from_inference_output(
    inference_output: Path,
    submissions_dir: Path,
    output_dir: Path,
    num_workers: int = 1,
    judge_type: str = "default",
    code_only: bool = False,
) -> None:
    """
    Evaluate submissions from inference output using paperbench infrastructure.

    Args:
        inference_output: Path to inference output.jsonl file.
        submissions_dir: Directory containing submission subdirectories.
        output_dir: Directory to write evaluation results.
        num_workers: Number of parallel workers.
        judge_type: Type of judge to use for grading.
        code_only: If True, skip execution and only evaluate code.
    """
    if not PAPERBENCH_AVAILABLE:
        logger.error("paperbench package not available. Cannot proceed with evaluation.")
        return

    # Load inference results
    inference_results = []
    with open(inference_output, "r") as f:
        for line in f:
            inference_results.append(json.loads(line))

    logger.info(f"Loaded {len(inference_results)} inference results")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect submission paths
    submissions_to_eval = []
    for result in inference_results:
        paper_id = result["instance_id"]
        submission_dir = submissions_dir / paper_id

        if not submission_dir.exists():
            logger.warning(f"Submission directory not found: {submission_dir}")
            continue

        submissions_to_eval.append((paper_id, submission_dir))

    logger.info(f"Found {len(submissions_to_eval)} submissions to evaluate")

    # Evaluate submissions
    all_results = []

    if num_workers > 1:
        # Parallel evaluation
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(
                    evaluate_one_submission,
                    paper_id,
                    submission_path,
                    output_dir,
                    judge_type,
                    code_only,
                ): paper_id
                for paper_id, submission_path in submissions_to_eval
            }

            for future in tqdm(as_completed(futures), total=len(futures), desc="Evaluating"):
                try:
                    result = future.result()
                    all_results.append(result)
                except Exception as e:
                    paper_id = futures[future]
                    logger.error(f"Error evaluating {paper_id}: {e}")
                    all_results.append({
                        "paper_id": paper_id,
                        "success": False,
                        "error": str(e),
                        "score": 0.0,
                    })
    else:
        # Sequential evaluation
        for paper_id, submission_path in tqdm(submissions_to_eval, desc="Evaluating"):
            result = evaluate_one_submission(
                paper_id, submission_path, output_dir, judge_type, code_only
            )
            all_results.append(result)

    # Write aggregate results
    successful = [r for r in all_results if r["success"]]
    summary = {
        "total": len(all_results),
        "successful": len(successful),
        "failed": len(all_results) - len(successful),
        "average_score": sum(r["score"] for r in all_results) / len(all_results) if all_results else 0.0,
        "average_score_successful": sum(r["score"] for r in successful) / len(successful) if successful else 0.0,
        "results": all_results,
    }

    summary_file = output_dir / "evaluation_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(
        f"Evaluation complete! "
        f"{successful}/{len(all_results)} successful "
        f"(avg score: {summary['average_score']:.2f})"
    )
    logger.info(f"Results written to {summary_file}")


def main():
    """Main entry point for Paperbench evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate Paperbench submissions using paperbench grading infrastructure",
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
        "--judge-type",
        type=str,
        default="default",
        help="Type of judge to use for grading (default: default)",
    )
    parser.add_argument(
        "--code-only",
        action="store_true",
        help="Skip execution and only evaluate code (Code-Dev variant)",
    )

    args = parser.parse_args()

    # Validate paperbench data directory is set up
    if not validate_paperbench_data():
        logger.error("Paperbench data validation failed. Cannot proceed with evaluation.")
        exit(1)

    evaluate_from_inference_output(
        inference_output=args.inference_output,
        submissions_dir=args.submissions_dir,
        output_dir=args.output_dir,
        num_workers=args.num_workers,
        judge_type=args.judge_type,
        code_only=args.code_only,
    )


if __name__ == "__main__":
    main()
