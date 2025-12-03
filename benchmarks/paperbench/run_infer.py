"""
Paperbench Inference Script

This script runs the Paperbench benchmark, which evaluates AI agents' capabilities
in replicating academic research by reproducing ICML 2024 papers from scratch.
"""

import argparse
import json
import logging
import os
import random
import shutil
import tarfile
import tempfile
from pathlib import Path
from typing import Any, List, Optional

from datasets import load_dataset
from jinja2 import Environment, FileSystemLoader

from benchmarks.utils.evaluation import Evaluation
from benchmarks.utils.evaluation_utils import get_default_on_result_writer
from benchmarks.utils.models import EvalInstance, EvalMetadata, EvalOutput
from openhands.sdk import Agent, Conversation, LLM
from openhands.sdk.critic import PassCritic
from openhands.tools.preset.default import get_default_tools
from openhands.workspace import DockerWorkspace, RemoteWorkspace

try:
    from paperbench.rubric.tasks import TaskNode
except ImportError:
    TaskNode = None
    logger.warning("paperbench not installed, leaf node extraction will be skipped")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)

# Default Docker image for Paperbench
DEFAULT_DOCKER_IMAGE = "paperbench-agent:latest"


def generate_instruction(instance_data: dict, template_path: str | None = None) -> str:
    """Generate instruction for the agent using Jinja template."""
    if template_path is None:
        # Use default template
        template_path = os.path.join(os.path.dirname(__file__), "prompts", "default.j2")

    # Set up Jinja2 environment
    prompts_dir = os.path.dirname(template_path)
    template_name = os.path.basename(template_path)
    env = Environment(loader=FileSystemLoader(prompts_dir))
    template = env.get_template(template_name)

    # Render the instruction
    instruction = template.render(instance=instance_data)
    return instruction


class PaperBenchEvaluation(Evaluation):
    """Paperbench evaluation orchestrator."""

    def prepare_instances(self) -> List[EvalInstance]:
        """
        Load and prepare Paperbench instances from HuggingFace dataset.

        Returns:
            List of EvalInstance objects ready for evaluation.
        """
        logger.info(f"Loading dataset: {self.metadata.dataset}")

        # Load dataset from HuggingFace
        dataset = load_dataset(
            self.metadata.dataset,
            split=self.metadata.dataset_split,
        )

        # Get paper_ids filter if specified
        paper_ids_filter = self.metadata.details.get("paper_ids", None)

        # Convert dataset to list for filtering/shuffling
        all_items = list(dataset)

        # Filter by paper IDs if specified
        if paper_ids_filter:
            logger.info(f"Filtering to {len(paper_ids_filter)} specified paper IDs")
            all_items = [item for item in all_items if item["paper_id"] in paper_ids_filter]
            if not all_items:
                logger.warning("No papers matched the specified paper IDs!")

        # Shuffle if seed is specified
        seed = self.metadata.details.get("seed", None)
        if seed is not None:
            logger.info(f"Shuffling papers with seed: {seed}")
            random.seed(seed)
            random.shuffle(all_items)

        # Apply eval_limit after filtering/shuffling
        if self.metadata.eval_limit and self.metadata.eval_limit > 0:
            all_items = all_items[:self.metadata.eval_limit]

        # Create instances
        instances = []
        for item in all_items:
            instance = EvalInstance(
                id=item["paper_id"],
                data={
                    "paper_id": item["paper_id"],
                    "paper_content": item["paper_content"],
                    "rubric": item["rubric"],
                    "config": item["config"],
                    "addendum": item.get("addendum", ""),
                    "blacklist": item.get("blacklist", []),
                    "assets": item.get("assets", []),
                }
            )
            instances.append(instance)

        logger.info(f"Loaded {len(instances)} instances for evaluation")
        if instances:
            logger.info(f"Paper IDs: {[inst.id for inst in instances]}")

        return instances

    def prepare_workspace(self, instance: EvalInstance) -> RemoteWorkspace:
        """
        Prepare workspace for a Paperbench instance.

        Args:
            instance: The evaluation instance to prepare workspace for.

        Returns:
            Configured workspace (Docker or Remote).
        """
        # Determine which Docker image to use
        server_image = self.metadata.details.get("server_image", DEFAULT_DOCKER_IMAGE)

        if self.metadata.workspace_type == "docker":
            workspace = DockerWorkspace(
                server_image=server_image,
                working_dir="/workspace",
                platform="linux/amd64",
            )
        else:
            # Remote workspace
            workspace = self._get_remote_workspace(server_image)

        # Run any setup commands if specified
        if self.metadata.env_setup_commands:
            for cmd in self.metadata.env_setup_commands:
                logger.info(f"Running setup command: {cmd}")
                result = workspace.execute_command(cmd)
                if result.returncode != 0:
                    logger.warning(
                        f"Setup command failed: {cmd}\n"
                        f"stdout: {result.stdout}\n"
                        f"stderr: {result.stderr}"
                    )

        return workspace

    def evaluate_instance(
        self, instance: EvalInstance, workspace: RemoteWorkspace
    ) -> EvalOutput:
        """
        Run the agent on a single Paperbench instance.

        Args:
            instance: The evaluation instance to run.
            workspace: The prepared workspace.

        Returns:
            Evaluation output containing results and metrics.
        """
        logger.info(f"Evaluating instance: {instance.id}")

        try:
            # Save paper files to workspace
            self._save_paper_files(instance, workspace)

            # Download paper assets if available
            self._download_assets(instance, workspace)

            # Build the instruction from the template
            instruction = self._build_instruction(instance, workspace)

            # Setup tools
            tools = get_default_tools(enable_browser=True)

            # Create agent
            agent = Agent(
                llm=self.metadata.llm,
                tools=tools,
                system_prompt_kwargs={"cli_mode": True},
            )

            # Create conversation
            conversation = Conversation(
                agent=agent,
                workspace=workspace,
                max_iteration_per_run=self.metadata.max_iterations,
            )

            # Execute the task
            conversation.send_message(instruction)
            conversation.run()

            # Collect results
            events = conversation.state.events
            metrics = conversation.conversation_stats.get_combined_metrics()

            # Extract the submission (code in /workspace)
            submission_info = self._extract_submission(workspace)

            # Save submission to local directory
            local_submission_path = self._save_submission_to_local(
                instance, workspace
            )
            submission_info["local_path"] = str(local_submission_path)

            # Create output
            output = EvalOutput(
                instance_id=instance.id,
                test_result={
                    "submission_ready": submission_info["has_reproduce_script"],
                    "submission_path": submission_info["path"],
                    "file_count": submission_info["file_count"],
                },
                instruction=instruction,
                error=None,
                history=events,
                metrics=metrics,
                instance=instance.data,
            )

            logger.info(f"Successfully evaluated instance: {instance.id}")
            return output

        except Exception as e:
            logger.error(f"Error evaluating instance {instance.id}: {e}", exc_info=True)
            return EvalOutput(
                instance_id=instance.id,
                test_result={"error": str(e)},
                instruction="Error occurred before instruction was built",
                error=str(e),
                history=[],
                metrics={},
                instance=instance.data,
            )

    def _extract_leaf_nodes(self, rubric: dict) -> List[dict]:
        """
        Extract leaf nodes from the rubric tree.

        Args:
            rubric: The rubric dictionary.

        Returns:
            List of leaf node dictionaries with task info.
        """
        if TaskNode is None:
            return []

        try:
            # Convert rubric dict to TaskNode
            task_tree = TaskNode(**rubric)

            # Get leaf nodes
            leaf_nodes = task_tree.get_leaf_nodes()

            # Convert to simple dicts for template
            leaf_tasks = []
            for node in leaf_nodes:
                leaf_tasks.append({
                    "id": node.id,
                    "requirements": node.requirements,
                    "weight": node.weight,
                    "task_category": node.task_category,
                    "finegrained_task_category": node.finegrained_task_category,
                })

            return leaf_tasks
        except Exception as e:
            logger.warning(f"Error extracting leaf nodes: {e}")
            return []

    def _save_paper_files(
        self, instance: EvalInstance, workspace: RemoteWorkspace
    ) -> None:
        """
        Save paper content, rubric, and addendum to files in the workspace.

        Args:
            instance: The evaluation instance.
            workspace: The workspace to save files to.
        """
        logger.info(f"Saving paper files for {instance.id}")

        # Create paper directory
        workspace.execute_command("mkdir -p /workspace/paper")

        # Save paper content
        paper_content = instance.data.get("paper_content", "")
        workspace.execute_command(
            f"cat > /workspace/paper/paper.md << 'EOFPAPER'\n{paper_content}\nEOFPAPER"
        )

        # Save rubric
        rubric = instance.data.get("rubric", {})
        rubric_json = json.dumps(rubric, indent=2)
        workspace.execute_command(
            f"cat > /workspace/paper/rubric.json << 'EOFRUBRIC'\n{rubric_json}\nEOFRUBRIC"
        )

        # Save addendum if present
        addendum = instance.data.get("addendum", "")
        if addendum:
            workspace.execute_command(
                f"cat > /workspace/paper/addendum.txt << 'EOFADDENDUM'\n{addendum}\nEOFADDENDUM"
            )

        logger.info(f"Paper files saved to /workspace/paper/")

    def _build_instruction(
        self, instance: EvalInstance, workspace: RemoteWorkspace
    ) -> str:
        """
        Build the instruction prompt for the agent.

        Args:
            instance: The evaluation instance.
            workspace: The workspace (not used but kept for consistency).

        Returns:
            The formatted instruction string.
        """
        # Extract leaf nodes from rubric
        rubric = instance.data.get("rubric", {})
        leaf_tasks = self._extract_leaf_nodes(rubric)

        # Create template context with leaf tasks
        template_data = {**instance.data, "leaf_tasks": leaf_tasks}

        # Generate instruction using template
        template_path = self.metadata.prompt_path
        instruction = generate_instruction(template_data, template_path)

        return instruction

    def _download_assets(
        self, instance: EvalInstance, workspace: RemoteWorkspace
    ) -> None:
        """
        Download paper assets (images, data files) to the workspace.

        Assets are stored in the paperbench repo at:
        data/papers/<paper_id>/<asset_filename>

        Args:
            instance: The evaluation instance.
            workspace: The workspace to download assets to.
        """
        assets = instance.data.get("assets", [])
        if not assets:
            return

        logger.info(f"Downloading {len(assets)} assets for {instance.id}")

        # Create assets directory
        workspace.execute_command("mkdir -p /workspace/assets")

        paper_id = instance.id
        base_url = "https://raw.githubusercontent.com/leandermaben/frontier-evals/main/project/paperbench"

        for asset in assets:
            try:
                # Assets are filenames, construct full URL to paperbench repo
                if asset.startswith("http://") or asset.startswith("https://"):
                    # Already a URL, use as-is
                    asset_url = asset
                else:
                    # Filename only, construct URL to paperbench repo
                    asset_url = f"{base_url}/data/papers/{paper_id}/{asset}"

                logger.info(f"Downloading asset: {asset_url}")
                cmd = f"wget -P /workspace/assets '{asset_url}'"
                result = workspace.execute_command(cmd)

                if result.returncode != 0:
                    logger.warning(f"Failed to download asset: {asset_url}")
            except Exception as e:
                logger.warning(f"Error downloading asset {asset}: {e}")

    def _save_submission_to_local(
        self, instance: EvalInstance, workspace: RemoteWorkspace
    ) -> Path:
        """
        Download the workspace contents (submission) to local directory.

        Args:
            instance: The evaluation instance.
            workspace: The workspace containing the submission.

        Returns:
            Path to the local submission directory.
        """
        paper_id = instance.id
        submissions_dir = Path(self.metadata.eval_output_dir) / "submissions"
        submission_dir = submissions_dir / paper_id
        submission_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Downloading submission for {paper_id} to {submission_dir}")

        try:
            # Create a tarball of the workspace in the container
            tar_path = f"/tmp/submission_{paper_id}.tar.gz"
            tar_cmd = f"tar -czf {tar_path} -C /workspace ."
            result = workspace.execute_command(tar_cmd)

            if result.returncode != 0:
                logger.error(f"Failed to create tarball: {result.stderr}")
                return submission_dir

            # Download the tarball using workspace file_download
            with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp_file:
                tmp_path = tmp_file.name

            try:
                # Read the file content from workspace
                cat_result = workspace.execute_command(f"cat {tar_path}")
                if cat_result.returncode == 0:
                    with open(tmp_path, "wb") as f:
                        f.write(cat_result.stdout.encode("latin1"))

                    # Extract the tarball to submission directory
                    with tarfile.open(tmp_path, "r:gz") as tar:
                        tar.extractall(submission_dir)

                    logger.info(f"Submission saved to {submission_dir}")
                else:
                    logger.error(f"Failed to download tarball: {cat_result.stderr}")
            finally:
                # Clean up temp file
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

                # Clean up tarball in container
                workspace.execute_command(f"rm -f {tar_path}")

        except Exception as e:
            logger.error(f"Error saving submission for {paper_id}: {e}", exc_info=True)

        return submission_dir

    def _extract_submission(self, workspace: RemoteWorkspace) -> dict[str, Any]:
        """
        Extract information about the agent's submission.

        Args:
            workspace: The workspace containing the submission.

        Returns:
            Dictionary with submission information.
        """
        # Check if reproduce.sh exists (required for Paperbench)
        check_script = workspace.execute_command("test -f /workspace/reproduce.sh")
        has_reproduce_script = check_script.returncode == 0

        # Count files in workspace
        count_result = workspace.execute_command(
            "find /workspace -type f | wc -l"
        )
        file_count = int(count_result.stdout.strip()) if count_result.returncode == 0 else 0

        return {
            "has_reproduce_script": has_reproduce_script,
            "path": "/workspace",
            "file_count": file_count,
        }


def main():
    """Main entry point for Paperbench inference."""
    parser = argparse.ArgumentParser(
        description="Run Paperbench evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "llm_config",
        type=str,
        help="Path to LLM configuration JSON file",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="leandermaben/paperbench",
        help="HuggingFace dataset name (default: leandermaben/paperbench)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to use (default: train)",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=100,
        help="Maximum iterations per instance (default: 100)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="eval_outputs/paperbench",
        help="Output directory for results",
    )
    parser.add_argument(
        "--n-limit",
        type=int,
        default=0,
        help="Limit number of instances to evaluate (0 = all)",
    )
    parser.add_argument(
        "--workspace-type",
        type=str,
        choices=["docker", "remote"],
        default="docker",
        help="Workspace type (default: docker)",
    )
    parser.add_argument(
        "--server-image",
        type=str,
        default=DEFAULT_DOCKER_IMAGE,
        help=f"Docker image to use (default: {DEFAULT_DOCKER_IMAGE})",
    )
    parser.add_argument(
        "--note",
        type=str,
        default="",
        help="Note to add to output directory name",
    )
    parser.add_argument(
        "--paper-ids",
        type=str,
        default=None,
        help="Path to text file containing paper IDs (one per line) to evaluate",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for shuffling papers. Useful for reproducible random sampling.",
    )

    args = parser.parse_args()

    # Load LLM configuration
    with open(args.llm_config, "r") as f:
        llm_config = f.read()
    llm = LLM.model_validate_json(llm_config)

    # Load paper IDs from file if specified
    paper_ids = None
    if args.paper_ids:
        logger.info(f"Loading paper IDs from: {args.paper_ids}")
        with open(args.paper_ids, "r") as f:
            paper_ids = [line.strip() for line in f if line.strip()]
        logger.info(f"Loaded {len(paper_ids)} paper IDs from file")

    # Create metadata
    metadata = EvalMetadata(
        llm=llm,
        dataset=args.dataset,
        dataset_split=args.split,
        max_iterations=args.max_iterations,
        eval_output_dir=args.output_dir,
        eval_limit=args.n_limit,
        workspace_type=args.workspace_type,
        critic=PassCritic(),  # Use PassCritic for paperbench (eval happens separately)
        details={
            "server_image": args.server_image,
            "paper_ids": paper_ids,
            "seed": args.seed,
        },
    )

    # Create output directory
    output_dir = Path(args.output_dir)
    if args.note:
        output_dir = output_dir / args.note
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save metadata
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata.model_dump(), f, indent=2, default=str)

    # Create evaluator
    evaluator = PaperBenchEvaluation(
        metadata=metadata,
        num_workers=args.num_workers,
    )

    # Custom result writer that saves trajectory and additional files
    def _paperbench_result_writer(output_dir: Path):
        """Create a result writer that saves trajectory and other files."""
        output_file = output_dir / "output.jsonl"

        # Use default writer for JSONL
        default_writer = get_default_on_result_writer(str(output_file))

        def _write_result(instance: EvalInstance, output: EvalOutput):
            # Write to JSONL using default writer
            default_writer(instance, output)

            # Save trajectory
            traj_file = output_dir / f"traj_{instance.id}.json"
            with open(traj_file, "w") as f:
                json.dump(output.history, f, indent=2)

            # Save test result
            eval_file = output_dir / f"eval_{instance.id}.json"
            with open(eval_file, "w") as f:
                json.dump(output.test_result, f, indent=2)

            # Save state info
            state_file = output_dir / f"state_{instance.id}.json"
            state_data = {
                "instance_id": instance.id,
                "history": output.history,
                "num_events": len(output.history) if output.history else 0,
                "submission_info": output.test_result,
                "metrics": output.metrics,
            }
            with open(state_file, "w") as f:
                json.dump(state_data, f, indent=2)

            logger.info(
                f"Saved results for {instance.id}: "
                f"trajectory={traj_file}, eval={eval_file}, state={state_file}"
            )

        return _write_result

    # Run evaluation
    logger.info(f"Starting evaluation, results will be written to {output_dir}")
    logger.info(f"Submissions will be saved to {output_dir / 'submissions'}")

    evaluator.run(on_result=_paperbench_result_writer(output_dir))

    logger.info("Evaluation complete!")


if __name__ == "__main__":
    main()
