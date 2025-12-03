"""
Paperbench Inference Script

This script runs the Paperbench benchmark, which evaluates AI agents' capabilities
in replicating academic research by reproducing ICML 2024 papers from scratch.
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, List

import jinja2
from datasets import load_dataset

from benchmarks.utils.evaluation import Evaluation
from benchmarks.utils.evaluation_utils import get_default_on_result_writer
from benchmarks.utils.models import EvalInstance, EvalMetadata, EvalOutput
from openhands.sdk import Agent, Conversation, LLM
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
DEFAULT_DOCKER_IMAGE = "ghcr.io/openhands/paperbench:latest"


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

        instances = []
        for idx, item in enumerate(dataset):
            if self.metadata.eval_limit and idx >= self.metadata.eval_limit:
                break

            # Create instance with paper_id as the unique identifier
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

        logger.info(f"Loaded {len(instances)} instances")
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
        # Load the prompt template
        if self.metadata.prompt_path:
            prompt_path = Path(self.metadata.prompt_path)
        else:
            prompt_path = Path(__file__).parent / "prompts" / "default.j2"

        with open(prompt_path, "r") as f:
            template_str = f.read()

        # Extract leaf nodes from rubric
        rubric = instance.data.get("rubric", {})
        leaf_tasks = self._extract_leaf_nodes(rubric)

        # Create template context with leaf tasks
        template_data = {**instance.data, "leaf_tasks": leaf_tasks}

        template = jinja2.Template(template_str)
        instruction = template.render(instance=template_data)

        return instruction

    def _download_assets(
        self, instance: EvalInstance, workspace: RemoteWorkspace
    ) -> None:
        """
        Download paper assets (images, data files) to the workspace.

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

        for asset in assets:
            try:
                # Download asset using HuggingFace Hub
                cmd = f"wget -P /workspace/assets '{asset}'"
                result = workspace.execute_command(cmd)

                if result.returncode != 0:
                    logger.warning(f"Failed to download asset: {asset}")
            except Exception as e:
                logger.warning(f"Error downloading asset {asset}: {e}")

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

    args = parser.parse_args()

    # Load LLM configuration
    with open(args.llm_config, "r") as f:
        llm_config = f.read()
    llm = LLM.model_validate_json(llm_config)

    # Create metadata
    metadata = EvalMetadata(
        llm=llm,
        dataset=args.dataset,
        dataset_split=args.split,
        max_iterations=args.max_iterations,
        eval_output_dir=args.output_dir,
        eval_limit=args.n_limit,
        workspace_type=args.workspace_type,
        details={
            "server_image": args.server_image,
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

    # Run evaluation
    output_file = output_dir / "output.jsonl"
    logger.info(f"Starting evaluation, results will be written to {output_file}")

    evaluator.run(on_result=get_default_on_result_writer(str(output_file)))

    logger.info("Evaluation complete!")


if __name__ == "__main__":
    main()
