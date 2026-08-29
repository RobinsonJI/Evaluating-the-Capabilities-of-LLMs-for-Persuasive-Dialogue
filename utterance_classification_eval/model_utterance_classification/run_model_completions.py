import asyncio
import logging
from pathlib import Path
from datetime import datetime
import argparse
import yaml
import sys
from typing import Dict, Any, List, Tuple
import math
import pandas as pd
import json

from persuasio.models.models import GenerateLLMResponses
from persuasio.datatypes.enums import ModelName
from persuasio.datatypes.pydantic_basemodels import UtteranceClass
from persuasio.prompts.system.utterance_classification import utt_class_sys_msg


from model_utterance_classification.dataset import load_testset_prompts, create_experiments_df

# Configure logging
log_dir = Path(__file__).parent.parent / "data" / "logs"
if not log_dir.exists():
    log_dir.resolve().mkdir(parents=True, exist_ok=False)

log_file = log_dir / f'experiment_run_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments with improved structure."""
    parser = argparse.ArgumentParser(
        description="Run persuasio debate experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --experiments-file experiments.csv
  %(prog)s --config config.yaml --start-server
  %(prog)s --config config.yaml --concurrent 8
"""
    )

    parser.add_argument(
        "--experiments-file",
        dest="experiments_file",
        help="Path to CSV/JSON/JSONL file with experiment configs",
        required=True
    )

    parser.add_argument(
        "--model-config", 
        dest="model_config", 
        type=str, 
        required=True, 
        help="Path to YAML configuration file for models"
    )

    parser.add_argument(
        "--output-name",
        type=str,
        default=None
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=20,
        help="Maximum concurrent sessions (default: %(default)s)"
    )

    
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Timeout in seconds for session requests (default: no timeout)"
    )

    return parser.parse_args()


def load_config(config_filename: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""

    config_dir = Path(__file__).parent.parent / "data" / "config" / config_filename
    with open(config_dir, 'r') as f:
        config = yaml.safe_load(f)
    return config


def parse_models(model_name):
    for model in ModelName:
        if model.value == model_name:
            return model

def create_prompt(data) -> List[Dict[str, str]]:

    sys_msg = utt_class_sys_msg

    human_msg = {
        "role": "user",
        "content": f"""Classify the following sentence according to the defined utterance types:
        
        Previous dialogue turns:
        User: {data["dialogue_turn_1"]}

        Model: {data["dialogue_turn_2"]}
        
        Current sentence:
        User: {data["target"]}"""
    }

    prompt = [sys_msg, human_msg]

    return prompt


def success_and_error(batch_results) -> Tuple[int, int, List[Dict], List[Dict]]:

    experiments_failed = []
    successful_experiments = []
    completed_count = 0
    failed_count = 0
    
    for result in batch_results:
        if result["success"]:
            completed_count += 1
            successful_experiments.append(result["data"])
        else:
            failed_count += 1
            experiments_failed.append(result["data"])

    return (completed_count, failed_count, successful_experiments, experiments_failed)


async def generate(data: Dict, sem: asyncio.Semaphore, batch_num: int):
    try:
        async with sem:

            model = parse_models(data["predictor"])
            prompt = create_prompt(data)

            #print(prompt)

            result = GenerateLLMResponses(
                model_choice=model,
                prompt=prompt,
                temperature=data["temp"],
                top_p=data["top_p"],
                seed=data["seed"],
                datatype_schema=UtteranceClass
            ).return_completion()

            data["classification"] = result["Classification"].replace("_", "").strip().lower()

            return {
                "success" : True,
                "data" : data
                }

    except Exception as e:
            logger.error(f"Failed experiment: {e}")
            return {
                "success": False,
                "data" : data,
                "error": str(e)
            }

async def main():

    args = parse_args()

    if args.output_name is None:
        args.output_name = "classification_results.csv"

    logger.info("=== Utterance Classification Experiments ===")

    test_file = args.experiments_file

    df = load_testset_prompts(file_name=test_file)

    logger.info("Test dataset loaded.")

    model_config = load_config(args.model_config)

    if "models" not in model_config:
        logger.error("Models were not included in the config YAML.")
        sys.exit(1)

    logger.info("Loaded model config.")

    experiments_list = create_experiments_df(test_df=df, model_config=model_config)
    
    logger.info(f"Created and saved {len(df)} experiments dataset.")

    NUM_EXPERIMENTS = len(experiments_list)

    BATCH_SIZE = args.batch_size
    total_batches = math.ceil(NUM_EXPERIMENTS / BATCH_SIZE)

    logger.info(f"Total number of batches is {total_batches}")

    semaphore = asyncio.Semaphore(BATCH_SIZE)

    experiments_metadata = {
        "total_completed" : 0,
        "total_failed" : 0,
        "failed_tests" : []
    }

    logger.info(f"Starting {NUM_EXPERIMENTS} experiments")

    results = []
    tests_to_run_again = []
    for batch_num in range(total_batches):

        logger.info(f"Starting batch {batch_num + 1}/{total_batches}...")

        batch_start = batch_num * BATCH_SIZE
        batch_end = min(batch_start + BATCH_SIZE, NUM_EXPERIMENTS)

        experiments_batch = experiments_list[batch_start:batch_end]

        tasks = [generate(data, semaphore, batch_num) for data in experiments_batch]

        batch_results = await asyncio.gather(*tasks)

        completed_count, failed_count, successful_results, failed_test_data = success_and_error(batch_results)

        results.extend(successful_results)
        experiments_metadata["total_completed"] += completed_count
        experiments_metadata["total_failed"] += failed_count
        if len(failed_test_data) > 0:
            tests_to_run_again.extend(failed_test_data)

        logger.info(f"Batch {batch_num+1}/{total_batches} completed: {completed_count} succeeded, {failed_count} failed")

    logger.info(f"Utterance classification experiment completed: {experiments_metadata['total_completed']} succeeded, {experiments_metadata['total_failed']} failed")

    results_df = pd.DataFrame.from_dict(results)
    save_dir_path = Path(__file__).parent.parent / "data" / "experiments" 
    results_file_path = save_dir_path / args.output_name
    results_df.to_csv(results_file_path, index=False)
    logger.info(f"Saved results to CSV here: '{results_file_path}'")


    save_dir_path = Path(__file__).parent.parent / "data" / "experiments" / "metadata" 
    if not save_dir_path.exists():
            save_dir_path.resolve().mkdir(parents=True, exist_ok=False)

    if len(tests_to_run_again) > 0:

        if not save_dir_path.exists():
            save_dir_path.resolve().mkdir(parents=True, exist_ok=False)

        save_file_path = save_dir_path / 'failed_tests.jsonl'
        failed_df = pd.DataFrame.from_dict(experiments_metadata["failed_tests"])
        failed_df.to_csv(save_dir_path, index=False)

        logger.info(f"Saved failed experiments in the following directory: '{save_dir_path}'. Use that CSV to run the remaining tests.")

        experiments_metadata.pop("failed_tests")

    save_file_path = save_dir_path / "metadata.json"
    with open(save_file_path, "w") as file:
        json.dump(experiments_metadata, file, indent = 2)

    logger.info(f"Saved metadata to {save_file_path}")

    logger.info("Shutting down...")
        
        



def run():
    asyncio.run(main())

if __name__ == "__main__":
    run()
   