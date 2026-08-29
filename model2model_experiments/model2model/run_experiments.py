import asyncio
import sys
import json
import math

if sys.platform == "win32":
    # Use SelectorEventLoop for Windows compatibility with psycopg
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
import httpx
import argparse
import logging
from pathlib import Path
from datetime import datetime
import sys
import pandas as pd
from typing import Dict, List, Tuple


from model2model.make_experiments import MatchupGenerator, load_config
from persuasio.datatypes.api import SessionParameters
from persuasio.datatypes.enums import SpeakerOrder, ModelName
from persuasio.utils.api_dependencies import session_parameters


PERSUASIO_URL = "http://127.0.0.1:8000"

# Configure logging
log_dir = Path(__file__).parent / "data" / "logs"
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

# Configure dialogue data directory
dialogues_dir = Path(__file__).parent / "data" / "dialogues"
if not dialogues_dir.exists():
    dialogues_dir.resolve().mkdir(parents=True, exist_ok=False)

# Configure experiments directory
experiments_dir = Path(__file__).parent / "data" / "experiments"
if not experiments_dir.exists():
    experiments_dir.resolve().mkdir(parents=True, exist_ok=False)


def load_experiments_from_file(file_path: str) -> pd.DataFrame:
    """Load experiments from CSV, JSON, or JSONL file."""
    logger.info(f"Loading experiments from {file_path}")

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Experiment file not found: {file_path}")

    suffix = path.suffix.lower()

    if suffix == '.csv':
        df = pd.read_csv(path)
    elif suffix == '.jsonl':
        df = pd.read_json(path, lines=True)
    elif suffix == '.json':
        df = pd.read_json(path)
    else:
        raise ValueError(f"Unsupported file format: {suffix}")

    return df

def generate_experiments_from_config(config_path: str) -> List[SessionParameters]:
    """Load experiments by generating them from a config.yaml file."""
    logger.info(f"Generating experiments from config: {config_path}")

    config = load_config(config_path)
    generator = MatchupGenerator(**config)
    matchups = generator.generate_all_matchups()

    return matchups

def success_and_error(batch_results, session_batch : List[SessionParameters]) -> Tuple[int, int]:

    sessions_failed = []
    completed_count = 0
    failed_count = 0

    if "batch" in batch_results:
        # batch has failed
        for session in session_batch:
            sessions_failed.append(session.session_id)
        return (0, len(session_batch), list(set(sessions_failed)))

    
    for result in batch_results:
        if result["success"]:
            completed_count += 1
        else:
            failed_count += 1
            sessions_failed.append(result["session_id"])

    return (completed_count, failed_count, list(set(sessions_failed)))


def check_persuasio_running() -> bool:
    """Check if persuasio server is already running."""
    try:
        response = httpx.get(f"{PERSUASIO_URL}/sessions/ongoing", timeout=5)
        return response.status_code == 200
    except httpx.RequestError:
        return False

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

    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--experiments-file",
        dest="experiments_file",
        help="Path to CSV/JSON/JSONL file with experiment configs"
    )
    source_group.add_argument(
        "--config",
        dest="config_file",
        help="Path to YAML config file for generating experiments"
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

async def fetch(client: httpx.AsyncClient, semaphore : asyncio.Semaphore, session_params: SessionParameters, batch_num : int) -> Dict:
    """Run batch of GET requests."""

    async with semaphore:
        logger.info(f"Batch {batch_num + 1}, running session '{session_params.session_id}'")
        #logger.info(f"Started session: {session_params.session_id}")

        try:

            params_dict = session_params.model_dump()

            for key, value in params_dict.items():
                if hasattr(value, 'value'):
                    params_dict[key] = value.value

            # default ClientResponse values?
            params_dict['utterance_from'] = SpeakerOrder.FIRST_SPEAKER.value

            request = httpx.Request(method="GET", url=f"{PERSUASIO_URL}/sessions/create",params=params_dict)
            response = await client.send(request)

            if response.status_code == 200:
                logger.info(f"Completed session '{session_params.session_id}'")
                response = response.json()

                session_file_name = f"{session_params.session_id}.json".replace(":", "_")
                session_file_path = dialogues_dir / session_file_name
                with open(session_file_path, 'w') as f:
                    json.dump(response, f, indent=4)

                logger.info(f"Session data saved to: {session_file_path}")

                return {
                    "success": True,
                    "session_id": session_params.session_id,
                }
            else:
                logger.error(f"Bad server response for session '{session_params.session_id}': {response.status_code} - {response.text}")
                return {
                    "success": False,
                    "session_id": session_params.session_id,
                    "error": f"HTTP {response.status_code}: {response.text}"
                }
        except Exception as e:
            logger.error(f"Failed running batch {batch_num + 1}: {e}")
            return {
                "success": False,
                "batch": True,
                "error": str(e)
            }

async def main():

    args = parse_args()

    logger.info("=== Persuasio Experiments ===")

    # check/start persuasio
    if check_persuasio_running():
        logger.info("Persuasio server is already running")
    else:
        logger.error("Persuasio server is not running and --start-server not specified")
        sys.exit(1)

    if args.timeout:
        logger.info(f"Session timeout: {args.timeout} seconds")
        TIMEOUT = float(args.timeout)
    else:
        logger.info("Session timeout: no timeout")
        TIMEOUT = None

    # Load experiments based on source type
    try:
        if args.experiments_file:
            experiments_df = load_experiments_from_file(args.experiments_file)

            # Shuffle df
            experiments_df = experiments_df.sample(frac=1)

            # Validate and convert to SessionParameters
            validated_experiments = []
            for idx, row in experiments_df.iterrows():
                try:
                    row_data = row.to_dict()
                    row_data["human_model_name"] = ModelName.NO_MODEL
                    session_params = SessionParameters(**row_data)
                    validated_experiments.append(session_params)
                except Exception as e:
                    logger.error(f"Validation failed for row {idx}: {e}")
                    sys.exit(1)

        elif args.config_file:
            validated_experiments = generate_experiments_from_config(args.config_file)

        else:
            logger.error("No experiment source specified")
            sys.exit(1)

    except Exception as e:
        logger.error(f"Failed to load experiments: {e}")
        sys.exit(1)

    logger.info(f"Loaded {len(validated_experiments)} experiments")

    NUM_EXPERIMENTS = len(validated_experiments)

    BATCH_SIZE = args.batch_size
    total_batches = math.ceil(NUM_EXPERIMENTS / BATCH_SIZE)

    logger.info(f"Total number of batches is {total_batches}")


    semaphore = asyncio.Semaphore(BATCH_SIZE)
    limits = httpx.Limits(max_connections=NUM_EXPERIMENTS)

    experiments_metadata = {
        "total_completed" : 0,
        "total_failed" : 0,
        "which_sessions_failed" : []
    }

    async with httpx.AsyncClient(http2=True, limits=limits) as client:
        client.timeout = TIMEOUT

        all_results = []
        for batch_num in range(total_batches):
            try:
                batch_start = batch_num * BATCH_SIZE
                batch_end = min(batch_start + BATCH_SIZE, NUM_EXPERIMENTS)

                validated_experiments_batch = validated_experiments[batch_start:batch_end]

                logger.info(f"Starting batch {batch_num + 1}/{total_batches}...")
                
                tasks = [
                    fetch(client=client, semaphore=semaphore, session_params=session_params, batch_num=batch_num)
                    for session_params in validated_experiments_batch
                ]
                batch_results = await asyncio.gather(*tasks)

                completed_count, failed_count, sessions_failed = success_and_error(batch_results=batch_results, session_batch=validated_experiments_batch)
                experiments_metadata["total_completed"] += completed_count
                experiments_metadata["total_failed"] += failed_count
                if sessions_failed:
                    logger.error(f"Sessions failed: {sessions_failed}")
                    experiments_metadata["which_sessions_failed"].extend(sessions_failed)
                logger.info(f"Batch {batch_num+1}/{total_batches} completed: {completed_count} succeeded, {failed_count} failed")
                all_results.extend(batch_results)
            except Exception as e:
                logger.error(f"Failed batch {batch_num + 1}: {e}")

    logger.info(f"Experiment run completed: {experiments_metadata['total_completed']} succeeded, {experiments_metadata['total_failed']} failed")
    metadata_file = experiments_dir / "metadata.json"
    with open(metadata_file, "w") as file:
        json.dump(experiments_metadata, file, indent = 2)
    logger.info("Saved experiment metadata")
    logger.info("Shutting down...")

def run():
    asyncio.run(main())

if __name__ == "__main__":
    run()
