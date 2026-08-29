import pandas as pd
from pathlib import Path
import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Any
import argparse
import yaml
import logging
from datetime import datetime
import sys
import json
import warnings
warnings.filterwarnings("ignore")

from persuasio.datatypes.enums import ModelName


# Configure logging
log_dir = Path(__file__).parent.parent / "data" / "logs"
if not log_dir.exists():
    log_dir.resolve().mkdir(parents=True, exist_ok=False)

log_file = log_dir / f'bootstrap_run_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass 
class BootstrapSampleData:
    model: str
    confusion_matrix: List[List[int]]
    f1_scores: List[float]
    macro_f1: float
    weighted_f1: float
    micro_f1:float
    classes: List[str]


def confusion_matrix_np(y_true, y_pred):
    classes = np.unique(np.concatenate((y_true, y_pred)))
    conf_matrix = np.zeros((len(classes), len(classes)), dtype=int)
    class_to_index = {cls: idx for idx, cls in enumerate(classes)}
    for t, p in zip(y_true, y_pred):
        conf_matrix[class_to_index[t], class_to_index[p]] += 1
    return conf_matrix, classes


def load_config(config_filename: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""

    config_dir = Path(__file__).parent.parent / "data" / "config" / config_filename
    with open(config_dir, 'r') as f:
        config = yaml.safe_load(f)
    return config


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
        help="Name of experiments file which contains model classifications and labels",
        required=True
    )

    parser.add_argument(
        "--bootstrap-config", 
        dest="bootstrap_config", 
        type=str, 
        required=True, 
        help="Name of YAML configuration file for bootstrap"
    )

    parser.add_argument(
        "--output-name",
        type=str,
        default=None
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None
    )

    return parser.parse_args()


def main():

    args = parse_args()

    if args.output_dir is None:
        save_dir = Path(__file__).parent.parent / "data" / "experiments" / "bootstrap"
        if not save_dir.exists():
            save_dir.resolve().mkdir(parents=True, exist_ok=False)

    if args.output_name is None:
        save_filename = "bootstrap_results.jsonl"


    logger.info("=== Utterance Classification Bootstrap Experiments ===")

    bootstrap_config = load_config(args.bootstrap_config)

    if "number_of_bootstrap_dfs" not in bootstrap_config:
        logger.error("Number of bootstrap datasets was not included in the config YAML.")
        sys.exit(1)

    logger.info("Loaded bootstrap config.")

    N = bootstrap_config["number_of_bootstrap_dfs"]

    logger.info(f"{N} bootstrap datasets will be generated")

    classification_dir = Path(__file__).parent.parent / "data" / "experiments" 
    classification_file = classification_dir / args.experiments_file
    df = pd.read_csv(classification_file)
    logger.info("Loaded classification results file.")

    df_no_why = df.copy()
    df_no_why = df_no_why[(df_no_why.label != "why") & (df_no_why.classification != "why")]
    df_no_question = df.copy()
    df_no_question = df_no_question[(df_no_question.label != "question") & (df_no_question.classification != "question")]
    df_no_question_and_why = df.copy()
    df_no_question_and_why = df_no_question_and_why[(df_no_question_and_why.label != "why") & (df_no_question_and_why.label != "question") & (df_no_question_and_why.classification != "why") & (df_no_question_and_why.classification != "question")]

    dfs = [df, df_no_why, df_no_question, df_no_question_and_why]
    test_file_names = ["full", "no_why", "no_question", "no_why_no_question"]

    for i in range(N):
        for file_name, _df in zip(test_file_names, dfs):
            bootstrap_df = _df.sample(n=len(_df), replace=True)
            for model_name, group in bootstrap_df.groupby("predictor"):
                y_true = group["label"].to_numpy()
                y_pred = group["classification"].to_numpy()

                conf_matrix, classes = confusion_matrix_np(y_true, y_pred)

                TP = np.diag(conf_matrix)
                FP = np.sum(conf_matrix, axis=0) - TP
                FN = np.sum(conf_matrix, axis=1) - TP
                precision = TP / (TP + FP)
                recall = TP / (TP + FN)
                f1 = 2 * precision * recall / (precision + recall)
                f1 = np.nan_to_num(f1, nan=0.0)

                support = np.sum(conf_matrix, axis=1)
                f1_weighted = np.sum(f1 * support) / np.sum(support)

                TP_micro = np.sum(TP)
                FP_micro = np.sum(FP)
                FN_micro = np.sum(FN)

                precision_micro = TP_micro / (TP_micro + FP_micro)
                recall_micro = TP_micro / (TP_micro + FN_micro)
                f1_micro = 2 * precision_micro * recall_micro / (precision_micro + recall_micro)

                result = BootstrapSampleData(
                    model=model_name, 
                    confusion_matrix=conf_matrix.tolist(), 
                    f1_scores=f1.tolist(), 
                    macro_f1=np.mean(f1).tolist(),
                    weighted_f1=f1_weighted.tolist(),
                    micro_f1=f1_micro.tolist(),
                    classes=classes.tolist()
                    )
                
                save_name = file_name + "_" + save_filename
                
                save_file_path = save_dir / save_name
                if i == 0:
                    with open(save_file_path, "w") as f:
                        f.write(json.dumps(result.__dict__) + "\n")
                else:
                    with open(save_file_path, "a") as f:
                        f.write(json.dumps(result.__dict__) + "\n")
            

        logger.info(f"Computed metrics for bootstrap dataset {i}")


    for file_name in test_file_names:
        model_f1s = {model.value : {
            "macro_f1" : [],
            "weighted_f1" : [],
            "micro_f1" : []
        } for model in ModelName if model.value != "gpt-5" and model.value != "" and model.value != "gpt-4o-mini"}

        logger.info(f"Creating {file_name} histogram data")

        save_name = file_name + "_" + save_filename
        save_file_path = save_dir / save_name
        with open(save_file_path, "r") as f:
            for line in f:
                datapoint = json.loads(line.strip())
                model = datapoint["model"]
                model_f1s[model]["macro_f1"].append(datapoint["macro_f1"])
                model_f1s[model]["micro_f1"].append(datapoint["micro_f1"])
                model_f1s[model]["weighted_f1"].append(datapoint["weighted_f1"])

        save_name = file_name + "_histogram_data.json"
        histogram_file = save_dir / save_name
        with open(histogram_file, "w") as f:
            json.dump(model_f1s, f, indent=4)

        logger.info(f"Histogram data computed and saved for {file_name}")


        cis = {model: {} for model in model_f1s}
        alpha = 0.05
        for model, values in model_f1s.items():
            for f1_type, bootstrap_f1s in values.items():
                lower = np.percentile(bootstrap_f1s, 100 * (alpha/2))
                upper = np.percentile(bootstrap_f1s, 100 * (1 - alpha/2))
                cis[model][f1_type] = {"mean" : np.mean(bootstrap_f1s), 
                                    "std" : np.std(bootstrap_f1s), 
                                    "lower" : lower, 
                                    "upper" : upper, 
                                    "difference" : upper - lower}

        save_name = file_name + "_summary_stats.json"
        confidence_interval_file = save_dir / save_name
        with open(confidence_interval_file, "w") as f:
            json.dump(cis, f, indent=4)

        logger.info(f"95% confidence intervals computed and saved for {file_name}")


    logger.info("Shutting down...")


if __name__ == "__main__":
    main()