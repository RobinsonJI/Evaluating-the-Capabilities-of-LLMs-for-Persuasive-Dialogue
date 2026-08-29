import argparse
import yaml
from typing import List, Dict
import pandas as pd
from datetime import datetime
import asyncio
from dataclasses import dataclass, field
import random
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, precision_recall_fscore_support
from sklearn.exceptions import UndefinedMetricWarning
import krippendorff

import warnings
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

from data import UtteranceDataset
from llm_client import UtteranceClassifier
from data_structures import UtteranceType, Utterance, Dialogue, Prediction, ModelParameters

@dataclass
class Experiment:
    """Comprehensive experiment class for evaluating utterance classification models"""

    # Configuration
    model_name: str
    dataset: UtteranceDataset
    repetitions: int = 5
    model_params: ModelParameters = field(default_factory=ModelParameters)
    max_concurrent: int = 10
    ai_config: Dict = field(default_factory=dict)
    output_dir: str = "data/"

    # Results (populated after run())
    predictions: List[Prediction] = field(default_factory=list)

    def __post_init__(self):
        """Initialize classifier after dataclass creation"""
        self.classifier = UtteranceClassifier(
            model=self.model_name,
            api_key=self.ai_config.get("key"),
            endpoint=self.ai_config.get("endpoint")
        )

    async def run(self, show_progress: bool = False) -> List[Prediction]:
        """Run the complete experiment with repetitions"""
        print(f"Running experiment for {self.model_name} with {self.repetitions} repetitions")

        all_dialogues = self.dataset.all_dialogues()
        if not all_dialogues:
            print("No dialogues found in dataset")
            return []

        # Use the efficient batch repetition method from our refactored classifier
        self.predictions = await self.classifier.classify_dialogues_with_repetitions_batch(
            dialogues=all_dialogues,
            repetitions=self.repetitions,
            max_concurrent=self.max_concurrent,
            params=self.model_params,
            show_progress=show_progress
        )

        print(f"Completed experiment: {len(self.predictions)} predictions")
        return self.predictions


    def get_ground_truth(self) -> List[UtteranceType]:
        """Extract ground truth labels from predictions"""
        return [pred.utterance.utterance_type for pred in self.predictions]

    def get_majority_predictions(self) -> List[UtteranceType]:
        """Extract majority vote predictions"""
        return [pred.label for pred in self.predictions]

    def get_individual_predictions(self) -> List[List[UtteranceType]]:
        """Extract all individual predictions for each utterance"""
        return [pred.r_labels or [] for pred in self.predictions]

    def get_majority_proportions(self) -> List[float]:
        """Extract majority vote proportions (percentage of repetitions that agreed with majority)"""
        proportions = []
        for pred in self.predictions:
            if pred.r_labels and len(pred.r_labels) > 0:
                proportion = pred.r_labels.count(pred.label) / len(pred.r_labels)
                proportions.append(proportion)
            else:
                proportions.append(1.0)  # Single prediction case
        return proportions

    def compute_accuracy(self) -> float:
        """Compute accuracy using majority vote predictions"""
        if not self.predictions:
            return 0.0

        ground_truth = self.get_ground_truth()
        predictions = self.get_majority_predictions()

        return accuracy_score(ground_truth, predictions)

    def compute_confusion_matrix(self) -> np.ndarray:
        """Compute confusion matrix"""
        if not self.predictions:
            return np.array([])

        ground_truth = [gt.value for gt in self.get_ground_truth()]
        predictions = [pred.value for pred in self.get_majority_predictions()]

        # Get all possible labels
        all_labels = sorted(list(set(ground_truth + predictions)))

        return confusion_matrix(ground_truth, predictions, labels=all_labels)

    def compute_f1_scores(self) -> Dict:
        """Compute F1 scores (macro and micro)"""
        if not self.predictions:
            return {"macro": 0.0, "micro": 0.0}

        ground_truth = [gt.value for gt in self.get_ground_truth()]
        predictions = [pred.value for pred in self.get_majority_predictions()]

        # Filter out predictions not in ground truth types (for more meaningful evaluation)
        ground_truth_types = list(set(ground_truth))
        filtered_gt, filtered_pred = [], []

        for gt, pred in zip(ground_truth, predictions):
            if pred in ground_truth_types:
                filtered_gt.append(gt)
                filtered_pred.append(pred)

        if not filtered_gt:
            return {"macro": 0.0, "micro": 0.0}

        return {
            "macro": f1_score(filtered_gt, filtered_pred, average="macro"),
            "micro": f1_score(filtered_gt, filtered_pred, average="micro"),
            "raw_macro": f1_score(ground_truth, predictions, average="macro"),
            "raw_micro": f1_score(ground_truth, predictions, average="micro")
        }

    def compute_per_class_metrics(self) -> Dict:
        """Compute per-class precision, recall, and F1 scores"""
        if not self.predictions:
            return {}

        ground_truth = [gt.value for gt in self.get_ground_truth()]
        predictions = [pred.value for pred in self.get_majority_predictions()]

        # Get all unique labels
        all_labels = sorted(list(set(ground_truth + predictions)))

        # Compute per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            ground_truth, predictions, labels=all_labels, average=None, zero_division=0
        )

        # Create per-class results
        per_class = {}
        for i, label in enumerate(all_labels):
            per_class[label] = {
                "precision": precision[i],
                "recall": recall[i],
                "f1": f1[i],
                "support": int(support[i])
            }

        return per_class

    def compute_krippendorff_alpha(self) -> float:
        """Compute Krippendorff's alpha for inter-annotator agreement"""
        if not self.predictions or self.repetitions < 2:
            return 0.0

        # Prepare data matrix for Krippendorff's alpha
        # Rows are annotators (repetitions), columns are utterances
        individual_preds = self.get_individual_predictions()

        category_map = {ut.value: idx for idx, ut in enumerate(UtteranceType)}
        
        dummy_preds = np.array([
            [category_map[pred.value] for pred in row]
            for row in individual_preds
        ]).T

        try:
            alpha = krippendorff.alpha(dummy_preds, level_of_measurement="nominal")
            return alpha if not np.isnan(alpha) else 0.0
        except Exception as e:
            print(f"Error computing Krippendorff's alpha: {e}")
            return 0.0

    def bootstrap_evaluate(self, n_bootstrap: int = 10000) -> Dict:
        """Perform bootstrap evaluation for confidence intervals"""
        if not self.predictions:
            return {}

        ground_truth = self.get_ground_truth()
        predictions = self.get_majority_predictions()

        n_samples = len(ground_truth)
        bootstrap_accuracies = []
        bootstrap_f1_macros = []
        bootstrap_f1_micros = []

        for _ in range(n_bootstrap):
            # Bootstrap sample
            indices = np.random.choice(n_samples, size=n_samples, replace=True)

            bootstrap_gt = [ground_truth[i].value for i in indices]
            bootstrap_pred = [predictions[i].value for i in indices]

            # Compute metrics
            acc = accuracy_score(bootstrap_gt, bootstrap_pred)

            # Filter for meaningful F1 computation
            gt_types = list(set(bootstrap_gt))
            filtered_gt, filtered_pred = [], []
            for gt, pred in zip(bootstrap_gt, bootstrap_pred):
                if pred in gt_types:
                    filtered_gt.append(gt)
                    filtered_pred.append(pred)

            if filtered_gt:
                f1_macro = f1_score(filtered_gt, filtered_pred, average="macro")
                f1_micro = f1_score(filtered_gt, filtered_pred, average="micro")
            else:
                f1_macro = 0.0
                f1_micro = 0.0

            bootstrap_accuracies.append(acc)
            bootstrap_f1_macros.append(f1_macro)
            bootstrap_f1_micros.append(f1_micro)

        # Compute confidence intervals
        def compute_ci(values, confidence=0.95):
            alpha = 1 - confidence
            lower = np.percentile(values, 100 * alpha / 2)
            upper = np.percentile(values, 100 * (1 - alpha / 2))
            return lower, upper

        return {
            "accuracy": {
                "mean": np.mean(bootstrap_accuracies),
                "std": np.std(bootstrap_accuracies),
                "ci_95": compute_ci(bootstrap_accuracies)
            },
            "f1_macro": {
                "mean": np.mean(bootstrap_f1_macros),
                "std": np.std(bootstrap_f1_macros),
                "ci_95": compute_ci(bootstrap_f1_macros)
            },
            "f1_micro": {
                "mean": np.mean(bootstrap_f1_micros),
                "std": np.std(bootstrap_f1_micros),
                "ci_95": compute_ci(bootstrap_f1_micros)
            }
        }

    def save_results(self) -> str:
        """Save experiment results to CSV"""
        if not self.predictions:
            return ""

        results_data = []
        for pred in self.predictions:
            # Get flattened utterance data
            row = pred.dialogue.flatten()
            # Calculate majority proportion
            if pred.r_labels and len(pred.r_labels) > 0:
                majority_proportion = pred.r_labels.count(pred.label) / len(pred.r_labels)
            else:
                majority_proportion = 1.0  # Single prediction case

            row.update({
                "model_name": pred.model_name,
                "predicted_type": pred.label.value,
                "ground_truth": pred.utterance.utterance_type.value,
                "is_correct": pred.label == pred.utterance.utterance_type,
                "individual_predictions": ",".join([p.value for p in pred.r_labels or []]),
                "individual_seeds": ",".join(map(str, pred.individual_seeds or [])),
                "majority_proportion": majority_proportion
            })
            results_data.append(row)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.output_dir}/experiment_results_{self.model_name.replace('/', '_')}_{timestamp}.csv"

        df = pd.DataFrame(results_data)
        df.to_csv(filename, index=False)

        return filename

    @classmethod
    def load_results(cls, filepath: str) -> 'Experiment':
        """Load experiment results from CSV file and reconstruct Experiment object"""
        df = pd.read_csv(filepath)

        if df.empty:
            raise ValueError("CSV file is empty")

        # Extract model name from first row
        model_name = df.iloc[0]["model_name"]

        # Create instance without calling __init__ or __post_init__
        instance = object.__new__(cls)

        # Manually set required fields
        instance.model_name = model_name
        instance.dataset = None  # Not needed for analysis-only instance
        instance.repetitions = 5  # Default value
        instance.model_params = ModelParameters()
        instance.max_concurrent = 10
        instance.ai_config = {}
        instance.output_dir = "data/"
        instance.predictions = []

        # Reconstruct predictions from CSV data
        predictions = []
        for _, row in df.iterrows():
            # Parse individual predictions and seeds
            r_labels = []
            if pd.notna(row.get("individual_predictions")) and row["individual_predictions"]:
                r_labels = [UtteranceType(label) for label in row["individual_predictions"].split(",")]

            individual_seeds = []
            if pd.notna(row.get("individual_seeds")) and row["individual_seeds"]:
                individual_seeds = [int(seed) for seed in str(row["individual_seeds"]).split(",")]

            # Reconstruct dialogue from flattened data
            dialogue = Dialogue.from_dict(row.to_dict())

            # Create prediction object
            prediction = Prediction(
                utterance=dialogue.data[-1],
                dialogue=dialogue,
                label=UtteranceType(row["predicted_type"]),
                model_params=ModelParameters(),
                model_name=row["model_name"],
                r_labels=r_labels if r_labels else None,
                individual_seeds=individual_seeds if individual_seeds else None
            )

            predictions.append(prediction)

        instance.predictions = predictions
        return instance


def print_confusion_matrix(cm: np.ndarray, labels: List[str]) -> None:
    """Print a formatted confusion matrix with row/column labels"""
    if cm.size == 0:
        print("No confusion matrix data available")
        return

    header = "True\\Pred"
    print(f"{header:<12}", end="")
    for label in labels:
        print(f"{label:<10}", end="")
    print()

    for i, true_label in enumerate(labels):
        print(f"{true_label:<12}", end="")
        for j in range(len(labels)):
            print(f"{cm[i][j]:<10}", end="")
        print()


async def run_experiments(experiments: List[Experiment], show_progress: bool = False) -> List[Experiment]:
    """Run multiple experiments in parallel"""
    print(f"Running {len(experiments)} experiments in parallel")

    tasks = [exp.run(show_progress=show_progress) for exp in experiments]
    await asyncio.gather(*tasks)

    return experiments


def print_experiment_results(experiment: Experiment, bootstrap: bool = False):
    """Print comprehensive results for a single experiment"""
    print(f"\n{'='*80}")
    print(f"Results for {experiment.model_name}")
    print(f"{'='*80}")

    if not experiment.predictions:
        print("No predictions to analyze")
        return

    # Calculate metrics
    accuracy = experiment.compute_accuracy()
    f1_scores = experiment.compute_f1_scores()
    krippendorff_alpha = experiment.compute_krippendorff_alpha()
    per_class_metrics = experiment.compute_per_class_metrics()
    majority_proportions = experiment.get_majority_proportions()

    # Overall summary
    print(f"\nDataset Size: {len(experiment.predictions)} samples")
    print(f"Repetitions: {experiment.repetitions}")

    # Majority proportion statistics
    if majority_proportions:
        avg_confidence = np.mean(majority_proportions)
        min_confidence = np.min(majority_proportions)
        low_confidence_count = sum(1 for p in majority_proportions if p < 0.8)
        print(f"Avg Majority Confidence: {avg_confidence:.3f}")
        print(f"Min Majority Confidence: {min_confidence:.3f}")
        print(f"Low Confidence Predictions (<80%): {low_confidence_count}")

    # Overall performance metrics
    print(f"\n{'─'*60}")
    print("OVERALL PERFORMANCE")
    print(f"{'─'*60}")

    # Compute macro averages
    ground_truth = [gt.value for gt in experiment.get_ground_truth()]
    predictions = [pred.value for pred in experiment.get_majority_predictions()]

    # Compute macro precision, recall, f1
    from sklearn.metrics import precision_recall_fscore_support
    precision, recall, f1, _ = precision_recall_fscore_support(
        ground_truth, predictions, average='macro', zero_division=0
    )

    overall_data = [{
        'Metric': 'Accuracy',
        'Value': f"{accuracy:.3f}"
    }, {
        'Metric': 'Macro Precision',
        'Value': f"{precision:.3f}"
    }, {
        'Metric': 'Macro Recall',
        'Value': f"{recall:.3f}"
    }, {
        'Metric': 'Macro F1-Score',
        'Value': f"{f1:.3f}"
    }, {
        'Metric': 'Krippendorff α',
        'Value': f"{krippendorff_alpha:.3f}"
    }]

    overall_df = pd.DataFrame(overall_data)
    print(overall_df.to_string(index=False))

    # Per-class metrics table
    print(f"\n{'─'*70}")
    print("PER-CLASS PERFORMANCE")
    print(f"{'─'*70}")

    if per_class_metrics:
        class_data = []
        for class_name, metrics in sorted(per_class_metrics.items()):
            class_data.append({
                'Class': class_name,
                'Precision': f"{metrics['precision']:.3f}",
                'Recall': f"{metrics['recall']:.3f}",
                'F1-Score': f"{metrics['f1']:.3f}",
                'Support': metrics['support']
            })

        class_df = pd.DataFrame(class_data)
        print(class_df.to_string(index=False))
    else:
        print("No per-class metrics available")

    # Confusion Matrix
    print(f"\n{'─'*50}")
    print("CONFUSION MATRIX")
    print(f"{'─'*50}")

    cm = experiment.compute_confusion_matrix()
    if cm.size > 0:
        ground_truth = [gt.value for gt in experiment.get_ground_truth()]
        predictions = [pred.value for pred in experiment.get_majority_predictions()]
        all_labels = sorted(list(set(ground_truth + predictions)))

        # Create confusion matrix DataFrame
        cm_df = pd.DataFrame(cm, index=all_labels, columns=all_labels)
        cm_df.index.name = 'True'
        cm_df.columns.name = 'Predicted'
        print(cm_df.to_string())
    else:
        print("No confusion matrix data available")

    # Bootstrap evaluation (only if requested)
    if bootstrap:
        print(f"\n{'─'*60}")
        print("BOOTSTRAP EVALUATION (95% CI)")
        print(f"{'─'*60}")

        bootstrap_results = experiment.bootstrap_evaluate(n_bootstrap=10000)
        bootstrap_data = []

        for metric, stats in bootstrap_results.items():
            mean_val = stats["mean"]
            ci_lower, ci_upper = stats["ci_95"]
            bootstrap_data.append({
                'Metric': metric.replace('_', ' ').title(),
                'Mean': f"{mean_val:.3f}",
                'Std': f"{stats['std']:.3f}",
                '95% CI': f"[{ci_lower:.3f}, {ci_upper:.3f}]"
            })

        if bootstrap_data:
            bootstrap_df = pd.DataFrame(bootstrap_data)
            print(bootstrap_df.to_string(index=False))


async def main(n_samples: int, max_concurrent: int, models: List[str], repetitions: int, bootstrap: bool = False, show_progress: bool = False):
    """Main evaluation function"""
    print(f"Evaluating utterance classification with {models}, {repetitions} repetitions")
    print(f"Extracting up to {n_samples} utterances per type from Neo4j")

    # Load config
    config = yaml.safe_load(open("config.yaml"))
    neo4j_params = config.get("neo4j", {})
    ai_config = config.get("ai", {})

    # Get dataset
    dataset = UtteranceDataset(n=n_samples, neo4j_params=neo4j_params)

    print(f"\nTotal utterances: {len(dataset.all_final_utterances())}")
    print(f"Type breakdown: {dataset.counts_by_type}")

    if not dataset.all_final_utterances():
        print("No utterances found in dataset")
        return

    # Create experiments for each model
    experiments = []
    for model in models:
        experiment = Experiment(
            model_name=model,
            dataset=dataset,
            repetitions=repetitions,
            max_concurrent=max_concurrent,
            ai_config=ai_config,
            output_dir="data/"
        )
        experiments.append(experiment)

    # Run all experiments in parallel
    completed_experiments = await run_experiments(experiments, show_progress=show_progress)

    # Print results for each experiment
    for experiment in completed_experiments:
        print_experiment_results(experiment, bootstrap)

        # Save results
        filename = experiment.save_results()
        if filename:
            print(f"Results saved to: {filename}")

    # Compare experiments if multiple models
    if len(completed_experiments) > 1:
        print("\n=== Model Comparison ===")
        comparison_data = []
        for exp in completed_experiments:
            if exp.predictions:
                accuracy = exp.compute_accuracy()
                f1_scores = exp.compute_f1_scores()
                alpha = exp.compute_krippendorff_alpha()
                comparison_row = {
                    "Model": exp.model_name,
                    "Accuracy": f"{accuracy:.3f}",
                    "F1 Macro": f"{f1_scores.get('macro', 0):.3f}",
                    "F1 Micro": f"{f1_scores.get('micro', 0):.3f}",
                    "Krippendorff α": f"{alpha:.3f}"
                }

                # Add bootstrap column only if requested
                if bootstrap:
                    bootstrap_results = exp.bootstrap_evaluate(n_bootstrap=1000)  # Smaller for comparison
                    comparison_row["Bootstrap Acc (95% CI)"] = f"{bootstrap_results.get('accuracy', {}).get('mean', 0):.3f} [{bootstrap_results.get('accuracy', {}).get('ci_95', (0, 0))[0]:.3f}, {bootstrap_results.get('accuracy', {}).get('ci_95', (0, 0))[1]:.3f}]"

                comparison_data.append(comparison_row)

        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data)
            print(comparison_df.to_string(index=False))

            # Save comparison
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            comparison_filename = f"model_comparison_{timestamp}.csv"
            comparison_df.to_csv(comparison_filename, index=False)
            print(f"\nComparison saved to: {comparison_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate utterance classification")
    parser.add_argument(
        "--n-samples",
        "-n",
        type=int,
        help="Number of samples per utterance type to extract from Neo4j",
        required=True
    )
    parser.add_argument(
        "--max-concurrent",
        "-c",
        type=int,
        default=10,
        help="Maximum concurrent API calls (default: 10)"
    )
    parser.add_argument(
        "--models",
        "-m",
        type=str,
        nargs="+",
        help="Models to use for classification (can specify multiple)",
        required=True
    )
    parser.add_argument(
        "--repetitions",
        "-r",
        type=int,
        help="Number of times to repeat the classification (default: 5)",
        default=5
    )
    parser.add_argument(
        "--bootstrap",
        "-b",
        action="store_true",
        help="Include bootstrap evaluation for confidence intervals (default: false)"
    )
    parser.add_argument(
        "--progress",
        "-p",
        action="store_true",
        help="Show progress bar during processing (requires tqdm)"
    )
    args = parser.parse_args()

    asyncio.run(main(args.n_samples, args.max_concurrent, args.models, args.repetitions, args.bootstrap, args.progress))