import json
import os
from datetime import datetime
from typing import Dict, List

import numpy as np
import torch
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import logging
from pykeen.datasets import PathDataset
from pykeen.evaluation import RankBasedEvaluator
from pykeen.pipeline import pipeline
from pykeen.training import SLCWATrainingLoop

from src.models.registry import build_model


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_experiments(
    data_root: str,
    model_names: List[str],
    embedding_dim: int,
    batch_size: int,
    num_epochs: int,
    learning_rate: float,
    random_seed: int,
    output_dir: str,
) -> Dict[str, Dict[str, float]]:
    """Train and evaluate multiple models on the same dataset and hyperparameters.

    Returns a mapping of model_name -> metrics dict (filtered MR, MRR, Hits@1/3/10).
    """
    _set_seed(random_seed)

    # Expect pre-converted TSVs under data_root/pykeen/
    train_path = os.path.join(data_root, "pykeen", "train.tsv")
    valid_path = os.path.join(data_root, "pykeen", "valid.tsv")
    test_path = os.path.join(data_root, "pykeen", "test.tsv")

    if not (os.path.exists(train_path) and os.path.exists(valid_path) and os.path.exists(test_path)):
        raise FileNotFoundError(
            "Expected pre-converted TSVs at pykeen/train.tsv, pykeen/valid.tsv, pykeen/test.tsv under the dataset root."
        )

    dataset = PathDataset(training_path=train_path, testing_path=test_path, validation_path=valid_path)

    all_results: Dict[str, Dict[str, float]] = {}

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base_run_dir = os.path.join(output_dir, f"runs_{timestamp}")
    os.makedirs(base_run_dir, exist_ok=True)

    for model_name in model_names:
        model_cls, model_kwargs = build_model(model_name, embedding_dim)

        # Prepare per-model run directory and logger
        run_dir = os.path.join(base_run_dir, model_name)
        os.makedirs(run_dir, exist_ok=True)
        _configure_logger(run_dir)

        try:
            if model_name == "TransR":
                # Special handling for TransR: train on MPS, evaluate on CPU
                logging.getLogger(__name__).info("Training TransR on MPS, will evaluate on CPU")
                
                # Train on MPS using lower-level API to avoid evaluation
                training_device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
                
                # Create model with required triples_factory parameter
                model = model_cls(triples_factory=dataset.training, **model_kwargs)
                model = model.to(training_device)
                
                # Create training loop
                training_loop = SLCWATrainingLoop(
                    model=model,
                    triples_factory=dataset.training,
                    optimizer=torch.optim.Adam(model.parameters(), lr=learning_rate),
                )
                
                # Train the model
                training_loop.train(
                    triples_factory=dataset.training,
                    num_epochs=num_epochs,
                    batch_size=batch_size,
                )
                
                # Create a mock result object for compatibility
                class MockResult:
                    def __init__(self, model, dataset):
                        self.model = model
                        self.dataset = dataset
                        self.metric_results = None
                    
                    def save_to_directory(self, directory):
                        # Save model and dataset info
                        torch.save(self.model.state_dict(), os.path.join(directory, "model.pt"))
                        with open(os.path.join(directory, "dataset_info.json"), "w") as f:
                            json.dump({
                                "num_entities": self.dataset.num_entities,
                                "num_relations": self.dataset.num_relations,
                            }, f)
                
                result = MockResult(model, dataset)
                
                # Evaluation will be done separately in metrics collection
                
            else:
                # Normal pipeline for other models
                device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
                
                result = pipeline(
                dataset=dataset,
                model=model_cls,
                model_kwargs=model_kwargs,
                training_loop=SLCWATrainingLoop,
                training_kwargs={
                    "batch_size": batch_size,
                    "num_epochs": num_epochs,
                },
                optimizer="adam",
                optimizer_kwargs={"lr": learning_rate},
                evaluator=RankBasedEvaluator(),
                random_seed=random_seed,
                device=device,
                )
        except ValueError as e:
            # If a model is unsupported by installed PyKEEN, log and skip
            logging.getLogger(__name__).error("Skipping %s due to error: %s", model_name, e)
            continue

        # Collect filtered metrics from test set evaluator
        if model_name == "TransR":
            # Metrics already collected in _evaluate_transr_on_cpu
            metrics = _evaluate_transr_on_cpu(result, dataset, run_dir)
        else:
            # PyKEEN 1.11 uses nested results; flatten and extract 'both.realistic.*'
            flat = result.metric_results.to_flat_dict()
            def _get(key: str) -> float:
                return float(flat.get(key, np.nan))

            metrics = {
                "mr": _get("both.realistic.arithmetic_mean_rank"),
                "mrr": _get("both.realistic.mean_reciprocal_rank"),
                "hits_at_1": _get("both.realistic.hits_at_1"),
                "hits_at_3": _get("both.realistic.hits_at_3"),
                "hits_at_10": _get("both.realistic.hits_at_10"),
            }

        # Persist artifacts
        result.save_to_directory(run_dir)
        if model_name != "TransR":  # Metrics already saved for TransR
            with open(os.path.join(run_dir, "metrics.json"), "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2)

        all_results[model_name] = metrics

    # Also save aggregated results
    with open(os.path.join(base_run_dir, "aggregated_results.json"), "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    # Save CSV summary
    _export_results_csv(all_results, base_run_dir)

    # Save PDF report with definitions
    _export_results_pdf(all_results, base_run_dir)

    return all_results


def _configure_logger(run_dir: str) -> None:
    """Configure logging to file and console for the current model run.

    Creates/overwrites train.log in the given run directory.
    """
    log_path = os.path.join(run_dir, "train.log")

    # Root logger setup
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Remove existing handlers to avoid duplicate logs across models
    for h in list(logger.handlers):
        logger.removeHandler(h)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)


def _export_results_csv(results: Dict[str, Dict[str, float]], out_dir: str) -> None:
    rows = []
    for model_name, metrics in results.items():
        row = {"model": model_name}
        row.update(metrics)
        rows.append(row)
    df = pd.DataFrame(rows, columns=[
        "model", "mrr", "mr", "hits_at_1", "hits_at_3", "hits_at_10"
    ])
    csv_path = os.path.join(out_dir, "aggregated_results.csv")
    df.to_csv(csv_path, index=False)


def _export_results_pdf(results: Dict[str, Dict[str, float]], out_dir: str) -> None:
    pdf_path = os.path.join(out_dir, "report.pdf")
    # Prepare table data
    header = ["Model", "MRR", "MR", "Hits@1", "Hits@3", "Hits@10"]
    table_data = [header]
    for model_name, m in results.items():
        table_data.append([
            model_name,
            f"{m.get('mrr', float('nan')):.4f}",
            f"{m.get('mr', float('nan')):.1f}",
            f"{m.get('hits_at_1', float('nan')):.4f}",
            f"{m.get('hits_at_3', float('nan')):.4f}",
            f"{m.get('hits_at_10', float('nan')):.4f}",
        ])

    definitions = (
        "Metric definitions (filtered setting):\n"
        "- MRR: Mean Reciprocal Rank. Average of 1/rank over test queries. Higher is better.\n"
        "- MR: Mean Rank. Average rank; lower is better.\n"
        "- Hits@k: Proportion of test queries where the correct entity is ranked in top-k. Higher is better.\n"
        "Ranking uses the filtered protocol (corruptions that are known true are removed)."
    )

    with PdfPages(pdf_path) as pdf:
        # Page 1: Title and definitions
        fig1, ax1 = plt.subplots(figsize=(8.5, 11))
        ax1.axis('off')
        title = "PyKEEN Link Prediction Results"
        ax1.text(0.5, 0.9, title, ha='center', va='center', fontsize=18, weight='bold')
        ax1.text(0.05, 0.8, definitions, ha='left', va='top', fontsize=11, wrap=True)
        pdf.savefig(fig1, bbox_inches='tight')
        plt.close(fig1)

        # Page 2: Results table
        fig2, ax2 = plt.subplots(figsize=(11, 8.5))
        ax2.axis('off')
        table = ax2.table(cellText=table_data[1:], colLabels=table_data[0], loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        pdf.savefig(fig2, bbox_inches='tight')
        plt.close(fig2)


def _evaluate_transr_on_cpu(result, dataset, run_dir):
    """Evaluate TransR model on CPU to avoid MPS tensor dimension limits."""
    from pykeen.evaluation import RankBasedEvaluator
    
    # Move model to CPU
    model = result.model
    model.cpu()
    
    # Create evaluator
    evaluator = RankBasedEvaluator()
    
    # Evaluate on CPU
    logging.getLogger(__name__).info("Starting TransR evaluation on CPU...")
    metric_results = evaluator.evaluate(
        model=model,
        mapped_triples=dataset.testing.mapped_triples,
        additional_filter_triples=[
            dataset.training.mapped_triples,
            dataset.validation.mapped_triples,
        ],
        batch_size=32,  # Use smaller batch size for CPU evaluation
    )
    
    # Extract metrics
    flat = metric_results.to_flat_dict()
    def _get(key: str) -> float:
        return float(flat.get(key, np.nan))

    metrics = {
        "mr": _get("both.realistic.arithmetic_mean_rank"),
        "mrr": _get("both.realistic.mean_reciprocal_rank"),
        "hits_at_1": _get("both.realistic.hits_at_1"),
        "hits_at_3": _get("both.realistic.hits_at_3"),
        "hits_at_10": _get("both.realistic.hits_at_10"),
    }
    
    # Save metrics
    with open(os.path.join(run_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    
    logging.getLogger(__name__).info("TransR evaluation completed on CPU")
    return metrics


