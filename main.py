import argparse
import os
from pathlib import Path

from src.utils.experiment import run_experiments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run PyKEEN experiments on OpenKE-formatted datasets."
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=str(
            Path(__file__).parent / "Data" / "FB15K237"
        ),
        help="Path to the dataset directory containing OpenKE files (entity2id.txt, relation2id.txt, train2id.txt, valid2id.txt, test2id.txt)",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="*",
        default=["TransE", "TransR", "Analogy", "HolE", "RotatE"],
        help="List of model names to train. Supported: TransE, TransR, Analogy, HolE, RotatE",
    )
    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=200,
        help="Embedding dimensionality for all models.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1024,
        help="Batch size for training.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help="Learning rate for optimizer.",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(Path("outputs")),
        help="Directory to store results and artifacts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data_root = os.path.abspath(args.data_root)
    output_dir = os.path.abspath(args.output_dir)

    os.makedirs(output_dir, exist_ok=True)

    results = run_experiments(
        data_root=data_root,
        model_names=args.models,
        embedding_dim=args.embedding_dim,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.learning_rate,
        random_seed=args.random_seed,
        output_dir=output_dir,
    )

    # Print concise summary
    print("\n=== Aggregated Results (filtered metrics on test set) ===")
    for model_name, metrics in results.items():
        mr = metrics.get("mr")
        mrr = metrics.get("mrr")
        hits_at_1 = metrics.get("hits_at_1")
        hits_at_3 = metrics.get("hits_at_3")
        hits_at_10 = metrics.get("hits_at_10")
        print(
            f"{model_name:8s} | MRR={mrr:.4f} | MR={mr:.1f} | H@1={hits_at_1:.4f} | H@3={hits_at_3:.4f} | H@10={hits_at_10:.4f}"
        )


if __name__ == "__main__":
    main()


