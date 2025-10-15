import argparse
import os
import sys
from pathlib import Path

# Ensure project root is on sys.path when running as a script
CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.openke_loader import convert_openke_to_tsv


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert OpenKE datasets to PyKEEN TSV format.")
    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="Path to dataset directory containing OpenKE files.",
    )
    args = parser.parse_args()

    dataset_dir = os.path.abspath(args.dataset_dir)
    train, valid, test = convert_openke_to_tsv(dataset_dir)
    print("Converted:")
    print(train)
    print(valid)
    print(test)


if __name__ == "__main__":
    main()


