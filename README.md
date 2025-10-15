## PyKEEN KRL Models Experiment

This repository provides a clean experimental setup to train and evaluate multiple Knowledge Representation Learning (KRL) models using PyKEEN on curated TSV triples derived from the datasets under `Data/FB15K237` and `Data/WN18RR`.

### Goal
- Train and evaluate the following models with the same training configuration:
  - TransE
  - TransR
  - ANALOGY (Analogy in PyKEEN)
  - HolE
  - RotatE
- Aggregate and compare standard link prediction metrics on the test set (MRR, MR, Hits@1/3/10).

### Repository Structure
```
.
├── main.py                       # Experiments entrypoint
├── src/
│   ├── data/
│   │   └── openke_loader.py      # One-time conversion utility to build TSV triples
│   ├── models/
│   │   └── registry.py           # Model registry + builders
│   └── utils/
│       └── experiment.py         # Training + evaluation driver
├── Data/
│   └── FB15K237/                 # Provided dataset (OpenKE format)
├── requirements.txt
└── README.md
```

### Dataset Format
This repository uses PyKEEN-compatible TSV files with columns `head\trelation\ttail`. They are stored under each dataset folder in `pykeen/`:
- `Data/FB15K237/pykeen/{train.tsv,valid.tsv,test.tsv}`
- `Data/WN18RR/pykeen/{train.tsv,valid.tsv,test.tsv}`

Note: The legacy OpenKE files have been converted and removed to make this repository independent of OpenKE.

### Installation
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you have a CUDA-capable GPU, install the matching `torch` build per the official install guide before installing `pykeen`.

### Usage
Run all models with default hyperparameters on FB15K237:
```bash
python main.py
```

Key options:
```bash
python main.py \
  --data_root "Data/FB15K237" \
  --models TransE TransR Analogy HolE RotatE \
  --embedding_dim 200 \
  --batch_size 1024 \
  --epochs 100 \
  --learning_rate 0.001 \
  --random_seed 42 \
  --output_dir outputs
```

Results are saved under `outputs/runs_<timestamp>/<MODEL_NAME>/`, with `metrics.json` for quick lookup and the full PyKEEN run artifacts. A summary table is printed to stdout and an aggregated JSON is saved to `outputs/runs_<timestamp>/aggregated_results.json`.

Additionally, an aggregated CSV and a PDF report with metric definitions are saved under the same `runs_<timestamp>/` directory:
- `aggregated_results.csv`
- `report.pdf` (includes a summary table and definitions for MRR, MR, Hits@k)

Training logs:
- Each model writes logs to `outputs/runs_<timestamp>/<MODEL_NAME>/train.log` and also streams to the console.

### Notes
- Model name `Analogy` corresponds to ANALOGY.
- All models share the same training parameters for a fair comparison. Adjust `embedding_dim`, `epochs`, etc., as needed.
- To reconvert datasets from legacy OpenKE files (if you re-add them), use: `python scripts/convert_openke_to_pykeen.py --dataset_dir Data/<DATASET>`


