import csv
import os
from pathlib import Path
from typing import Dict, Tuple


def _read_id_mapping(path: str) -> Dict[int, str]:
    mapping: Dict[int, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        first_line = f.readline().strip()
        # Some OpenKE files have a count on the first line
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            name = " ".join(parts[:-1])
            idx = int(parts[-1])
            mapping[idx] = name
    return mapping


def _convert_split(
    split_path: str,
    entity_id_to_label: Dict[int, str],
    relation_id_to_label: Dict[int, str],
    out_path: str,
) -> None:
    with open(split_path, "r", encoding="utf-8") as fin, open(
        out_path, "w", encoding="utf-8", newline=""
    ) as fout:
        writer = csv.writer(fout, delimiter="\t")
        first_line = fin.readline()
        # If first token is an integer, treat as count header and skip
        tokens = first_line.strip().split()
        if len(tokens) >= 1:
            try:
                int(tokens[0])
                # header line, skip
            except ValueError:
                # not a header, process as data
                if len(tokens) >= 3:
                    h_id, t_id, r_id = int(tokens[0]), int(tokens[1]), int(tokens[2])
                    writer.writerow([
                        entity_id_to_label[h_id],
                        relation_id_to_label[r_id],
                        entity_id_to_label[t_id],
                    ])
        # Process remaining lines
        for line in fin:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            h_id, t_id, r_id = int(parts[0]), int(parts[1]), int(parts[2])
            writer.writerow([
                entity_id_to_label[h_id],
                relation_id_to_label[r_id],
                entity_id_to_label[t_id],
            ])


def convert_openke_to_tsv(dataset_root: str) -> Tuple[str, str, str]:
    """Convert OpenKE-formatted dataset to PyKEEN-compatible TSV triples.

    Returns paths to train.tsv, valid.tsv, test.tsv.
    """
    root = Path(dataset_root)
    entity2id = root / "entity2id.txt"
    relation2id = root / "relation2id.txt"
    train2id = root / "train2id.txt"
    valid2id = root / "valid2id.txt"
    test2id = root / "test2id.txt"

    if not all(p.exists() for p in [entity2id, relation2id, train2id, valid2id, test2id]):
        raise FileNotFoundError(
            "Dataset directory must contain entity2id.txt, relation2id.txt, train2id.txt, valid2id.txt, test2id.txt"
        )

    entity_id_to_label = _read_id_mapping(str(entity2id))
    relation_id_to_label = _read_id_mapping(str(relation2id))

    out_dir = root / "pykeen"
    os.makedirs(out_dir, exist_ok=True)

    train_tsv = out_dir / "train.tsv"
    valid_tsv = out_dir / "valid.tsv"
    test_tsv = out_dir / "test.tsv"

    _convert_split(str(train2id), entity_id_to_label, relation_id_to_label, str(train_tsv))
    _convert_split(str(valid2id), entity_id_to_label, relation_id_to_label, str(valid_tsv))
    _convert_split(str(test2id), entity_id_to_label, relation_id_to_label, str(test_tsv))

    return str(train_tsv), str(valid_tsv), str(test_tsv)


