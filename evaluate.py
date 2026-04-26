"""
Evaluation script for the dual-branch molecular property prediction model.

Metrics reported:
  - RMSE / MAE  for regression datasets  (ESOL, FreeSolv, Lipophilicity)
  - AUROC       for classification / multi-label datasets

Usage
-----
# Single checkpoint (dataset / split inferred from config stored inside .pt):
    python evaluate.py --checkpoint checkpoints/best_model_BACE_seed42.pt

# Evaluate on a specific dataset:
    python evaluate.py --checkpoint checkpoints/best_model_ESOL_seed42.pt --dataset ESOL

# Scan a whole directory — prints summary table:
    python evaluate.py --checkpoint_dir checkpoints/

# Override data / embedding paths:
    python evaluate.py --checkpoint_dir checkpoints/ --data_dir dataset --embeddings_dir embeddings

# Save results as JSON:
    python evaluate.py --checkpoint_dir checkpoints/ --output results.json
"""

import argparse
import json
import os
import sys
from glob import glob

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.dataset import DATASET_CONFIG, collate_fn, get_dataset, scaffold_split, random_split
from models.cross_attention import DualBranchModel



# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _rmse(labels: np.ndarray, preds: np.ndarray) -> float:
    return float(np.sqrt(np.mean((labels - preds) ** 2)))


def _auroc_results(labels: np.ndarray, scores: np.ndarray, task_type: str,
                   label_cols: list) -> dict:
    from sklearn.metrics import roc_auc_score

    if task_type == "classification":
        try:
            return {"AUROC": roc_auc_score(labels, scores[:, 1])}
        except Exception:
            return {"AUROC": float("nan")}

    # multilabel
    per_task = []
    for t in range(labels.shape[1]):
        yt, ys = labels[:, t], scores[:, t]
        if len(np.unique(yt)) < 2:
            continue
        try:
            per_task.append(roc_auc_score(yt, ys))
        except Exception:
            pass
    return {
        "AUROC_mean": float(np.mean(per_task)) if per_task else float("nan"),
        "AUROC_per_task": per_task,
    }


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------

def load_model(checkpoint_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    config = ckpt["config"]

    full_dataset = get_dataset(
        config["dataset_name"],
        data_dir=config.get("data_dir", "dataset"),
        embeddings_dir=config.get("embeddings_dir", "embeddings"),
        task_type=config.get("task_type"),
    )

    sample = full_dataset[0]["graph"]
    atom_dim = sample.x.size(1)
    bond_dim = (
        sample.edge_attr.size(1)
        if sample.edge_attr.numel() > 0
        else config.get("bond_dim", 10)
    )

    if full_dataset.task_type == "regression":
        num_classes = 1
    elif full_dataset.task_type == "multilabel":
        num_classes = len(full_dataset.label_cols)
    else:
        num_classes = len(set(full_dataset.valid_labels))

    model = DualBranchModel(
        atom_dim=atom_dim,
        bond_dim=bond_dim,
        graph_dim=config.get("graph_dim", 512),
        smiles_dim=config.get("smiles_dim", 3072),
        hidden_dim=config.get("hidden_dim", 512),
        num_classes=int(num_classes),
        num_layers=config.get("num_layers", 3),
        num_heads=config.get("num_heads", 8),
        pool_type=config.get("pool_type", "attention"),
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    normalizer = None
    return model, config, full_dataset, normalizer


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_inference(model, loader, task_type: str, device: torch.device,
                  normalizer=None):
    all_labels, all_scores = [], []

    for batch in tqdm(loader, desc="  Inference", leave=False):
        graphs = batch["graph"].to(device)
        labels = batch["label"]
        emb = batch.get("smiles_embedding")

        if emb is None:
            raise RuntimeError(
                "No precomputed SMILES embeddings found. "
                "Run embedding generation before evaluation."
            )
        emb = emb.to(device)

        logits, _, _ = model(graphs, emb)

        if task_type == "regression":
            if logits.dim() > 1:
                logits = logits.squeeze(1)
            scores = logits.cpu().numpy()
        elif task_type == "multilabel":
            scores = torch.sigmoid(logits).cpu().numpy()
        else:
            scores = torch.softmax(logits, dim=1).cpu().numpy()

        all_scores.append(scores)
        all_labels.append(labels.cpu().numpy())

    labels_arr = np.concatenate(all_labels)
    scores_arr = np.concatenate(all_scores)

    if task_type == "regression" and normalizer is not None:
        scores_arr = normalizer.inverse_transform(scores_arr)

    return labels_arr, scores_arr


# ---------------------------------------------------------------------------
# Per-checkpoint evaluation
# ---------------------------------------------------------------------------

def evaluate_checkpoint(
    checkpoint_path: str,
    dataset_name: str | None = None,
    data_dir: str = "dataset",
    embeddings_dir: str = "embeddings",
    device: torch.device | None = None,
    split: str = "test",
    batch_size: int = 32,
) -> dict:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*60}")
    print(f"Checkpoint : {checkpoint_path}")

    model, config, full_dataset, normalizer = load_model(checkpoint_path, device)

    if dataset_name is None:
        dataset_name = config["dataset_name"]

    task_type = full_dataset.task_type
    label_cols = full_dataset.label_cols
    seed = config.get("seed", 42)
    split_type = config.get("split_type", "random")

    print(f"Dataset    : {dataset_name}  |  task={task_type}  |  "
          f"split={split_type}/{split}  |  seed={seed}")

    if split_type == "scaffold":
        train_idx, val_idx, test_idx = scaffold_split(full_dataset, seed=seed)
    else:
        train_idx, val_idx, test_idx = random_split(full_dataset, seed=seed)

    eval_indices = {"train": train_idx, "val": val_idx, "test": test_idx}[split]
    loader = DataLoader(
        Subset(full_dataset, eval_indices),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    labels, scores = run_inference(model, loader, task_type, device, normalizer)

    metrics = {
        "dataset": dataset_name,
        "task_type": task_type,
        "split": split,
        "split_type": split_type,
        "checkpoint": checkpoint_path,
    }

    if task_type == "regression":
        metrics["RMSE"] = _rmse(labels, scores)
        metrics["MAE"] = float(np.mean(np.abs(labels - scores)))
        print(f"  RMSE : {metrics['RMSE']:.4f}")
        print(f"  MAE  : {metrics['MAE']:.4f}")
    elif task_type == "multilabel":
        r = _auroc_results(labels, scores, task_type, label_cols)
        metrics.update(r)
        print(f"  AUROC (mean) : {r['AUROC_mean']:.4f}")
        for col, v in zip(label_cols, r.get("AUROC_per_task", [])):
            print(f"    {col:<20s}: {v:.4f}")
    else:
        r = _auroc_results(labels, scores, task_type, label_cols)
        metrics.update(r)
        print(f"  AUROC : {r['AUROC']:.4f}")

    return metrics


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary_table(results: list):
    print(f"\n{'='*65}")
    print("SUMMARY")
    print(f"{'='*65}")
    print(f"{'Dataset':<20s} {'Setting':<10s} {'Task':<14s} {'Metric':<12s} {'Value':>8s}")
    print("-" * 65)
    for r in results:
        ds, setting, task = r["dataset"], r.get("split_type", "random"), r["task_type"]
        if task == "regression":
            print(f"{ds:<20s} {setting:<10s} {task:<14s} {'RMSE':<12s} {r['RMSE']:>8.4f}")
            print(f"{'':<20s} {'':<10s} {'':<14s} {'MAE':<12s} {r['MAE']:>8.4f}")
        elif task == "multilabel":
            v = r.get("AUROC_mean", float("nan"))
            print(f"{ds:<20s} {setting:<10s} {task:<14s} {'AUROC_mean':<12s} {v:>8.4f}")
        else:
            v = r.get("AUROC", float("nan"))
            print(f"{ds:<20s} {setting:<10s} {task:<14s} {'AUROC':<12s} {v:>8.4f}")
    print("=" * 65)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate dual-branch model checkpoints on MoleculeNet benchmarks"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--checkpoint", type=str, help="Path to a single .pt checkpoint")
    group.add_argument("--checkpoint_dir", type=str,
                       help="Directory to scan for all *.pt checkpoints")

    parser.add_argument("--dataset", type=str, default=None,
                        choices=list(DATASET_CONFIG.keys()),
                        help="Override dataset name from checkpoint")
    parser.add_argument("--data_dir", type=str, default="dataset")
    parser.add_argument("--embeddings_dir", type=str, default="embeddings")
    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "val", "test"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=str, default=None,
                        help="Save results as JSON to this path")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    print(f"Device: {device}")

    checkpoints = (
        [args.checkpoint]
        if args.checkpoint
        else sorted(glob(os.path.join(args.checkpoint_dir, "*.pt")))
    )
    if not checkpoints:
        print(f"No .pt files found.")
        sys.exit(1)
    print(f"Evaluating {len(checkpoints)} checkpoint(s)…")

    results = []
    for ckpt in checkpoints:
        try:
            r = evaluate_checkpoint(
                checkpoint_path=ckpt,
                dataset_name=args.dataset,
                data_dir=args.data_dir,
                embeddings_dir=args.embeddings_dir,
                device=device,
                split=args.split,
                batch_size=args.batch_size,
            )
            results.append(r)
        except Exception as exc:
            print(f"[ERROR] {ckpt}: {exc}")

    if len(results) > 1:
        print_summary_table(results)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, default=float)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
