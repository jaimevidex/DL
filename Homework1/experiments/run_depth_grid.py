#!/usr/bin/env python

import argparse
import importlib.util
import json
import sys
from argparse import Namespace
from pathlib import Path


def load_hw1_module(module_path: Path):
    module_dir = module_path.parent
    if str(module_dir) not in sys.path:
        sys.path.insert(0, str(module_dir))
    spec = importlib.util.spec_from_file_location("hw1_ffn", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args():
    parser = argparse.ArgumentParser(description="Run depth grid for Question 2.")
    parser.add_argument(
        "--hw1_ffn_path",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "skeleton_code" / "hw1-ffn.py",
        help="Path to hw1-ffn.py file.",
    )
    parser.add_argument(
        "--data_path",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "emnist-letters.npz",
        help="Path to EMNIST letters dataset.",
    )
    parser.add_argument(
        "--metrics_dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "figures" / "depth_study" / "metrics",
        help="Directory to store per-run metrics JSON files.",
    )
    parser.add_argument(
        "--summary_path",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "figures" / "depth_study" / "depth_summary.json",
        help="Path to summary JSON output.",
    )
    parser.add_argument("--depths", type=int, nargs="+", default=[1, 3, 5, 7, 9])
    parser.add_argument("--hidden_size", type=int, default=32,
                        help="Width to reuse from best configuration.")
    parser.add_argument("--learning_rate", type=float, default=0.0003)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--l2_decay", type=float, default=0.0)
    parser.add_argument("--optimizer", choices=["sgd", "adam"], default="adam")
    parser.add_argument("--activation", choices=["relu", "tanh"], default="relu")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    module = load_hw1_module(args.hw1_ffn_path)

    metrics_dir = args.metrics_dir.resolve()
    metrics_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.summary_path.resolve()
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    data_path = args.data_path.resolve()

    summary = {"depths": []}

    for idx, depth in enumerate(args.depths, start=1):
        metrics_path = metrics_dir / f"depth{depth}.json"
        opt = Namespace(
            epochs=args.epochs,
            batch_size=args.batch_size,
            hidden_size=args.hidden_size,
            layers=depth,
            learning_rate=args.learning_rate,
            l2_decay=args.l2_decay,
            dropout=args.dropout,
            activation=args.activation,
            optimizer=args.optimizer,
            data_path=str(data_path),
            model=f"depth-{depth}",
            no_plots=True,
            metrics_path=str(metrics_path),
            seed=args.seed + idx,
        )
        metrics = module.run_training(opt)
        summary["depths"].append({
            "depth": depth,
            "best_val_acc": metrics["best_val_acc"],
            "final_train_acc": metrics["final_train_acc"],
            "test_acc": metrics["test_acc"],
            "metrics_path": str(metrics_path),
        })
        print(
            f"[{idx:02d}] depth={depth} "
            f"val={metrics['best_val_acc']:.4f} "
            f"train={metrics['final_train_acc']:.4f} "
            f"test={metrics['test_acc']:.4f}"
        )

    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved depth summary to {summary_path}")


if __name__ == "__main__":
    main()

