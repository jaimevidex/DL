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
    parser = argparse.ArgumentParser(description="Run width grid for Question 2.")
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
        default=Path(__file__).resolve().parents[1] / "figures" / "width_study" / "metrics",
        help="Directory to store per-run metrics JSON files.",
    )
    parser.add_argument(
        "--summary_path",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "figures" / "width_study" / "width_summary.json",
        help="Path to summary JSON output.",
    )
    parser.add_argument("--widths", type=int, nargs="+", default=[16, 32, 64, 128, 256])
    parser.add_argument("--learning_rates", type=float, nargs="+", default=[1e-4, 3e-4, 1e-3, 3e-3])
    parser.add_argument("--dropout", type=float, default=0.2,
                        help="Non-zero dropout probability to test.")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help="Non-zero weight decay value to test.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--activation", choices=["relu", "tanh"], default="relu")
    parser.add_argument("--optimizer", choices=["adam", "sgd"], default="adam")
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

    widths = args.widths
    learning_rates = args.learning_rates
    dropouts = [0.0, args.dropout]
    weight_decays = [0.0, args.weight_decay]

    summary = {"widths": []}
    run_counter = 0

    for width in widths:
        width_results = []
        best_result = None
        for lr in learning_rates:
            for dropout in dropouts:
                for weight_decay in weight_decays:
                    run_counter += 1
                    metrics_path = metrics_dir / f"width{width}_lr{lr}_drop{dropout}_wd{weight_decay}.json"
                    opt = Namespace(
                        epochs=args.epochs,
                        batch_size=args.batch_size,
                        hidden_size=width,
                        layers=1,
                        learning_rate=lr,
                        l2_decay=weight_decay,
                        dropout=dropout,
                        activation=args.activation,
                        optimizer=args.optimizer,
                        data_path=str(data_path),
                        model=f"width-{width}",
                        no_plots=True,
                        metrics_path=str(metrics_path),
                        seed=args.seed + run_counter,
                    )
                    metrics = module.run_training(opt)
                    result = {
                        "width": width,
                        "learning_rate": lr,
                        "dropout": dropout,
                        "l2_decay": weight_decay,
                        "best_val_acc": metrics["best_val_acc"],
                        "final_train_acc": metrics["final_train_acc"],
                        "test_acc": metrics["test_acc"],
                        "metrics_path": str(metrics_path),
                    }
                    width_results.append(result)
                    print(
                        f"[{run_counter:03d}] width={width} lr={lr} "
                        f"drop={dropout} wd={weight_decay} "
                        f"val={result['best_val_acc']:.4f} "
                        f"train={result['final_train_acc']:.4f} "
                        f"test={result['test_acc']:.4f}"
                    )
                    if best_result is None or result["best_val_acc"] > best_result["best_val_acc"]:
                        best_result = result

        summary["widths"].append({
            "width": width,
            "best": best_result,
            "runs": width_results,
        })

    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved width summary to {summary_path}")


if __name__ == "__main__":
    main()

