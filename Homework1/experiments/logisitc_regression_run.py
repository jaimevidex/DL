#!/usr/bin/env python

from pathlib import Path
import argparse
import numpy as np
import os
import sys
import time
import json
import itertools

current_dir = os.path.dirname(os.path.abspath(__file__))

parent_dir = os.path.dirname(current_dir)

sys.path.append(parent_dir)

import skeleton_code.hw1_logistic_regression as lr
import skeleton_code.utils as utils

def apply_max_pooling(X):
    """
    Downsamples 28x28 images to 14x14 images by taking the MAX of 2x2 pixel blocks.
    Makes the input smaller so calculations are faster without losing too much accuracy.
    
    Args:
        X: Numpy array of shape (N, 784)
        
    Returns:
        X_pooled: Numpy array of shape (N, 196)
    """
    # 1. Reshape to image format (N, 28, 28)
    n_samples = X.shape[0]
    if X.ndim == 2:
        X_img = X.reshape(n_samples, 28, 28)
    else:
        X_img = X

    # 2. Reshape to (N, 14, 2, 14, 2)
    # We split 28 rows -> 14 chunks of 2
    # We split 28 cols -> 14 chunks of 2
    X_reshaped = X_img.reshape(n_samples, 14, 2, 14, 2)
    
    # 3. Take the MAX over the blocks (axis 2 and 4)
    # This preserves the strongest signal in each 2x2 block
    X_pooled = X_reshaped.max(axis=(2, 4))
    
    # 4. Flatten back to vectors (N, 14*14) -> (N, 196)
    return X_pooled.reshape(n_samples, -1)

def run(learning_rate, l2_penalty, data, logging: bool, question: str, save_path: str, accuracy_plot: str, scores: str, n_epochs=20):
    X_train, y_train = data["train"]
    X_valid, y_valid = data["valid"]
    X_test, y_test = data["test"]

    n_classes = np.unique(y_train).size
    n_feats = X_train.shape[1]

    model = lr.SoftmaxRegressionL2(learning_rate, l2_penalty, n_epochs, n_classes, n_feats)

    train_acc_history = []
    val_acc_history = []
    epochs = np.arange(1, n_epochs + 1)

    best_valid = 0.0
    best_epoch = -1

    start = time.time()

    if logging:
        print(f"Training with LR={learning_rate} and L2={l2_penalty}...")

    for i in epochs:
        if logging:
            print("-" * 30)
            print(f"Running epoch {i}/{n_epochs}")
        model.train_epoch(X_train, y_train)

        train_pred = model.predict(X_train)
        train_acc = np.mean(train_pred == y_train)
        train_acc_history.append(train_acc)
        val_pred = model.predict(X_valid)
        val_acc = np.mean(val_pred == y_valid)
        val_acc_history.append(val_acc)

        if logging:
            print('train acc: {:.4f} | val acc: {:.4f}'.format(train_acc, val_acc))

        if val_acc > best_valid:
            best_valid = val_acc
            best_epoch = i
            if question == "a":
                model.save(save_path)
                print(f"New best model saved at epoch {i} with val acc {val_acc:.4f}")

        if logging:
            print()

    elapsed_time = time.time() - start
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    print('Training took {} minutes and {} seconds'.format(minutes, seconds))
    if question == "a":
        print("Reloading best checkpoint")
        best_model: lr.SoftmaxRegressionL2 = lr.SoftmaxRegressionL2.load(save_path)
        test_pred = best_model.predict(X_test)
        test_acc = np.mean(test_pred == y_test)

        print('Best model test acc: {:.4f}'.format(test_acc))

        utils.plot(
            "Epoch", "Accuracy",
            {"train": (epochs, train_acc_history), "valid": (epochs, val_acc_history)},
            filename=accuracy_plot
        )

        with open(scores, "w") as f:
            json.dump(
                {"best_valid": float(best_valid),
                 "selected_epoch": int(best_epoch),
                 "test": float(test_acc),
                 "time": elapsed_time},
                f,
                indent=4
            )

    return best_valid

def main(args):
    question = args.question
    data_path = args.data_path
    save_path = args.save_path
    accuracy_plot = args.accuracy_plot
    scores = args.scores

    data = utils.load_dataset(data_path=data_path)

    X_train, y_train = data["train"]
    X_train_pooled = apply_max_pooling(X_train)
    X_valid, y_valid = data["dev"]
    X_valid_pooled = apply_max_pooling(X_valid)
    X_test, y_test = data["test"]
    X_test_pooled = apply_max_pooling(X_test)

    datasets = [
            {"name": "normal", "train": (X_train, y_train), "valid": (X_valid, y_valid), "test": (X_test, y_test)},
            {"name": "pooled", "train": (X_train_pooled, y_train), "valid": (X_valid_pooled, y_valid), "test": (X_test_pooled, y_test)},
    ]

    learning_rates = [0.0001, 0.001, 0.01]
    l2_penalties = [0.00001, 0.0001]

    best_valid_list = []

    dataset = {}
    match args.dataset_format:
        case "normal":
            dataset = datasets[0]
        case "pooled":
            dataset = datasets[1]

    match question:
        case "a":
            run(args.learning_rate, args.l2_penalty, dataset, True, question, save_path, accuracy_plot, scores)
        case "c":
            for dataset, learning_rate, l2_penalty in itertools.product(datasets, learning_rates, l2_penalties):
                print(f"Running {dataset['name']} with LR={learning_rate} and L2={l2_penalty}...")
                best_valid = run(learning_rate, l2_penalty, dataset, args.logging, question, save_path, accuracy_plot, scores)
                stats = {
                        "dataset": dataset["name"],
                        "learning_rate": learning_rate,
                        "l2_penalty": l2_penalty,
                        "best_valid": best_valid,
                        }
                best_valid_list.append(stats)
                if args.logging:
                    print(f"dataset: {stats['dataset']}")
                    print(f"learning_rate: {stats['learning_rate']}")
                    print(f"l2_penalty: {stats['l2_penalty']}")
                    print(f"best_valid: {stats["best_valid"]:.4f}")
                    print()

            true_best_valid = {
                    "best_valid": 0.0,
                    }
            for stats in best_valid_list:
                if stats["best_valid"] > true_best_valid["best_valid"]:
                    true_best_valid = stats
                print(f"{stats['dataset']} LR={stats['learning_rate']} L2={stats['l2_penalty']} best_valid={stats['best_valid']:.4f}")
                print()

            print(f"The configuration with the best valid accuracy is: ")
            print(f"Dataset Structure: {true_best_valid['dataset']}")
            print(f"Learning Rate: {true_best_valid['learning_rate']}")
            print(f"L2 Penalty: {true_best_valid['l2_penalty']}")
            print(f"Best Validation Accuracy: {true_best_valid['best_valid']:.4f}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run logistic regression for Question 2.")
    parser.add_argument("-q", "--question", type=str, choices=["a", "c"], default="a", required=True)
    parser.add_argument("--save-path", type=Path, default=Path(__file__).resolve().parents[1] / "results" / "logistic_regression" / "a" / "Q1-2-logistic-a.pkl")
    parser.add_argument("--accuracy-plot", type=Path, default=Path(__file__).resolve().parents[1] / "results" / "logistic_regression" / "a" / "Q1-2-logistic-a.pdf")
    parser.add_argument("--scores", type=Path, default=Path(__file__).resolve().parents[1] / "results" / "logistic_regression" / "a" / "Q1-2-scores.json")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=0.0001)
    parser.add_argument("--l2-penalty", type=float, default=0.00001)
    parser.add_argument("--dataset-format", type=str, choices=["normal", "pooled"], default="normal")
    parser.add_argument("--data-path", type=Path, default=Path(__file__).resolve().parents[1] / "emnist-letters.npz")
    parser.add_argument("--logging", type=bool, default=False)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    main(args)
