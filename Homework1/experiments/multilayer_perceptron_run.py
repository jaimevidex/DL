#!/usr/bin/env python

from pathlib import Path
import argparse
import numpy as np
import os
import sys
import time
import json

current_dir = os.path.dirname(os.path.abspath(__file__))

parent_dir = os.path.dirname(current_dir)

sys.path.append(parent_dir)

import skeleton_code.hw1_multilayer_perceptron as mlp
import skeleton_code.utils as utils

def main():
    data_path = Path(__file__).resolve().parents[1] / "emnist-letters.npz"
    save_path = Path(__file__).resolve().parents[1] / "results" / "multilayer_perceptron" / "Q1-2-mlp-a.pkl"
    accuracy_plot = Path(__file__).resolve().parents[1] / "results" / "multilayer_perceptron" / "Q1-2-mlp-a.pdf"
    scores = Path(__file__).resolve().parents[1] / "results" / "multilayer_perceptron" / "Q1-2-scores.json"
    data = utils.load_dataset(data_path=data_path)

    X_train, y_train = data["train"]
    X_valid, y_valid = data["dev"]
    X_test, y_test = data["test"]

    logging = True
    n_classes = np.unique(y_train).size
    input_size = X_train.shape[1]
    n_samples = X_train.shape[0]
    n_epochs = 20
    model = mlp.MLP(n_samples, input_size=input_size, hidden_size=100, output_size=n_classes, learning_rate=0.001, epochs=n_epochs)


    train_acc_history = []
    val_acc_history = []
    loss_history = []
    epochs = np.arange(1, n_epochs + 1)

    best_valid = 0.0
    best_epoch = -1

    start = time.time()
    if logging:
        print(f"Training with LR={model.lr}...")

    for i in epochs:
        if logging:
            print("-" * 30)
            print(f"Running epoch {i}/{n_epochs}")
        avg_loss = model.train_epoch(X_train, y_train)
        loss_history.append(avg_loss)

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
            model.save(save_path)
            print(f"New best model saved at epoch {i} with val acc {val_acc:.4f}")

        if logging:
            print()

    elapsed_time = time.time() - start
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    print('Training took {} minutes and {} seconds'.format(minutes, seconds))
    print("Reloading best checkpoint")
    best_model: mlp.MLP = mlp.MLP.load(save_path)
    test_pred = best_model.predict(X_test)
    test_acc = np.mean(test_pred == y_test)

    print('Best model test acc: {:.4f}'.format(test_acc))

    utils.plot(
        "Epoch", "Accuracy",
        {"train": (epochs, train_acc_history), "valid": (epochs, val_acc_history), "loss": (epochs, loss_history)},
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

if __name__ == '__main__':
    main()
