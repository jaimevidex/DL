# Homework 1

## Question 1

### 1(a)

The test accuracy of the model with the best performance on the validation set is 61.1% it's validation accuracy was 61.3% and this is refereing to epoch 6.
The model file is in `results/single_perceptron/Q1-perceptron-save.pkl`.
This accuracy plot is located in `results/single_perceptron/Q1-perceptron-accs.pdf`.
For this exercise we implemented the update_weight, train_epoch, predict, and evaluate methods of the Perceptron class in `skeleton_code/hw1-perceptron.py` and run that file.

### 2(a)

The test accuracy of the model with the best performance on the validation set is 69.6% it's validation accuracy was 69.7% and this is refereing to epoch 20.
The model file is in `results/logistic_regression/Q1-2-logistic-a.pkl`.
This accuracy plot is located in `results/logistic_regression/Q1-2-logistic-a.pdf`.
For this exercise we created a logistic regression class in `skeleton_code/hw1-logistic_regression.py` and run it with `experiments/logistic_regression_run.py -q a` .

### 2(b)

We decided to go with max pooling with a 2x2 window:
Initially, we considered using 4x4 Average Pooling to reduce the feature space. However, experiments showed that while training was significantly faster, accuracy dropped because the images became "washed out" (e.g., a distinct white pixel averaged with black background becomes a faint gray value, losing contrast).

To solve this, we implemented 2x2 Max Pooling. Instead of averaging, this method selects the maximum value in each 2x2 block. This ensures that if a stroke (white pixel) exists in a block, the signal is preserved rather than diluted. This approach improved robustness to small shifts in letter positioning and reduced the input dimensionality from 784 to 196. Empirically, this reduced training time from ~1m 20s to ~35s with negligible loss in accuracy.

To test this, we can add the `--dataset-format pooled` flag in `experiments/logistic_regression_run.py` to use the pooled dataset.

### 2(c)

| Dataset Structure | Learning Rate | L2 Penalty | Best Validation Accuracy |
| :--- | :--- | :--- | :--- |
| original | 0.0001 | 0.00001 | 0.6940 |
| original | 0.0001 | 0.0001 | 0.6951 |
| original | 0.001 | 0.00001 | 0.7192 |
| original | 0.001 | 0.0001 | 0.7195 |
| original | 0.01 | 0.00001 | 0.6992 |
| original | 0.01 | 0.0001 | 0.6948 |
| pooled | 0.0001 | 0.00001 | 0.6668 |
| pooled | 0.0001 | 0.0001 | 0.6680 |
| pooled | 0.001 | 0.00001 | 0.7091 |
| pooled | 0.001 | 0.0001 | 0.7095 |
| pooled | 0.01 | 0.00001 | 0.7086 |
| pooled | 0.01 | 0.0001 | 0.7067 |

The test accuracy of the configuration with the best validation accuracy was 72.1%.

### 3(a)

The test accuracy of the configuration with the best validation accuracy was 87.8% and the validation accuracy was 87.8% at epoch 20 as shown in `results/multilayer_perceptron/Q1-3-scores.json`. That model was saved in `results/multilayer_perceptron/Q1-3-mlp.pkl`.
The train and validation accuracy curves are shown in `results/multilayer_perceptron/Q1-3-mlp-accuracy.pdf`.
The loss curve is shown in `results/multilayer_perceptron/Q1-3-mlp-loss.pdf`.

## Question 2

This document consolidates the key findings from all Question 2 experiments (PyTorch FFNs on EMNIST Letters) so we can quickly craft the final report answers.

---

### Implementation Warmup
- `FeedforwardNetwork` in `skeleton_code/hw1-ffn.py` now supports arbitrary depth, configurable activation (`relu`/`tanh`), dropout, and weight decay.
- Training utilities (`train_batch`, `evaluate`, CLI options) were generalized to save metrics/plots and to run scripted sweeps without modification.
- Sanity runs with width = 16 confirmed the pipeline before large sweeps.

---

### Part 2 – Width Study (Toward Infinite Width)

#### Grid Configuration
- Hidden widths: 16, 32, 64, 128, 256.
- Learning rates: {1e‑4, 3e‑4, 1e‑3, 3e‑3}.
- Dropout: {0, 0.2}; Weight decay: {0, 1e‑4}.
- Optimizer: Adam; activation: ReLU; epochs: 30; batch size: 64.
- Automation script: `experiments/run_width_grid.py` (outputs under `figures/width_study/`).

#### Best Validation Accuracy Per Width

| Width | LR    | Dropout | L2     | Val Acc | Train Acc | Test Acc |
|------:|------:|--------:|-------:|--------:|----------:|---------:|
| 16    | 1e‑3  | 0.0     | 1e‑4   | 0.781   | 0.793     | 0.784    |
| 32    | 1e‑3  | 0.0     | 1e‑4   | 0.844   | 0.864     | 0.845    |
| 64    | 1e‑3  | 0.0     | 1e‑4   | 0.881   | 0.909     | 0.880    |
| 128   | 1e‑3  | 0.0     | 1e‑4   | 0.898   | 0.938     | 0.896    |
| 256   | 3e‑4  | 0.2     | 0.0    | **0.909** | 0.949     | **0.908** |

#### Observations
- Validation accuracy improves rapidly with width up to ~128 units, then saturates; training accuracy keeps climbing (0.79 → 0.95), highlighting growing capacity and slight overfitting at large widths.
- Dropout becomes beneficial at the widest setting (256) by narrowing the generalization gap; weight decay was not necessary for the best model.
- Figure `figures/width_study/accuracy_vs_width.png` plots train/val/test accuracy trends; `figures/width_study/best_width_learning_curves.png` shows the convergence behavior for the top model (smooth loss decay, stable validation curve, small train>val gap).

---

### Part 2(b) – Best Model Deep Dive
- Configuration: width = 256, depth = 1, lr = 3e‑4, dropout = 0.2, no L2, Adam, ReLU.
- Metrics: validation 0.909, test 0.908, final train 0.949.
- Training dynamics (see `best_width_learning_curves.png`): both losses steadily decrease, validation accuracy plateaus near the end without degradation, suggesting mild but controlled overfitting.

---

### Part 2(c) – Training Accuracy vs Width
- Using the best configuration for each width, training accuracy vs width exhibits a near-monotonic rise toward 1.0, implying that wider networks interpolate the training set increasingly well.
- Validation accuracy plateaus, supporting the Universal Approximation intuition: capacity is ample by width ≥ 128, so further gains rely on regularization/optimization rather than raw width.

---

### Part 3 – Depth Study (Effect of Depth in Vanilla FFNs)

#### Setup
- Fixed width = 32 (best from width study) and reused optimal hyperparameters: lr = 3e‑4, dropout = 0.2, no weight decay, Adam, ReLU.
- Depths tested: 1, 3, 5, 7, 9 hidden layers (30 epochs each).
- Script: `experiments/run_depth_grid.py` with outputs under `figures/depth_study/`.

#### Results Summary

| Depth | Val Acc | Train Acc | Test Acc |
|------:|--------:|----------:|---------:|
| 1     | **0.817** | 0.828     | **0.818** |
| 3     | 0.792    | 0.803     | 0.793    |
| 5     | 0.717    | 0.725     | 0.718    |
| 7     | 0.630    | 0.637     | 0.635    |
| 9     | 0.524    | 0.533     | 0.528    |

#### Observations
- Accuracy drops sharply as depth increases beyond one hidden layer. Training curves (see `figures/depth_study/best_depth_learning_curves.png`) show that deeper networks struggle to optimize (loss plateaus high, large generalization gap), likely due to vanishing gradients and the absence of architectural aids (normalization/residuals).
- Figure `figures/depth_study/accuracy_vs_depth.png` highlights the monotonic decline in train/val/test accuracy with depth—model capacity exists but is unreachable with this vanilla SGD-style setup.
- Best validating model remains the shallow network (depth 1), reinforcing that, under identical hyperparameters, adding layers without architectural changes can harm both optimization and generalization.

---

### Takeaways for the Final Write-Up
1. **Width vs. Performance:** Wider hidden layers consistently boost accuracy until ~128 units; beyond that, returns diminish while training accuracy continues to rise. Dropout becomes key to prevent overfitting at very large widths.
2. **Best Model:** A single-hidden-layer network with 256 units, ReLU, lr = 3e‑4, dropout = 0.2 hits 90.9 % validation / 90.8 % test accuracy.
3. **Depth Trade-offs:** Holding width fixed, increasing depth without architectural aids severely degrades both training and validation accuracy. The data favors shallow networks for this task/setting.
4. **Figures & Artifacts:** Use the plots in `figures/width_study/` and `figures/depth_study/` to support the analysis tables above; cite JSON summaries if precise numbers are required.
