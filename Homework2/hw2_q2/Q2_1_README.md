# Question 2.1: Quick Start Guide

## Overview

This implementation provides two deep learning models for RNA binding affinity prediction:
- **CNN**: Convolutional Neural Network for motif detection
- **LSTM**: Bidirectional LSTM for sequence modeling

## Quick Start

### 1. Train a Model

**CNN Example:**
```bash
python hw2_q2_train.py \
    --model_type cnn \
    --protein RBFOX1 \
    --batch_size 64 \
    --learning_rate 0.001 \
    --num_filters 64 \
    --filter_sizes 3 5 7 \
    --dropout_rate 0.2 \
    --hidden_units 128 \
    --num_dense_layers 2
```

**LSTM Example:**
```bash
python hw2_q2_train.py \
    --model_type lstm \
    --protein RBFOX1 \
    --batch_size 64 \
    --learning_rate 0.001 \
    --lstm_hidden_size 128 \
    --num_lstm_layers 2 \
    --dropout_rate 0.2 \
    --dense_units 128 \
    --aggregation last
```

### 2. Hyperparameter Tuning

```bash
# Random search (recommended - faster)
python hw2_q2_tune.py \
    --model_type both \
    --protein RBFOX1 \
    --search_type random \
    --num_random 20

# Grid search (comprehensive but slow)
python hw2_q2_tune.py \
    --model_type both \
    --protein RBFOX1 \
    --search_type grid
```

### 3. Evaluate on Test Set

```bash
# Single model
python hw2_q2_evaluate.py \
    --checkpoint results/cnn/cnn_best.pt \
    --model_type cnn \
    --protein RBFOX1 \
    --num_filters 64 \
    --filter_sizes 3 5 7 \
    --dropout_rate 0.2 \
    --hidden_units 128 \
    --num_dense_layers 2

# Compare both models
python hw2_q2_evaluate.py \
    --compare \
    --checkpoints results/cnn/cnn_best.pt results/lstm/lstm_best.pt \
    --model_types cnn lstm \
    --protein RBFOX1 \
    --num_filters 64 --filter_sizes 3 5 7 --dropout_rate 0.2 --hidden_units 128 --num_dense_layers 2 \
    --lstm_hidden_size 128 --num_lstm_layers 2 --dense_units 128 --aggregation last
```

## File Structure

- `hw2_q2_models.py`: Model architectures (CNN, LSTM)
- `hw2_q2_train.py`: Training script with early stopping
- `hw2_q2_tune.py`: Hyperparameter tuning script
- `hw2_q2_evaluate.py`: Evaluation script for test set
- `Q2_1_REPORT.md`: Comprehensive theoretical report
- `results/`: Directory for saved models, plots, and metrics

## Key Features

- ✅ Masked MSE loss (handles NaN values)
- ✅ Spearman correlation evaluation
- ✅ Early stopping to prevent overfitting
- ✅ Automatic checkpointing
- ✅ Training curve visualization
- ✅ Hyperparameter tuning framework

## Important Notes

1. **Always use validation set for hyperparameter tuning** (not test set)
2. **Test set should only be used once** for final evaluation
3. **Must use masked_mse_loss** (provided in utils.py)
4. **Protein must be RBFOX1** for this assignment

## Dependencies

- PyTorch
- NumPy
- Matplotlib
- Pandas
- openpyxl (for reading metadata.xlsx)

See `Q2_1_REPORT.md` for detailed theoretical explanations and architecture descriptions.

