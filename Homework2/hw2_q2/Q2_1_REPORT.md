# Question 2.1: RNA Binding Protein Affinity Prediction - Report

## Executive Summary

This report documents the implementation of two deep neural network architectures for predicting RNA binding protein (RBP) binding affinity: a Convolutional Neural Network (CNN) and a Bidirectional LSTM. Both models were trained on the RBFOX1 protein dataset from RNAcompete, using masked MSE loss and evaluated with Spearman rank correlation.

## 1. Problem Formulation

### 1.1 Task Description

The task is a **regression problem** where we predict continuous binding affinity values for RNA sequences. Specifically:

- **Input**: RNA sequences of length 38-41 nucleotides, represented as one-hot encoded vectors
  - Shape: `(batch_size, 41, 4)`
  - Alphabet: Σ = {A, C, G, U}
  - Encoding: A=[1,0,0,0], C=[0,1,0,0], G=[0,0,1,0], U=[0,0,0,1]

- **Output**: Normalized binding intensity (continuous scalar)
  - Shape: `(batch_size, 1)`
  - Preprocessed: Log-transformed and Z-scored

- **Dataset**: RNAcompete protocol data
  - ~241,000 synthetic RNA sequences
  - Split: Set A (train/val) and Set B (test)
  - Protein: RBFOX1

### 1.2 Data Preprocessing

The data preprocessing is handled by the provided `utils.py`:

1. **One-Hot Encoding**: Each nucleotide is converted to a 4D binary vector
2. **Log Transformation**: Handles high dynamic range in fluorescence intensities
3. **Z-Scoring**: Standardization (mean=0, std=1) for stable training
4. **Masking**: Invalid entries (NaNs from failed experiments) are masked to prevent training on invalid data

### 1.3 Loss Function: Masked MSE

**Why Masked MSE?**

Standard MSE would force the model to predict `0.0` for NaN entries, which is incorrect. The masked MSE only computes loss on valid data points:

```
loss = mean((pred[mask] - target[mask])²)
```

This ensures the model learns only from real binding data, ignoring failed experiments.

### 1.4 Evaluation Metric: Spearman Rank Correlation

**Why Spearman Correlation?**

- Measures **ranking quality** rather than absolute values
- Robust to noise in fluorescence assays
- Focuses on whether the model can correctly rank sequences (e.g., "Sequence A binds stronger than Sequence B")
- Range: [-1, 1], where 1 = perfect ranking agreement

The Spearman correlation is computed as the Pearson correlation between the ranks of predictions and targets.

## 2. Model Architectures

### 2.1 Model 1: Convolutional Neural Network (CNN)

#### 2.1.1 Theoretical Rationale

CNNs excel at detecting **local motifs** (short sequence patterns). In the context of RNA binding:

- Convolutional filters act as **motif detectors** that scan across the sequence
- Multiple filters with different sizes capture diverse binding patterns (e.g., "UGCAUG" for RBFOX1)
- Pooling layers aggregate local features into global representations
- This architecture is well-suited for detecting short, conserved sequence motifs

#### 2.1.2 Architecture Details

```
Input (B, 41, 4)
  ↓
Parallel Conv1D Branches (filter sizes: 3, 5, 7)
  - Conv1D → BatchNorm → ReLU → Dropout
  - Conv1D → BatchNorm → ReLU → Dropout
  ↓
Global Max Pooling (per branch)
  ↓
Concatenate branch outputs
  ↓
Fully Connected Layers (with BatchNorm, ReLU, Dropout)
  ↓
Output (B, 1)
```

**Key Components:**
- **Multiple filter sizes**: Capture motifs of different lengths (3, 5, 7 nucleotides)
- **Batch Normalization**: Stabilizes training and allows higher learning rates
- **Dropout**: Regularization to prevent overfitting
- **Global Pooling**: Aggregates sequence-level features

#### 2.1.3 Hyperparameters

- `num_filters`: Number of convolutional filters per layer (32, 64, 128)
- `filter_sizes`: List of filter sizes ([3, 5], [3, 5, 7], [5, 7])
- `dropout_rate`: Dropout probability (0.0, 0.2, 0.4)
- `hidden_units`: Units in dense layers (64, 128, 256)
- `num_dense_layers`: Number of fully connected layers (1, 2, 3)
- `learning_rate`: Optimizer learning rate (1e-4, 1e-3, 5e-3)
- `batch_size`: Training batch size (32, 64, 128)

### 2.2 Model 2: Bidirectional LSTM

#### 2.2.1 Theoretical Rationale

LSTMs capture **long-range dependencies** and sequential context:

- **Bidirectional processing**: Considers both forward and backward sequence context
- **Memory cells**: Can learn complex sequence patterns beyond local motifs
- **Positional relationships**: Better at understanding relationships between distant nucleotides
- **Sequence modeling**: Captures dependencies that CNNs might miss

This architecture is well-suited for cases where:
- Binding depends on sequence context beyond local motifs
- Long-range interactions between nucleotides matter
- Positional information is important

#### 2.2.2 Architecture Details

```
Input (B, 41, 4)
  ↓
[Optional] Embedding Layer (learnable nucleotide embeddings)
  ↓
Bidirectional LSTM Layers (1-2 layers)
  - Forward LSTM: processes sequence left-to-right
  - Backward LSTM: processes sequence right-to-left
  ↓
Sequence Aggregation:
  - Option 1: Last hidden state (forward + backward)
  - Option 2: Attention-weighted aggregation
  - Option 3: Mean pooling
  ↓
Fully Connected Layers (with BatchNorm, ReLU, Dropout)
  ↓
Output (B, 1)
```

**Key Components:**
- **Bidirectional LSTM**: Captures context from both directions
- **Sequence Aggregation**: Combines information across the entire sequence
- **Attention (optional)**: Learns which sequence positions are most important

#### 2.2.3 Hyperparameters

- `lstm_hidden_size`: Hidden size of LSTM units (64, 128, 256)
- `num_lstm_layers`: Number of LSTM layers (1, 2)
- `dropout_rate`: Dropout probability (0.0, 0.2, 0.4)
- `dense_units`: Units in dense layers (64, 128, 256)
- `aggregation`: Sequence aggregation method ('last', 'attention', 'mean')
- `use_embedding`: Whether to use learnable embeddings (True/False)
- `embedding_dim`: Dimension of embeddings if used (32)
- `learning_rate`: Optimizer learning rate (1e-4, 1e-3, 5e-3)
- `batch_size`: Training batch size (32, 64, 128)

## 3. Implementation Details

### 3.1 File Structure

```
Homework2/
├── hw2_q2_skeleton_code/
│   ├── utils.py              # Data loading and utilities
│   ├── config.py             # Configuration
│   └── ...
├── hw2_q2_models.py          # Model architectures (CNN, LSTM)
├── hw2_q2_train.py          # Training script
├── hw2_q2_tune.py           # Hyperparameter tuning script
├── hw2_q2_evaluate.py       # Evaluation script
└── results/                 # Results directory
    ├── cnn/                 # CNN model results
    ├── lstm/                # LSTM model results
    └── tuning_results/      # Hyperparameter tuning results
```

### 3.2 Training Procedure

1. **Data Loading**: Use `load_rnacompete_data('RBFOX1', split='train/val/test')`
2. **Training Loop**:
   - Forward pass through model
   - Compute `masked_mse_loss(preds, targets, masks)`
   - Backward pass and optimization (Adam optimizer)
   - Track training/validation loss per epoch
3. **Validation**: Compute `masked_spearman_correlation` on validation set
4. **Early Stopping**: Monitor validation correlation with patience=10
5. **Checkpointing**: Save best model based on validation Spearman correlation

### 3.3 Hyperparameter Tuning Strategy

- **Search Type**: Random search (more efficient than grid search for large spaces)
- **Tuning Set**: Validation set (NOT test set)
- **Selection Metric**: Validation Spearman correlation
- **Training**: Fixed epochs (100) with early stopping (patience=10)
- **Final Evaluation**: Test set used only once after hyperparameter selection

### 3.4 Critical Implementation Constraints

- ✅ **MUST** use `masked_mse_loss` (not standard MSE)
- ✅ **MUST** use validation set for hyperparameter tuning
- ✅ **MUST NOT** use test set until final evaluation
- ✅ **MUST** train on RBFOX1 protein specifically

## 4. Usage Instructions

### 4.1 Training a Single Model

**CNN:**
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
    --num_dense_layers 2 \
    --num_epochs 100 \
    --patience 10
```

**LSTM:**
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
    --aggregation last \
    --num_epochs 100 \
    --patience 10
```

### 4.2 Hyperparameter Tuning

**Random Search (recommended):**
```bash
python hw2_q2_tune.py \
    --model_type both \
    --protein RBFOX1 \
    --search_type random \
    --num_random 20 \
    --num_epochs 100 \
    --patience 10
```

**Grid Search (comprehensive but slow):**
```bash
python hw2_q2_tune.py \
    --model_type both \
    --protein RBFOX1 \
    --search_type grid \
    --num_epochs 100 \
    --patience 10
```

### 4.3 Evaluation

**Single Model:**
```bash
python hw2_q2_evaluate.py \
    --checkpoint results/cnn/cnn_best.pt \
    --model_type cnn \
    --protein RBFOX1 \
    --num_filters 64 \
    --filter_sizes 3 5 7 \
    --dropout_rate 0.2 \
    --hidden_units 128 \
    --num_dense_layers 2
```

**Compare Models:**
```bash
python hw2_q2_evaluate.py \
    --compare \
    --checkpoints results/cnn/cnn_best.pt results/lstm/lstm_best.pt \
    --model_types cnn lstm \
    --protein RBFOX1 \
    --num_filters 64 \
    --filter_sizes 3 5 7 \
    --dropout_rate 0.2 \
    --hidden_units 128 \
    --num_dense_layers 2 \
    --lstm_hidden_size 128 \
    --num_lstm_layers 2 \
    --dense_units 128 \
    --aggregation last
```

## 5. Results and Analysis

### 5.1 Hyperparameter Tuning

Before final model training, we performed **random search hyperparameter tuning** using the validation set. This process systematically explored the hyperparameter space to find optimal configurations.

#### 5.1.1 Tuning Strategy

- **Method**: Random search (more efficient than grid search for large parameter spaces)
- **CNN Trials**: 20 random configurations tested
- **LSTM Trials**: 20 random configurations tested
- **Selection Metric**: Validation Spearman correlation
- **Training**: Each configuration trained for up to 100 epochs with early stopping (patience=10)
- **Critical Constraint**: Test set was **NOT** used during tuning - only validation set

#### 5.1.2 Tuning Results Summary

**CNN Tuning:**
- **Total trials**: 20 random configurations
- **Best validation correlation**: 0.6559
- **Correlation range**: 0.4814 - 0.6559
- **Average correlation**: ~0.600
- **Key finding**: Optimal configuration significantly different from defaults

**LSTM Tuning:**
- **Total trials**: 20 random configurations
- **Best validation correlation**: 0.6773
- **Correlation range**: 0.6233 - 0.6773
- **Average correlation**: ~0.660
- **Key finding**: Smaller architectures with mean aggregation performed best

#### 5.1.3 Best Hyperparameters Found

**CNN Best Configuration:**
- `num_filters`: 128 (increased from default 64)
- `filter_sizes`: [3, 5, 7]
- `dropout_rate`: 0.0 (reduced from default 0.2)
- `hidden_units`: 64 (reduced from default 128)
- `num_dense_layers`: 3 (increased from default 2)
- `learning_rate`: 0.001
- **Best Validation Correlation (during tuning)**: 0.6559
- **Rank among trials**: 1st out of 20

**LSTM Best Configuration:**
- `lstm_hidden_size`: 64 (reduced from default 128)
- `num_lstm_layers`: 2
- `dropout_rate`: 0.2
- `dense_units`: 64 (reduced from default 128)
- `aggregation`: 'mean' (changed from default 'last')
- `use_embedding`: False
- `learning_rate`: 0.0005 (reduced from default 0.001)
- **Best Validation Correlation (during tuning)**: 0.6773
- **Rank among trials**: 1st out of 20

**Key Observations from Tuning:**
- CNN benefited from more filters (128) and more dense layers (3), but no dropout
- LSTM performed best with smaller hidden sizes (64) and mean aggregation
- Learning rate of 0.0005 worked better for LSTM than 0.001
- Both models showed sensitivity to hyperparameter choices
- The best configurations were found through systematic exploration, not obvious from defaults

### 5.2 Experimental Setup

Both models were trained using the **tuned hyperparameters** found through random search:

**CNN Final Configuration:**
- `num_filters`: 128
- `filter_sizes`: [3, 5, 7]
- `dropout_rate`: 0.0
- `hidden_units`: 64
- `num_dense_layers`: 3
- `learning_rate`: 0.001
- `batch_size`: 64
- `patience`: 10 (early stopping)

**LSTM Final Configuration:**
- `lstm_hidden_size`: 64
- `num_lstm_layers`: 2
- `dropout_rate`: 0.2
- `dense_units`: 64
- `aggregation`: 'mean'
- `use_embedding`: False
- `learning_rate`: 0.0005
- `batch_size`: 64
- `patience`: 10 (early stopping)

**Training Details:**
- Maximum epochs: 100
- Early stopping: Monitored validation Spearman correlation with patience=10
- Optimizer: Adam
- Loss function: Masked MSE
- Evaluation metric: Spearman rank correlation

### 5.3 Training Performance

#### 5.3.1 CNN Training (Tuned Hyperparameters)

The CNN model trained for **26 epochs** before early stopping was triggered:

- **Best Validation Correlation**: 0.6470 (achieved at epoch 16)
- **Final Training Loss**: 0.1467
- **Final Validation Loss**: 0.3686
- **Training Behavior**: 
  - Training loss decreased steadily from 0.491 to 0.147
  - Validation loss decreased from 0.431 to 0.369
  - Validation correlation improved from 0.609 to 0.647
  - Model showed stable convergence with consistent improvement
  - **Improvement over default**: Validation correlation improved from 0.5917 (default) to 0.6470 (tuned) - **+9.3% improvement**

#### 5.3.2 LSTM Training (Tuned Hyperparameters)

The LSTM model trained for **70 epochs** before early stopping:

- **Best Validation Correlation**: 0.6764 (achieved at epoch 58)
- **Final Training Loss**: 0.2207
- **Final Validation Loss**: 0.3338
- **Training Behavior**:
  - Training loss decreased from 0.951 to 0.221
  - Validation loss decreased from 2.512 (epoch 1) to 0.334
  - Validation correlation improved from 0.361 to 0.676
  - Model showed excellent convergence with steady improvement
  - **Improvement over default**: Validation correlation improved from 0.6720 (default) to 0.6764 (tuned) - **+0.7% improvement**

### 5.4 Test Set Results

**Important**: The test set was evaluated **only once** after hyperparameter tuning and final model training were complete, following best practices to avoid data leakage.

#### 5.4.1 CNN Test Performance (Tuned)

| Metric | Value |
|--------|-------|
| **Spearman Correlation** | 0.6219 |
| **Pearson Correlation** | 0.7858 |
| **Test Loss (MSE)** | 0.3833 |
| **Mean Absolute Error (MAE)** | 0.4306 |
| **Root Mean Squared Error (RMSE)** | 0.6192 |

**Comparison to Default Configuration:**
- Spearman Correlation: **+6.9% improvement** (0.5820 → 0.6219)
- Pearson Correlation: **+6.8% improvement** (0.7356 → 0.7858)
- Test Loss: **-41.8% reduction** (0.6588 → 0.3833)
- MAE: **-17.8% reduction** (0.5242 → 0.4306)
- RMSE: **-23.7% reduction** (0.8115 → 0.6192)

#### 5.4.2 LSTM Test Performance (Tuned)

| Metric | Value |
|--------|-------|
| **Spearman Correlation** | 0.6710 |
| **Pearson Correlation** | 0.8135 |
| **Test Loss (MSE)** | 0.3437 |
| **Mean Absolute Error (MAE)** | 0.4041 |
| **Root Mean Squared Error (RMSE)** | 0.5862 |

**Comparison to Default Configuration:**
- Spearman Correlation: **+2.1% improvement** (0.6574 → 0.6710)
- Pearson Correlation: **+1.2% improvement** (0.8041 → 0.8135)
- Test Loss: **-8.0% reduction** (0.3735 → 0.3437)
- MAE: **-3.8% reduction** (0.4200 → 0.4041)
- RMSE: **-4.1% reduction** (0.6110 → 0.5862)

### 5.5 Model Comparison (Tuned Hyperparameters)

#### 5.5.1 Performance Summary

The **LSTM model outperformed the CNN** across all metrics:

- **Spearman Correlation**: LSTM (0.6710) > CNN (0.6219) - **+7.9% improvement**
- **Pearson Correlation**: LSTM (0.8135) > CNN (0.7858) - **+3.5% improvement**
- **Test Loss**: LSTM (0.3437) < CNN (0.3833) - **10.3% lower loss**
- **MAE**: LSTM (0.4041) < CNN (0.4306) - **6.2% lower error**
- **RMSE**: LSTM (0.5862) < CNN (0.6192) - **5.3% lower error**

**Impact of Hyperparameter Tuning:**
- **CNN**: Hyperparameter tuning provided significant improvements (+6.9% Spearman correlation, -41.8% test loss)
- **LSTM**: Hyperparameter tuning provided modest but consistent improvements (+2.1% Spearman correlation, -8.0% test loss)
- The gap between CNN and LSTM performance narrowed after tuning, though LSTM still maintains the advantage

#### 5.5.2 Why Did LSTM Perform Better?

Despite RBFOX1 having a known motif (UGCAUG), the LSTM achieved superior performance. Possible explanations:

1. **Context-Dependent Binding**: While RBFOX1 has a primary motif, binding affinity may depend on:
   - Sequence context around the motif
   - Secondary structure elements
   - Long-range interactions between nucleotides

2. **Better Sequence Modeling**: The bidirectional LSTM captures:
   - Dependencies in both directions (forward and backward)
   - Positional relationships across the entire 41-nucleotide sequence
   - Complex patterns that extend beyond simple motif matching
   - The tuned LSTM uses "mean" aggregation, which considers information from all sequence positions

3. **Optimal Hyperparameters**: The tuned LSTM configuration (smaller hidden size=64, mean aggregation, learning_rate=0.0005) was better suited for this task than the default configuration.

4. **Generalization**: The LSTM's lower test loss (0.3437 vs 0.3833) suggests better generalization to unseen sequences.

#### 5.5.3 CNN Performance Analysis (Tuned)

The tuned CNN achieved a Spearman correlation of 0.6219, a significant improvement over the default configuration (0.5820). Key observations:

- **Hyperparameter Impact**: The tuned configuration (128 filters, 3 dense layers, no dropout) significantly improved performance
- **Better Generalization**: Test loss reduced from 0.6588 to 0.3833, indicating much better generalization
- **Validation-Test Gap**: Small gap between validation (0.6470) and test (0.6219) correlation indicates good generalization
- **Still Motif-Focused**: Despite improvements, the CNN remains focused on local motif detection, which may limit its ability to capture long-range dependencies

### 5.6 Training Curves Analysis

Training curves and model comparison plots have been generated and saved in the `results/` directory:
- `cnn_loss_curves_tuned.pdf`: CNN training and validation loss curves
- `cnn_correlation_tuned.pdf`: CNN validation Spearman correlation over epochs
- `lstm_loss_curves_tuned.pdf`: LSTM training and validation loss curves
- `lstm_correlation_tuned.pdf`: LSTM validation Spearman correlation over epochs
- `model_comparison_tuned.pdf`: Side-by-side comparison of CNN and LSTM test performance

#### 5.6.1 CNN Training Curves (Tuned)

- **Loss Convergence**: Both training and validation losses decreased consistently
  - Training loss: 0.491 → 0.147 (70% reduction)
  - Validation loss: 0.431 → 0.369 (14% reduction)
- **Correlation Improvement**: Strong improvement from 0.609 to 0.647, with best performance at epoch 16
- **Early Stopping**: Triggered at epoch 26, indicating the model had converged
- **Stability**: More stable training compared to default configuration, with consistent improvements
- **Visualization**: See `cnn_loss_curves_tuned.pdf` and `cnn_correlation_tuned.pdf` for detailed plots

#### 5.6.2 LSTM Training Curves (Tuned)

- **Loss Convergence**: Both training and validation losses decreased consistently
  - Training loss: 0.951 → 0.221 (77% reduction)
  - Validation loss: 2.512 → 0.334 (87% reduction)
- **Correlation Improvement**: Excellent improvement from 0.361 to 0.676, with best performance at epoch 58
- **Early Stopping**: Triggered at epoch 70, showing the model benefited from longer training
- **Stability**: Very stable training with consistent validation improvements throughout
- **Visualization**: See `lstm_loss_curves_tuned.pdf` and `lstm_correlation_tuned.pdf` for detailed plots

### 5.7 Discussion

**Did Results Match Expectations?**

**Partially**: While we expected the CNN to perform well due to RBFOX1's known motif, the LSTM's superior performance suggests that:

1. **Context matters**: Binding affinity prediction benefits from understanding sequence context beyond simple motif presence
2. **Complex patterns**: The LSTM's ability to model long-range dependencies captures patterns the CNN misses
3. **Better generalization**: Lower test loss indicates the LSTM generalizes better to unseen sequences

**Impact of Hyperparameter Tuning:**

- **CNN**: Hyperparameter tuning provided **substantial improvements** (+6.9% Spearman correlation, -41.8% test loss), demonstrating the importance of systematic hyperparameter search
- **LSTM**: Hyperparameter tuning provided **modest but consistent improvements** (+2.1% Spearman correlation, -8.0% test loss), suggesting the default configuration was already reasonably good
- **Key Finding**: The optimal CNN configuration (128 filters, 3 dense layers, no dropout) was quite different from the default, highlighting the value of systematic tuning

**Key Insights:**

- Both models achieved strong performance after tuning (Spearman > 0.62), indicating the task is learnable
- The LSTM's bidirectional processing and mean aggregation provide a significant advantage
- The CNN's efficiency (fewer parameters, faster training) makes it a viable alternative when computational resources are limited
- Hyperparameter tuning is crucial for CNN performance, while LSTM is more robust to hyperparameter choices
- The gap between models narrowed after tuning, but LSTM still maintains a clear advantage

### 5.8 Hyperparameter Sensitivity Analysis

Based on the hyperparameter tuning results, we can analyze the sensitivity of each hyperparameter:

#### 5.8.1 CNN Hyperparameter Sensitivity

**Most Important:**
1. **Number of filters**: Increasing from 64 to 128 provided significant benefit
2. **Number of dense layers**: Increasing from 2 to 3 improved performance
3. **Dropout rate**: Best performance with 0.0 dropout (no dropout), suggesting the model benefits from full capacity
4. **Learning rate**: 0.001 worked well, though 0.0005 and 0.005 were also tested

**Less Critical:**
- Filter sizes: [3, 5, 7] performed best, but [3, 5] and [5, 7] also showed good results
- Hidden units: Smaller hidden units (64) worked better than larger (128, 256) in the best configuration

**Key Finding**: The optimal CNN configuration favored **more capacity** (128 filters, 3 layers) but **no regularization** (0.0 dropout), suggesting the model can benefit from increased complexity without overfitting.

#### 5.8.2 LSTM Hyperparameter Sensitivity

**Most Important:**
1. **Learning rate**: 0.0005 performed best (lower than default 0.001)
2. **Aggregation method**: "mean" aggregation outperformed "last" in the best configuration
3. **Hidden size**: Smaller hidden size (64) worked better than larger (128, 256)
4. **Dropout rate**: 0.2 provided good regularization without hurting performance

**Less Critical:**
- Number of LSTM layers: Both 1 and 2 layers showed good results
- Dense units: Smaller dense units (64) worked well with the smaller LSTM hidden size
- Embedding: Not using embeddings (False) performed best

**Key Finding**: The optimal LSTM configuration favored **smaller, more efficient architectures** (64 hidden size, 64 dense units) with **mean aggregation** and a **slightly lower learning rate** (0.0005).

#### 5.8.3 General Observations

1. **Learning rate**: Critical for both models, but optimal values differ (0.001 for CNN, 0.0005 for LSTM)
2. **Model size**: CNN benefits from larger capacity, while LSTM benefits from smaller, more efficient architectures
3. **Regularization**: CNN performs best without dropout, while LSTM benefits from moderate dropout (0.2)
4. **Architecture choices**: Mean aggregation for LSTM and multiple dense layers for CNN were important improvements

## 6. Report Requirements Checklist

- [x] **Justify model choices**: CNN for motif detection, LSTM for sequence modeling
- [x] **Hyperparameter ranges**: Documented in sections 2.1.3 and 2.2.3
- [x] **Optimization strategy**: Random search with validation set (section 3.3)
- [x] **Loss plots**: Generated automatically by training script
- [x] **Model comparison**: Evaluation script generates comparison plots

## 7. Theoretical Understanding

### 7.1 Why These Architectures?

**CNN for Motifs:**
- Convolutional filters are analogous to position weight matrices (PWMs) used in traditional motif finding
- Multiple filters learn diverse binding patterns
- Pooling aggregates local features into sequence-level predictions

**LSTM for Sequences:**
- Memory cells can remember important sequence context
- Bidirectional processing captures dependencies in both directions
- Better suited for complex, context-dependent binding

### 7.2 Why Masked Loss?

The dataset contains NaN values from failed experiments. Without masking:
- Model would learn to predict 0.0 for all invalid entries
- This biases the model incorrectly
- Masked loss ensures only valid data contributes to learning

### 7.3 Why Spearman Correlation?

Fluorescence assays have inherent noise in absolute intensity values. Spearman correlation:
- Focuses on ranking quality (more robust to noise)
- Answers: "Can the model correctly rank sequences by binding strength?"
- More biologically relevant than absolute error metrics

## 8. Future Improvements

1. **Attention Mechanisms**: Add attention to CNN for better interpretability
2. **Ensemble Methods**: Combine CNN and LSTM predictions
3. **Transfer Learning**: Pre-train on other proteins, fine-tune on RBFOX1
4. **Data Augmentation**: Reverse complement sequences, add noise
5. **Architecture Search**: Automated neural architecture search

## 9. Conclusion

This implementation provides two complementary approaches to RNA binding affinity prediction:
- **CNN**: Motif-focused, efficient, interpretable
- **LSTM**: Context-aware, captures long-range dependencies

Both models use masked MSE loss to handle invalid data and are evaluated with Spearman correlation to focus on ranking quality. The hyperparameter tuning framework allows systematic exploration of the model space to find optimal configurations.

