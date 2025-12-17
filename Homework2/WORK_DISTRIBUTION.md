# Homework 2 - Work Division Summary

## Quick Overview

| Person | Task | Points | Timeline |
|--------|------|--------|----------|
| **Person 1 (You)** | Q2a: Two Architecture Implementation | 40 | **FIRST** (before vacation) |
| **Person 2** | Q1: CNNs + Q2c: Discussion | 30 | Independent (can start anytime) |
| **Person 3** | Q2b: Attention Extension | 30 | After Person 1 finishes Q2a |

---

## Detailed Breakdown

### 👤 Person 1: Q2a - Two Architecture Implementation (40 points)
**What to do:**
- Implement TWO different deep neural network architectures (CNN, RNN, or Transformer)
- Train both on RBFOX1 protein data
- Hyperparameter tuning with documentation
- Generate loss plots (train/val) for both models
- Compare performance and write report section

**Why first:** Independent work, sets foundation for Q2b, flexible timeline before vacation

**Deliverables for Person 3:**
- Both model architectures (clean code with comments)
- Trained model weights/checkpoints
- Performance results and comparisons
- Hyperparameter tuning documentation

---

### 👤 Person 2: Q1 + Q2c (30 points)

**Q1: Image Classification with CNNs (15 points)**
- Implement CNN for BloodMNIST (32→64→128 channels, linear layers)
- Train 200 epochs (Adam, lr=0.001, batch=64)
- Compare with/without Softmax
- Add MaxPool2d layers and re-run experiments
- Generate 3 plots: training loss, validation accuracy, test accuracy
- Discuss efficiency/effectiveness impact

**Q2c: Multi-Protein Discussion (15 points)**
- Describe modifications for multi-protein setting (data, architecture, training)
- Discuss one benefit and one challenge

**Why independent:** No dependencies, can work in parallel with Person 1

---

### 👤 Person 3: Q2b - Attention Extension (30 points)

**What to do:**
- Choose one architecture from Person 1's Q2a work
- Implement attention mechanism (self-attention or attention-pooling)
- Run experiments with/without attention
- Generate plots comparing performance
- Write report section with expectations and results

**When to start:** After Person 1 completes Q2a (receives code/models)

---

## Timeline & Dependencies

```
Week 1-2: Person 1 → Complete Q2a (foundation work)
Week 1-3: Person 2 → Work independently on Q1 + Q2c
Week 2-3: Person 3 → Start Q2b after receiving Person 1's work
```

**Key Dependencies:**
- Person 1 → Person 3: Q2a code/models needed for Q2b
- Person 2: Completely independent, no dependencies

---

## Data Setup (Everyone)

- **Q1:** `pip install medmnist` or download `bloodmnist.npz`
- **Q2:** Download from Google Drive, use provided `utils.py`

---

## Point Distribution: 40-30-30

This matches the timeline structure and complexity:
- Person 1: Foundation work (most complex, but done first)
- Person 2: Independent work (medium complexity, flexible)
- Person 3: Extension work (moderate complexity, depends on Person 1)

