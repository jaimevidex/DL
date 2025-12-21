"""
Training script for RNA Binding Protein Affinity Prediction.

This script handles:
- Model training with masked MSE loss
- Validation with Spearman correlation
- Early stopping
- Model checkpointing
- Hyperparameter support
"""

import os
import json
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional

# Import from skeleton code
import sys
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
skeleton_path = os.path.join(script_dir, 'hw2_q2_skeleton_code')
sys.path.append(skeleton_path)
from utils import (
    load_rnacompete_data,
    masked_mse_loss,
    masked_spearman_correlation,
    configure_seed,
    plot
)

# Import models (in same directory as this script)
sys.path.insert(0, script_dir)
from hw2_q2_models import create_model


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = 'max'):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'max' for metrics to maximize (e.g., correlation), 'min' for loss
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
        if mode == 'max':
            self.is_better = lambda current, best: current > best + min_delta
            self.best_score = float('-inf')
        else:
            self.is_better = lambda current, best: current < best - min_delta
            self.best_score = float('inf')
    
    def __call__(self, score: float) -> bool:
        """
        Check if training should stop.
        
        Args:
            score: Current validation score
        
        Returns:
            True if should stop, False otherwise
        """
        if self.is_better(score, self.best_score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> float:
    """
    Train for one epoch.
    
    Args:
        model: Model to train
        train_loader: Training data loader
        optimizer: Optimizer
        device: Device to run on
    
    Returns:
        Average training loss
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch in train_loader:
        x, y, mask = batch
        x = x.to(device)
        y = y.to(device)
        mask = mask.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        predictions = model(x)
        
        # Compute masked loss
        loss = masked_mse_loss(predictions, y, mask)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches if num_batches > 0 else 0.0


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device
) -> Tuple[float, float]:
    """
    Validate model.
    
    Args:
        model: Model to validate
        val_loader: Validation data loader
        device: Device to run on
    
    Returns:
        Tuple of (validation_loss, spearman_correlation)
    """
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_targets = []
    all_masks = []
    num_batches = 0
    
    with torch.no_grad():
        for batch in val_loader:
            x, y, mask = batch
            x = x.to(device)
            y = y.to(device)
            mask = mask.to(device)
            
            # Forward pass
            predictions = model(x)
            
            # Compute loss
            loss = masked_mse_loss(predictions, y, mask)
            total_loss += loss.item()
            
            # Collect predictions for correlation
            all_predictions.append(predictions)
            all_targets.append(y)
            all_masks.append(mask)
            num_batches += 1
    
    # Concatenate all batches
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    all_masks = torch.cat(all_masks, dim=0)
    
    # Compute Spearman correlation
    spearman_corr = masked_spearman_correlation(all_predictions, all_targets, all_masks)
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    
    return avg_loss, spearman_corr.item()


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    device: torch.device,
    save_dir: str,
    model_name: str,
    patience: int = 10,
    verbose: bool = True
) -> Dict:
    """
    Train model with early stopping.
    
    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        optimizer: Optimizer
        num_epochs: Maximum number of epochs
        device: Device to run on
        save_dir: Directory to save checkpoints
        model_name: Name for saving model
        patience: Early stopping patience
        verbose: Whether to print progress
    
    Returns:
        Dictionary with training history
    """
    os.makedirs(save_dir, exist_ok=True)
    
    early_stopping = EarlyStopping(patience=patience, mode='max')
    
    train_losses = []
    val_losses = []
    val_correlations = []
    best_correlation = float('-inf')
    
    for epoch in range(num_epochs):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device)
        train_losses.append(train_loss)
        
        # Validate
        val_loss, val_corr = validate(model, val_loader, device)
        val_losses.append(val_loss)
        val_correlations.append(val_corr)
        
        if verbose:
            print(f"Epoch {epoch+1}/{num_epochs} - "
                  f"Train Loss: {train_loss:.4f}, "
                  f"Val Loss: {val_loss:.4f}, "
                  f"Val Correlation: {val_corr:.4f}")
        
        # Save best model
        if val_corr > best_correlation:
            best_correlation = val_corr
            checkpoint_path = os.path.join(save_dir, f"{model_name}_best.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_correlation': val_corr,
                'val_loss': val_loss,
            }, checkpoint_path)
        
        # Early stopping
        if early_stopping(val_corr):
            if verbose:
                print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Save training history
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_correlations': val_correlations,
        'best_val_correlation': best_correlation,
        'num_epochs_trained': len(train_losses)
    }
    
    history_path = os.path.join(save_dir, f"{model_name}_history.json")
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    # Plot training curves
    epochs = list(range(1, len(train_losses) + 1))
    plot_path = os.path.join(save_dir, f"{model_name}_loss_curves.pdf")
    plot(epochs, {
        'Train Loss': train_losses,
        'Validation Loss': val_losses
    }, filename=plot_path)
    
    plot_path = os.path.join(save_dir, f"{model_name}_correlation.pdf")
    plot(epochs, {
        'Validation Spearman Correlation': val_correlations
    }, filename=plot_path)
    
    return history


def main():
    parser = argparse.ArgumentParser(description='Train RNA Binding Affinity Model')
    
    # Model parameters
    parser.add_argument('--model_type', type=str, default='cnn', choices=['cnn', 'lstm'],
                        help='Model architecture: cnn or lstm')
    parser.add_argument('--protein', type=str, default='RBFOX1',
                        help='Protein name (default: RBFOX1)')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # Model hyperparameters (CNN)
    parser.add_argument('--num_filters', type=int, default=64, help='Number of CNN filters')
    parser.add_argument('--filter_sizes', type=int, nargs='+', default=[3, 5, 7],
                        help='CNN filter sizes')
    parser.add_argument('--dropout_rate', type=float, default=0.2, help='Dropout rate')
    parser.add_argument('--hidden_units', type=int, default=128, help='Hidden units (CNN dense)')
    parser.add_argument('--num_dense_layers', type=int, default=2, help='Number of dense layers (CNN)')
    
    # Model hyperparameters (LSTM)
    parser.add_argument('--lstm_hidden_size', type=int, default=128, help='LSTM hidden size')
    parser.add_argument('--num_lstm_layers', type=int, default=2, help='Number of LSTM layers')
    parser.add_argument('--dense_units', type=int, default=128, help='Dense units (LSTM)')
    parser.add_argument('--use_embedding', action='store_true', help='Use embedding layer (LSTM)')
    parser.add_argument('--embedding_dim', type=int, default=32, help='Embedding dimension (LSTM)')
    parser.add_argument('--aggregation', type=str, default='last', choices=['last', 'attention', 'mean'],
                        help='Sequence aggregation method (LSTM)')
    
    # Other
    parser.add_argument('--save_dir', type=str, default='results', help='Directory to save results')
    parser.add_argument('--device', type=str, default='auto', help='Device (auto, cpu, cuda)')
    
    args = parser.parse_args()
    
    # Set seed
    configure_seed(args.seed)
    
    # Device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    print(f"Model type: {args.model_type}")
    print(f"Protein: {args.protein}")
    
    # Load data
    print("Loading data...")
    train_dataset = load_rnacompete_data(args.protein, split='train')
    val_dataset = load_rnacompete_data(args.protein, split='val')
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create model
    if args.model_type == 'cnn':
        model = create_model(
            'cnn',
            num_filters=args.num_filters,
            filter_sizes=args.filter_sizes,
            dropout_rate=args.dropout_rate,
            hidden_units=args.hidden_units,
            num_dense_layers=args.num_dense_layers
        )
    else:  # lstm
        model = create_model(
            'lstm',
            lstm_hidden_size=args.lstm_hidden_size,
            num_lstm_layers=args.num_lstm_layers,
            dropout_rate=args.dropout_rate,
            dense_units=args.dense_units,
            use_embedding=args.use_embedding,
            embedding_dim=args.embedding_dim,
            aggregation=args.aggregation
        )
    
    model = model.to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Create save directory
    model_dir = os.path.join(args.save_dir, args.model_type)
    os.makedirs(model_dir, exist_ok=True)
    
    # Generate model name from hyperparameters
    if args.model_type == 'cnn':
        model_name = f"cnn_f{args.num_filters}_fs{'-'.join(map(str, args.filter_sizes))}_d{args.dropout_rate}_h{args.hidden_units}_l{args.num_dense_layers}_lr{args.learning_rate}_bs{args.batch_size}"
    else:
        model_name = f"lstm_h{args.lstm_hidden_size}_l{args.num_lstm_layers}_d{args.dropout_rate}_u{args.dense_units}_agg{args.aggregation}_lr{args.learning_rate}_bs{args.batch_size}"
    
    # Train
    print("\nStarting training...")
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        num_epochs=args.num_epochs,
        device=device,
        save_dir=model_dir,
        model_name=model_name,
        patience=args.patience,
        verbose=True
    )
    
    print(f"\nTraining completed!")
    print(f"Best validation correlation: {history['best_val_correlation']:.4f}")
    print(f"Results saved to: {model_dir}")


if __name__ == '__main__':
    main()

