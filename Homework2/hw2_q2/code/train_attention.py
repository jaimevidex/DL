import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional
import sys

# Import components from skeleton code
script_dir = os.path.dirname(os.path.abspath(__file__))
skeleton_path = os.path.join(script_dir, 'hw2_q2_skeleton_code')
sys.path.append(skeleton_path)
from utils import masked_mse_loss, masked_spearman_correlation, plot

def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> float:
    """Train for one epoch."""
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
    """Validate model."""
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

class EarlyStopping:
    """Early stopping to prevent overfitting."""
    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = float('-inf')
        self.early_stop = False
    
    def __call__(self, score: float) -> bool:
        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop

def train_attention_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    device: torch.device,
    save_dir: str,
    patience: int = 10
) -> Dict:
    """Train the attention model."""
    os.makedirs(save_dir, exist_ok=True)
    early_stopping = EarlyStopping(patience=patience)
    
    train_losses = []
    val_losses = []
    val_correlations = []
    best_correlation = float('-inf')
    
    print(f"Training on device: {device}")
    
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss, val_corr = validate(model, val_loader, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_correlations.append(val_corr)
        
        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Train Loss: {train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, "
              f"Val Corr: {val_corr:.4f}")
        
        # Save best model
        if val_corr > best_correlation:
            best_correlation = val_corr
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pt"))
            
        if early_stopping(val_corr):
            print(f"Early stopping at epoch {epoch+1}")
            break
            
    # Save history
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_correlations': val_correlations,
        'best_val_correlation': best_correlation
    }
    
    with open(os.path.join(save_dir, "history.json"), 'w') as f:
        json.dump(history, f, indent=2)
        
    # Plot curves
    plot(range(1, len(train_losses)+1), 
         {'Train Loss': train_losses, 'Val Loss': val_losses}, 
         filename=os.path.join(save_dir, "loss_curves.pdf"))
         
    return history

