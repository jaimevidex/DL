"""
Evaluation script for RNA Binding Protein Affinity Prediction.

This script evaluates trained models on the test set and generates comparison plots.
"""

import os
import json
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np

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
    configure_seed
)

# Import models (in same directory as this script)
sys.path.insert(0, script_dir)
from hw2_q2_models import create_model


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device
) -> Dict[str, float]:
    """
    Evaluate model on test set.
    
    Args:
        model: Trained model
        test_loader: Test data loader
        device: Device to run on
    
    Returns:
        Dictionary with metrics
    """
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_targets = []
    all_masks = []
    num_batches = 0
    
    with torch.no_grad():
        for batch in test_loader:
            x, y, mask = batch
            x = x.to(device)
            y = y.to(device)
            mask = mask.to(device)
            
            # Forward pass
            predictions = model(x)
            
            # Compute loss
            loss = masked_mse_loss(predictions, y, mask)
            total_loss += loss.item()
            
            # Collect predictions
            all_predictions.append(predictions)
            all_targets.append(y)
            all_masks.append(mask)
            num_batches += 1
    
    # Concatenate all batches
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    all_masks = torch.cat(all_masks, dim=0)
    
    # Compute metrics
    test_loss = total_loss / num_batches if num_batches > 0 else 0.0
    spearman_corr = masked_spearman_correlation(all_predictions, all_targets, all_masks)
    
    # Convert to numpy for additional metrics
    pred_np = all_predictions.squeeze().cpu().numpy()
    target_np = all_targets.squeeze().cpu().numpy()
    mask_np = all_masks.squeeze().cpu().numpy().astype(bool)
    
    # Pearson correlation (on valid data only)
    valid_preds = pred_np[mask_np]
    valid_targets = target_np[mask_np]
    
    if len(valid_preds) > 1:
        pearson_corr = np.corrcoef(valid_preds, valid_targets)[0, 1]
        mae = np.mean(np.abs(valid_preds - valid_targets))
        rmse = np.sqrt(np.mean((valid_preds - valid_targets) ** 2))
    else:
        pearson_corr = 0.0
        mae = 0.0
        rmse = 0.0
    
    return {
        'test_loss': float(test_loss),
        'spearman_correlation': float(spearman_corr.item()),
        'pearson_correlation': float(pearson_corr),
        'mae': float(mae),
        'rmse': float(rmse),
        'num_samples': int(mask_np.sum())
    }


def load_model_checkpoint(
    checkpoint_path: str,
    model_type: str,
    device: torch.device,
    **model_kwargs
) -> nn.Module:
    """
    Load model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model_type: 'cnn' or 'lstm'
        device: Device to load model on
        **model_kwargs: Model hyperparameters
    
    Returns:
        Loaded model
    """
    model = create_model(model_type, **model_kwargs)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    return model


def plot_predictions_vs_targets(
    predictions: np.ndarray,
    targets: np.ndarray,
    masks: np.ndarray,
    model_name: str,
    save_path: str
):
    """
    Plot predictions vs targets scatter plot.
    
    Args:
        predictions: Model predictions
        targets: True targets
        masks: Validity masks
        model_name: Name of model for title
        save_path: Path to save plot
    """
    valid_preds = predictions[masks]
    valid_targets = targets[masks]
    
    plt.figure(figsize=(8, 8))
    plt.scatter(valid_targets, valid_preds, alpha=0.5, s=10)
    plt.plot([valid_targets.min(), valid_targets.max()], 
             [valid_targets.min(), valid_targets.max()], 
             'r--', lw=2, label='Perfect prediction')
    plt.xlabel('True Binding Affinity')
    plt.ylabel('Predicted Binding Affinity')
    plt.title(f'{model_name}: Predictions vs Targets')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def compare_models(
    results: Dict[str, Dict],
    save_dir: str
):
    """
    Generate comparison plots and summary for multiple models.
    
    Args:
        results: Dictionary mapping model names to evaluation results
        save_dir: Directory to save comparison plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Extract metrics
    model_names = list(results.keys())
    spearman_corrs = [results[name]['spearman_correlation'] for name in model_names]
    pearson_corrs = [results[name]['pearson_correlation'] for name in model_names]
    test_losses = [results[name]['test_loss'] for name in model_names]
    
    # Comparison bar plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].bar(model_names, spearman_corrs, color=['#1f77b4', '#ff7f0e'])
    axes[0].set_ylabel('Spearman Correlation')
    axes[0].set_title('Spearman Correlation Comparison')
    axes[0].set_ylim([0, 1])
    axes[0].grid(True, alpha=0.3, axis='y')
    
    axes[1].bar(model_names, pearson_corrs, color=['#1f77b4', '#ff7f0e'])
    axes[1].set_ylabel('Pearson Correlation')
    axes[1].set_title('Pearson Correlation Comparison')
    axes[1].set_ylim([0, 1])
    axes[1].grid(True, alpha=0.3, axis='y')
    
    axes[2].bar(model_names, test_losses, color=['#1f77b4', '#ff7f0e'])
    axes[2].set_ylabel('Test Loss (MSE)')
    axes[2].set_title('Test Loss Comparison')
    axes[2].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'model_comparison.pdf'), bbox_inches='tight')
    plt.close()
    
    # Save summary
    summary = {
        'model_comparison': {
            name: {
                'spearman_correlation': results[name]['spearman_correlation'],
                'pearson_correlation': results[name]['pearson_correlation'],
                'test_loss': results[name]['test_loss'],
                'mae': results[name]['mae'],
                'rmse': results[name]['rmse']
            }
            for name in model_names
        }
    }
    
    summary_path = os.path.join(save_dir, 'evaluation_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\nModel Comparison Summary:")
    print("=" * 80)
    for name in model_names:
        print(f"\n{name}:")
        print(f"  Spearman Correlation: {results[name]['spearman_correlation']:.4f}")
        print(f"  Pearson Correlation:  {results[name]['pearson_correlation']:.4f}")
        print(f"  Test Loss (MSE):     {results[name]['test_loss']:.4f}")
        print(f"  MAE:                 {results[name]['mae']:.4f}")
        print(f"  RMSE:                {results[name]['rmse']:.4f}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate RNA Binding Affinity Models')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--model_type', type=str, required=True, choices=['cnn', 'lstm'],
                        help='Model type')
    parser.add_argument('--protein', type=str, default='RBFOX1', help='Protein name')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--device', type=str, default='auto', help='Device')
    
    # Model hyperparameters (needed to recreate model architecture)
    # CNN
    parser.add_argument('--num_filters', type=int, default=64)
    parser.add_argument('--filter_sizes', type=int, nargs='+', default=[3, 5, 7])
    parser.add_argument('--dropout_rate', type=float, default=0.2)
    parser.add_argument('--hidden_units', type=int, default=128)
    parser.add_argument('--num_dense_layers', type=int, default=2)
    
    # LSTM
    parser.add_argument('--lstm_hidden_size', type=int, default=128)
    parser.add_argument('--num_lstm_layers', type=int, default=2)
    parser.add_argument('--dense_units', type=int, default=128)
    parser.add_argument('--use_embedding', action='store_true')
    parser.add_argument('--embedding_dim', type=int, default=32)
    parser.add_argument('--aggregation', type=str, default='last', choices=['last', 'attention', 'mean'])
    
    # Other
    parser.add_argument('--save_dir', type=str, default='results', help='Save directory')
    parser.add_argument('--compare', action='store_true',
                        help='If True, compare multiple models (requires --checkpoints and --model_types)')
    parser.add_argument('--checkpoints', type=str, nargs='+', help='Multiple checkpoints for comparison')
    parser.add_argument('--model_types', type=str, nargs='+', help='Model types for comparison')
    
    args = parser.parse_args()
    
    # Device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Load test data
    print("Loading test data...")
    test_dataset = load_rnacompete_data(args.protein, split='test')
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    print(f"Test samples: {len(test_dataset)}")
    
    if args.compare and args.checkpoints:
        # Compare multiple models
        results = {}
        for checkpoint_path, model_type in zip(args.checkpoints, args.model_types):
            model_name = f"{model_type.upper()}"
            print(f"\nEvaluating {model_name}...")
            
            # Get model kwargs
            if model_type == 'cnn':
                model_kwargs = {
                    'num_filters': args.num_filters,
                    'filter_sizes': args.filter_sizes,
                    'dropout_rate': args.dropout_rate,
                    'hidden_units': args.hidden_units,
                    'num_dense_layers': args.num_dense_layers
                }
            else:
                model_kwargs = {
                    'lstm_hidden_size': args.lstm_hidden_size,
                    'num_lstm_layers': args.num_lstm_layers,
                    'dropout_rate': args.dropout_rate,
                    'dense_units': args.dense_units,
                    'use_embedding': args.use_embedding,
                    'embedding_dim': args.embedding_dim,
                    'aggregation': args.aggregation
                }
            
            # Load model
            model = load_model_checkpoint(checkpoint_path, model_type, device, **model_kwargs)
            
            # Evaluate
            metrics = evaluate_model(model, test_loader, device)
            results[model_name] = metrics
            
            print(f"  Spearman Correlation: {metrics['spearman_correlation']:.4f}")
            print(f"  Test Loss: {metrics['test_loss']:.4f}")
        
        # Generate comparison
        compare_models(results, args.save_dir)
    
    else:
        # Evaluate single model
        print(f"\nEvaluating {args.model_type.upper()} model...")
        
        # Get model kwargs
        if args.model_type == 'cnn':
            model_kwargs = {
                'num_filters': args.num_filters,
                'filter_sizes': args.filter_sizes,
                'dropout_rate': args.dropout_rate,
                'hidden_units': args.hidden_units,
                'num_dense_layers': args.num_dense_layers
            }
        else:
            model_kwargs = {
                'lstm_hidden_size': args.lstm_hidden_size,
                'num_lstm_layers': args.num_lstm_layers,
                'dropout_rate': args.dropout_rate,
                'dense_units': args.dense_units,
                'use_embedding': args.use_embedding,
                'embedding_dim': args.embedding_dim,
                'aggregation': args.aggregation
            }
        
        # Load model
        model = load_model_checkpoint(args.checkpoint, args.model_type, device, **model_kwargs)
        
        # Evaluate
        metrics = evaluate_model(model, test_loader, device)
        
        # Save results
        model_dir = os.path.join(args.save_dir, args.model_type)
        os.makedirs(model_dir, exist_ok=True)
        
        results_path = os.path.join(model_dir, 'test_evaluation.json')
        with open(results_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print("\nTest Set Evaluation Results:")
        print("=" * 80)
        print(f"Spearman Correlation: {metrics['spearman_correlation']:.4f}")
        print(f"Pearson Correlation:  {metrics['pearson_correlation']:.4f}")
        print(f"Test Loss (MSE):      {metrics['test_loss']:.4f}")
        print(f"MAE:                   {metrics['mae']:.4f}")
        print(f"RMSE:                  {metrics['rmse']:.4f}")
        print(f"Valid Samples:         {metrics['num_samples']}")
        print(f"\nResults saved to: {results_path}")


if __name__ == '__main__':
    main()

