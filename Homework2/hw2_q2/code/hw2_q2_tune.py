"""
Hyperparameter tuning script for RNA Binding Protein Affinity Prediction.

This script performs grid search or random search over hyperparameter space
for both CNN and LSTM models, using the validation set for selection.
"""

import os
import json
import argparse
import itertools
import random
import subprocess
import sys
from typing import Dict, List, Any
from pathlib import Path


def generate_cnn_configs(grid_search: bool = True, num_random: int = 20) -> List[Dict]:
    """
    Generate hyperparameter configurations for CNN.
    
    Args:
        grid_search: If True, do grid search. If False, do random search.
        num_random: Number of random configurations if not grid search.
    
    Returns:
        List of hyperparameter dictionaries
    """
    # Define hyperparameter ranges
    learning_rates = [1e-4, 1e-3, 5e-3]
    batch_sizes = [32, 64, 128]
    num_filters = [32, 64, 128]
    filter_sizes_options = [[3, 5], [3, 5, 7], [5, 7]]
    dropout_rates = [0.0, 0.2, 0.4]
    hidden_units = [64, 128, 256]
    num_dense_layers = [1, 2, 3]
    
    if grid_search:
        # Full grid search (will be large, so we'll limit combinations)
        configs = []
        # Limit grid search to most important parameters
        for lr in learning_rates:
            for bs in batch_sizes:
                for nf in num_filters:
                    for fs in filter_sizes_options:
                        for dr in dropout_rates:
                            for hu in hidden_units:
                                for ndl in num_dense_layers:
                                    configs.append({
                                        'model_type': 'cnn',
                                        'learning_rate': lr,
                                        'batch_size': bs,
                                        'num_filters': nf,
                                        'filter_sizes': fs,
                                        'dropout_rate': dr,
                                        'hidden_units': hu,
                                        'num_dense_layers': ndl
                                    })
        return configs
    else:
        # Random search
        configs = []
        for _ in range(num_random):
            configs.append({
                'model_type': 'cnn',
                'learning_rate': random.choice(learning_rates),
                'batch_size': random.choice(batch_sizes),
                'num_filters': random.choice(num_filters),
                'filter_sizes': random.choice(filter_sizes_options),
                'dropout_rate': random.choice(dropout_rates),
                'hidden_units': random.choice(hidden_units),
                'num_dense_layers': random.choice(num_dense_layers)
            })
        return configs


def generate_lstm_configs(grid_search: bool = True, num_random: int = 20) -> List[Dict]:
    """
    Generate hyperparameter configurations for LSTM.
    
    Args:
        grid_search: If True, do grid search. If False, do random search.
        num_random: Number of random configurations if not grid search.
    
    Returns:
        List of hyperparameter dictionaries
    """
    # Define hyperparameter ranges
    learning_rates = [1e-4, 1e-3, 5e-3]
    batch_sizes = [32, 64, 128]
    lstm_hidden_sizes = [64, 128, 256]
    num_lstm_layers = [1, 2]
    dropout_rates = [0.0, 0.2, 0.4]
    dense_units = [64, 128, 256]
    aggregations = ['last', 'attention', 'mean']
    use_embeddings = [False, True]
    
    if grid_search:
        # Full grid search (will be large)
        configs = []
        # Limit grid search to most important parameters
        for lr in learning_rates:
            for bs in batch_sizes:
                for lhs in lstm_hidden_sizes:
                    for nll in num_lstm_layers:
                        for dr in dropout_rates:
                            for du in dense_units:
                                for agg in aggregations:
                                    for emb in use_embeddings:
                                        configs.append({
                                            'model_type': 'lstm',
                                            'learning_rate': lr,
                                            'batch_size': bs,
                                            'lstm_hidden_size': lhs,
                                            'num_lstm_layers': nll,
                                            'dropout_rate': dr,
                                            'dense_units': du,
                                            'aggregation': agg,
                                            'use_embedding': emb
                                        })
        return configs
    else:
        # Random search
        configs = []
        for _ in range(num_random):
            configs.append({
                'model_type': 'lstm',
                'learning_rate': random.choice(learning_rates),
                'batch_size': random.choice(batch_sizes),
                'lstm_hidden_size': random.choice(lstm_hidden_sizes),
                'num_lstm_layers': random.choice(num_lstm_layers),
                'dropout_rate': random.choice(dropout_rates),
                'dense_units': random.choice(dense_units),
                'aggregation': random.choice(aggregations),
                'use_embedding': random.choice(use_embeddings)
            })
        return configs


def config_to_args(config: Dict, base_args: List[str]) -> List[str]:
    """
    Convert config dictionary to command-line arguments.
    
    Args:
        config: Hyperparameter dictionary
        base_args: Base command arguments
    
    Returns:
        List of command-line arguments
    """
    args = base_args.copy()
    
    args.extend(['--model_type', config['model_type']])
    args.extend(['--learning_rate', str(config['learning_rate'])])
    args.extend(['--batch_size', str(config['batch_size'])])
    args.extend(['--dropout_rate', str(config['dropout_rate'])])
    
    if config['model_type'] == 'cnn':
        args.extend(['--num_filters', str(config['num_filters'])])
        args.extend(['--filter_sizes'] + [str(fs) for fs in config['filter_sizes']])
        args.extend(['--hidden_units', str(config['hidden_units'])])
        args.extend(['--num_dense_layers', str(config['num_dense_layers'])])
    else:  # lstm
        args.extend(['--lstm_hidden_size', str(config['lstm_hidden_size'])])
        args.extend(['--num_lstm_layers', str(config['num_lstm_layers'])])
        args.extend(['--dense_units', str(config['dense_units'])])
        args.extend(['--aggregation', config['aggregation']])
        if config.get('use_embedding', False):
            args.append('--use_embedding')
            args.extend(['--embedding_dim', '32'])
    
    return args


def run_training(config: Dict, train_script: str, protein: str, num_epochs: int, 
                 patience: int, seed: int, save_dir: str) -> Dict:
    """
    Run training for a single configuration.
    
    Args:
        config: Hyperparameter configuration
        train_script: Path to training script
        protein: Protein name
        num_epochs: Number of epochs
        patience: Early stopping patience
        seed: Random seed
        save_dir: Save directory
    
    Returns:
        Dictionary with results
    """
    base_args = [
        'python', train_script,
        '--protein', protein,
        '--num_epochs', str(num_epochs),
        '--patience', str(patience),
        '--seed', str(seed),
        '--save_dir', save_dir
    ]
    
    args = config_to_args(config, base_args)
    
    print(f"\n{'='*80}")
    print(f"Training configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print(f"{'='*80}\n")
    
    # Run training
    try:
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            check=True
        )
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"Error during training: {e}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return {
            'config': config,
            'success': False,
            'best_val_correlation': -1.0,
            'error': str(e)
        }
    
    # Load results from history file
    model_type = config['model_type']
    model_dir = os.path.join(save_dir, model_type)
    
    # Find the history file (it will have a name based on hyperparameters)
    history_files = list(Path(model_dir).glob('*_history.json'))
    if not history_files:
        return {
            'config': config,
            'success': False,
            'best_val_correlation': -1.0,
            'error': 'History file not found'
        }
    
    # Get the most recent history file
    latest_history = max(history_files, key=lambda p: p.stat().st_mtime)
    
    with open(latest_history, 'r') as f:
        history = json.load(f)
    
    return {
        'config': config,
        'success': True,
        'best_val_correlation': history['best_val_correlation'],
        'num_epochs_trained': history['num_epochs_trained']
    }


def main():
    parser = argparse.ArgumentParser(description='Hyperparameter Tuning for RNA Binding Models')
    
    parser.add_argument('--model_type', type=str, default='both', choices=['cnn', 'lstm', 'both'],
                        help='Which model(s) to tune')
    parser.add_argument('--protein', type=str, default='RBFOX1', help='Protein name')
    parser.add_argument('--search_type', type=str, default='random', choices=['grid', 'random'],
                        help='Search type: grid or random')
    parser.add_argument('--num_random', type=int, default=20,
                        help='Number of random configurations (if random search)')
    parser.add_argument('--num_epochs', type=int, default=100, help='Max epochs per run')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--save_dir', type=str, default='results', help='Results directory')
    parser.add_argument('--train_script', type=str, default='hw2_q2_train.py',
                        help='Path to training script')
    
    args = parser.parse_args()
    
    # Generate configurations
    all_configs = []
    
    if args.model_type in ['cnn', 'both']:
        cnn_configs = generate_cnn_configs(
            grid_search=(args.search_type == 'grid'),
            num_random=args.num_random
        )
        all_configs.extend(cnn_configs)
        print(f"Generated {len(cnn_configs)} CNN configurations")
    
    if args.model_type in ['lstm', 'both']:
        lstm_configs = generate_lstm_configs(
            grid_search=(args.search_type == 'grid'),
            num_random=args.num_random
        )
        all_configs.extend(lstm_configs)
        print(f"Generated {len(lstm_configs)} LSTM configurations")
    
    print(f"Total configurations to try: {len(all_configs)}")
    
    # Run training for each configuration
    results = []
    for i, config in enumerate(all_configs, 1):
        print(f"\n\n{'#'*80}")
        print(f"Configuration {i}/{len(all_configs)}")
        print(f"{'#'*80}")
        
        result = run_training(
            config=config,
            train_script=args.train_script,
            protein=args.protein,
            num_epochs=args.num_epochs,
            patience=args.patience,
            seed=args.seed,
            save_dir=args.save_dir
        )
        results.append(result)
    
    # Find best configurations
    cnn_results = [r for r in results if r['config']['model_type'] == 'cnn' and r['success']]
    lstm_results = [r for r in results if r['config']['model_type'] == 'lstm' and r['success']]
    
    best_cnn = max(cnn_results, key=lambda x: x['best_val_correlation']) if cnn_results else None
    best_lstm = max(lstm_results, key=lambda x: x['best_val_correlation']) if lstm_results else None
    
    # Save results
    results_dir = os.path.join(args.save_dir, 'tuning_results')
    os.makedirs(results_dir, exist_ok=True)
    
    results_file = os.path.join(results_dir, 'all_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save best configurations
    best_configs = {}
    if best_cnn:
        best_configs['cnn'] = {
            'config': best_cnn['config'],
            'best_val_correlation': best_cnn['best_val_correlation']
        }
    if best_lstm:
        best_configs['lstm'] = {
            'config': best_lstm['config'],
            'best_val_correlation': best_lstm['best_val_correlation']
        }
    
    best_file = os.path.join(results_dir, 'best_configs.json')
    with open(best_file, 'w') as f:
        json.dump(best_configs, f, indent=2)
    
    # Print summary
    print(f"\n\n{'='*80}")
    print("TUNING SUMMARY")
    print(f"{'='*80}")
    
    if best_cnn:
        print(f"\nBest CNN Configuration:")
        print(f"  Validation Correlation: {best_cnn['best_val_correlation']:.4f}")
        print(f"  Hyperparameters:")
        for key, value in best_cnn['config'].items():
            print(f"    {key}: {value}")
    
    if best_lstm:
        print(f"\nBest LSTM Configuration:")
        print(f"  Validation Correlation: {best_lstm['best_val_correlation']:.4f}")
        print(f"  Hyperparameters:")
        for key, value in best_lstm['config'].items():
            print(f"    {key}: {value}")
    
    print(f"\nResults saved to: {results_dir}")


if __name__ == '__main__':
    main()

