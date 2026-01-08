import json
import matplotlib.pyplot as plt
import os
import numpy as np

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 1. Load Baseline Data (results.json)
    baseline_path = os.path.join(script_dir, '..', 'results', 'results.json')
    try:
        baseline_data = load_json(baseline_path)
        # Extract Baseline metrics
        lstm_res = baseline_data['lstm']['test_results']
        cnn_res = baseline_data['cnn']['test_results']
    except Exception as e:
        print(f"Error loading baseline: {e}")
        return

    # 2. Load Attention Data
    attention_path = os.path.join(script_dir, '..', 'results', 'attention_study', 'test_results.json')
    try:
        attn_res = load_json(attention_path)
    except Exception as e:
        print(f"Error loading attention results: {e}")
        return

    # 3. Prepare Data
    models = ['CNN', 'LSTM (Baseline)', 'LSTM (Attention)']
    
    spearman = [
        cnn_res['spearman_correlation'],
        lstm_res['spearman_correlation'],
        attn_res['spearman_correlation']
    ]
    
    pearson = [
        cnn_res['pearson_correlation'],
        lstm_res['pearson_correlation'],
        attn_res['pearson_correlation']
    ]
    
    losses = [
        cnn_res['test_loss'],
        lstm_res['test_loss'],
        attn_res['test_loss']
    ]
    
    # 4. Plotting
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Colors
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    # Plot 1: Spearman
    axes[0].bar(models, spearman, color=colors)
    axes[0].set_ylabel('Spearman Correlation')
    axes[0].set_title('Spearman Correlation (Higher is Better)')
    axes[0].set_ylim([0, 1.0])
    for i, v in enumerate(spearman):
        axes[0].text(i, v + 0.01, f'{v:.3f}', ha='center', fontweight='bold')
    axes[0].grid(True, axis='y', alpha=0.3)

    # Plot 2: Pearson
    axes[1].bar(models, pearson, color=colors)
    axes[1].set_ylabel('Pearson Correlation')
    axes[1].set_title('Pearson Correlation (Higher is Better)')
    axes[1].set_ylim([0, 1.0])
    for i, v in enumerate(pearson):
        axes[1].text(i, v + 0.01, f'{v:.3f}', ha='center', fontweight='bold')
    axes[1].grid(True, axis='y', alpha=0.3)

    # Plot 3: Loss
    axes[2].bar(models, losses, color=colors)
    axes[2].set_ylabel('Test Loss (MSE)')
    axes[2].set_title('Test Loss (Lower is Better)')
    for i, v in enumerate(losses):
        axes[2].text(i, v + 0.01, f'{v:.3f}', ha='center', fontweight='bold')
    axes[2].grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    # 5. Save
    save_path = os.path.join(script_dir, '..', 'results', 'attention_study', 'model_comparison_bar.pdf')
    plt.savefig(save_path)
    plt.close()
    
    print(f"Bar chart comparison saved to: {save_path}")

if __name__ == '__main__':
    main()

