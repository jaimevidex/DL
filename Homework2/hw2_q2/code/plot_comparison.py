import json
import matplotlib.pyplot as plt
import os
import sys

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def main():
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 1. Load Baseline Data
    # Assuming the results.json is in Homework2/hw2_q2/results/results.json
    # relative to code/ it is ../results/results.json
    baseline_path = os.path.join(script_dir, '..', 'results', 'results.json')
    try:
        baseline_data = load_json(baseline_path)
        # Extract LSTM baseline
        if 'lstm' in baseline_data:
            baseline_hist = baseline_data['lstm']['training_history']
        else:
            print("Error: 'lstm' key not found in baseline results.json")
            return
    except FileNotFoundError:
        print(f"Error: Baseline file not found at {baseline_path}")
        return

    # 2. Load Attention Data
    attention_path = os.path.join(script_dir, 'results', 'attention_study', 'history.json')
    try:
        attention_hist = load_json(attention_path)
    except FileNotFoundError:
        print(f"Error: Attention history file not found at {attention_path}")
        return

    # 3. Prepare Data
    # Baseline
    b_train_loss = baseline_hist['train_losses']
    b_val_loss = baseline_hist['val_losses']
    # Some older result files might store 'val_correlations' or 'val_accs'
    b_val_corr = baseline_hist.get('val_correlations', [])
    
    b_epochs = range(1, len(b_train_loss) + 1)

    # Attention
    a_train_loss = attention_hist['train_losses']
    a_val_loss = attention_hist['val_losses']
    a_val_corr = attention_hist.get('val_correlations', [])
    
    a_epochs = range(1, len(a_train_loss) + 1)

    # 4. Plotting
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: Training Loss
    axes[0].plot(b_epochs, b_train_loss, label='Baseline LSTM (No Attention)', linestyle='--', color='gray')
    axes[0].plot(a_epochs, a_train_loss, label='Attention LSTM', color='blue')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('MSE Loss')
    axes[0].set_title('Training Loss Comparison')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Validation Loss
    axes[1].plot(b_epochs, b_val_loss, label='Baseline LSTM (No Attention)', linestyle='--', color='gray')
    axes[1].plot(a_epochs, a_val_loss, label='Attention LSTM', color='orange')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('MSE Loss')
    axes[1].set_title('Validation Loss Comparison')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Plot 3: Validation Correlation
    if b_val_corr and a_val_corr:
        axes[2].plot(b_epochs, b_val_corr, label='Baseline LSTM (No Attention)', linestyle='--', color='gray')
        axes[2].plot(a_epochs, a_val_corr, label='Attention LSTM', color='green')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Spearman Correlation')
        axes[2].set_title('Validation Correlation Comparison')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 5. Save
    save_dir = os.path.dirname(attention_path)
    save_path = os.path.join(save_dir, 'comparison_curves.pdf')
    plt.savefig(save_path)
    plt.close()
    
    print(f"Comparison plot saved to: {save_path}")
    print("\nFinal Stats Comparison:")
    print(f"Baseline Best Corr: {max(b_val_corr):.4f}")
    print(f"Attention Best Corr: {max(a_val_corr):.4f}")

if __name__ == '__main__':
    main()

