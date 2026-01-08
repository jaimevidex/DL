import torch
import os
import sys
import json
from torch.utils.data import DataLoader

# Import local modules
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)
from attention_model import RNABindingLSTMAttention

# Import skeleton utils
skeleton_path = os.path.join(script_dir, 'hw2_q2_skeleton_code')
sys.path.append(skeleton_path)
from utils import load_rnacompete_data, configure_seed, masked_mse_loss, masked_spearman_correlation
import numpy as np

def evaluate():
    # Configuration
    config = {
        'protein': 'RBFOX1',
        'lstm_hidden_size': 128,
        'num_lstm_layers': 2,
        'dropout_rate': 0.2,
        'dense_units': 128,
        'embedding_dim': 32,
        'use_embedding': False,
        'batch_size': 64,
        'seed': 42,
        'checkpoint': '../results/attention_study/best_model.pt',
        'save_dir': '../results/attention_study'
    }
    
    configure_seed(config['seed'])
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
    print(f"Using device: {device}")
    
    # Load Test Data
    print("Loading Test Data...")
    test_dataset = load_rnacompete_data(config['protein'], split='test')
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
    
    # Initialize Model
    model = RNABindingLSTMAttention(
        lstm_hidden_size=config['lstm_hidden_size'],
        num_lstm_layers=config['num_lstm_layers'],
        dropout_rate=config['dropout_rate'],
        dense_units=config['dense_units'],
        use_embedding=config['use_embedding'],
        embedding_dim=config['embedding_dim']
    ).to(device)
    
    # Load Weights (Handling raw state_dict)
    print(f"Loading weights from {config['checkpoint']}...")
    state_dict = torch.load(config['checkpoint'], map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    
    # Evaluation Loop
    total_loss = 0.0
    all_predictions = []
    all_targets = []
    all_masks = []
    num_batches = 0
    
    print("Evaluating...")
    with torch.no_grad():
        for batch in test_loader:
            x, y, mask = batch
            x = x.to(device)
            y = y.to(device)
            mask = mask.to(device)
            
            predictions = model(x)
            loss = masked_mse_loss(predictions, y, mask)
            total_loss += loss.item()
            
            all_predictions.append(predictions)
            all_targets.append(y)
            all_masks.append(mask)
            num_batches += 1
            
    # Aggregate
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    all_masks = torch.cat(all_masks, dim=0)
    
    # Metrics
    spearman_corr = masked_spearman_correlation(all_predictions, all_targets, all_masks)
    test_loss = total_loss / num_batches if num_batches > 0 else 0.0
    
    # Numpy metrics
    pred_np = all_predictions.squeeze().cpu().numpy()
    target_np = all_targets.squeeze().cpu().numpy()
    mask_np = all_masks.squeeze().cpu().numpy().astype(bool)
    
    valid_preds = pred_np[mask_np]
    valid_targets = target_np[mask_np]
    
    pearson_corr = np.corrcoef(valid_preds, valid_targets)[0, 1]
    mae = np.mean(np.abs(valid_preds - valid_targets))
    rmse = np.sqrt(np.mean((valid_preds - valid_targets) ** 2))
    
    results = {
        "spearman_correlation": float(spearman_corr),
        "pearson_correlation": float(pearson_corr),
        "test_loss": float(test_loss),
        "mae": float(mae),
        "rmse": float(rmse)
    }
    
    print("\nTest Results:")
    print(json.dumps(results, indent=2))
    
    # Save to file
    with open(os.path.join(config['save_dir'], 'test_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == '__main__':
    evaluate()

