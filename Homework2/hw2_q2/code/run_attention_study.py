import torch
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader

# Import local modules
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)
from attention_model import RNABindingLSTMAttention
from train_attention import train_attention_model

# Import skeleton utils
skeleton_path = os.path.join(script_dir, "hw2_q2_skeleton_code")
sys.path.append(skeleton_path)
from utils import load_rnacompete_data, configure_seed


def plot_attention_weights(sequence, attention, title, save_path):
    """Plot attention weights over the sequence."""
    plt.figure(figsize=(12, 4))
    plt.plot(attention, marker="o", linestyle="-", color="b", alpha=0.6)
    plt.xticks(range(len(sequence)), list(sequence), fontsize=10)
    plt.xlabel("Nucleotide Position")
    plt.ylabel("Attention Weight")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def one_hot_to_seq(one_hot):
    """Convert one-hot tensor to sequence string."""
    alphabet = ["A", "C", "G", "U"]  # Assuming this order based on skeleton
    seq = ""
    for vec in one_hot:
        idx = torch.argmax(vec).item()
        # Check if it's a valid one-hot (sum should be 1, if 0 it's padding)
        if vec.sum() > 0.5:
            seq += alphabet[idx]
        else:
            seq += "N"
    return seq


def run_study():
    # 1. Configuration
    config = {
        "protein": "RBFOX1",
        "lstm_hidden_size": 128,
        "num_lstm_layers": 2,
        "dropout_rate": 0.2,
        "dense_units": 128,
        "embedding_dim": 32,
        "use_embedding": False,
        "batch_size": 64,
        "learning_rate": 0.001,
        "num_epochs": 50,
        "patience": 100,
        "seed": 42,
        "save_dir": "results/attention_study",
    }

    print("=" * 50)
    print("Starting Single-Head Attention LSTM Study")
    print("=" * 50)

    # 2. Setup
    configure_seed(config["seed"])
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # 3. Load Data
    print("Loading data...")
    train_dataset = load_rnacompete_data(config["protein"], split="train")
    val_dataset = load_rnacompete_data(config["protein"], split="val")

    train_loader = DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True
    )
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)

    # 4. Initialize Model
    model = RNABindingLSTMAttention(
        lstm_hidden_size=config["lstm_hidden_size"],
        num_lstm_layers=config["num_lstm_layers"],
        dropout_rate=config["dropout_rate"],
        dense_units=config["dense_units"],
        use_embedding=config["use_embedding"],
        embedding_dim=config["embedding_dim"],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

    # 5. Train
    print("Starting training...")
    train_attention_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        num_epochs=config["num_epochs"],
        device=device,
        save_dir=config["save_dir"],
        patience=config["patience"],
    )

    # 6. Visualization Analysis
    print("\nStarting Attention Analysis...")

    # Load best model
    model.load_state_dict(
        torch.load(os.path.join(config["save_dir"], "best_model.pt"), weights_only=True)
    )
    model.eval()

    # Find high affinity sequences in validation set
    all_x = []
    all_y = []

    # Collect a batch of data
    for x, y, mask in val_loader:
        # Move to device first
        x = x.to(device)
        y = y.to(device)
        mask = mask.to(device)

        # Filter out masked items
        # mask is (B, 1). We want a 1D boolean mask of shape (B,)
        valid_indices = mask.view(-1).bool()

        if valid_indices.sum() > 0:
            all_x.append(x[valid_indices])
            all_y.append(y[valid_indices])

    all_x = torch.cat(all_x, dim=0)
    all_y = torch.cat(all_y, dim=0)

    # Get top 5 highest affinity sequences
    # Note: y is binding intensity, higher might mean stronger binding depending on normalization
    # Assuming higher value = higher affinity
    values, indices = torch.topk(all_y.squeeze(), 5)

    print(f"Analyzing top 5 high-affinity sequences...")

    for i, idx in enumerate(indices):
        x_sample = all_x[idx].unsqueeze(0).to(device)  # (1, 41, 4)
        target_val = all_y[idx].item()

        # Get attention weights
        with torch.no_grad():
            pred, attn = model(x_sample, return_attention=True)

        # Process for plotting
        attn_weights = attn.squeeze().cpu().numpy()  # (41,)
        # Convert one-hot to string
        seq_str = one_hot_to_seq(x_sample.squeeze().cpu())

        # Plot
        plot_name = f"top_{i+1}_affinity_{target_val:.2f}.pdf"
        save_path = os.path.join(config["save_dir"], plot_name)
        plot_attention_weights(
            seq_str,
            attn_weights,
            f"Sequence {i+1} (Affinity: {target_val:.2f})",
            save_path,
        )
        print(f"Saved visualization to {save_path}")


if __name__ == "__main__":
    run_study()
