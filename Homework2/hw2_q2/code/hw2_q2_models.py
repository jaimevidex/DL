"""
Model architectures for RNA Binding Protein Affinity Prediction.

This module contains two different deep learning architectures:
1. CNN: Convolutional Neural Network for motif detection
2. LSTM: Bidirectional LSTM for sequence modeling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RNABindingCNN(nn.Module):
    """
    Convolutional Neural Network for RNA binding affinity prediction.
    
    Architecture:
    - Multiple Conv1D layers with different filter sizes to detect motifs
    - Batch normalization and dropout for regularization
    - Global pooling to aggregate features
    - Fully connected layers for regression
    
    Args:
        num_filters: Number of convolutional filters per layer
        filter_sizes: List of filter sizes (e.g., [3, 5, 7])
        dropout_rate: Dropout probability
        hidden_units: Number of units in dense layers
        num_dense_layers: Number of fully connected layers
    """
    
    def __init__(
        self,
        num_filters: int = 64,
        filter_sizes: list = [3, 5, 7],
        dropout_rate: float = 0.2,
        hidden_units: int = 128,
        num_dense_layers: int = 2
    ):
        super(RNABindingCNN, self).__init__()
        
        self.num_filters = num_filters
        self.filter_sizes = filter_sizes
        self.dropout_rate = dropout_rate
        
        # Input shape: (batch, 41, 4)
        # Create parallel convolutional branches for different filter sizes
        self.conv_branches = nn.ModuleList()
        for filter_size in filter_sizes:
            branch = nn.Sequential(
                nn.Conv1d(
                    in_channels=4,  # One-hot encoded nucleotides
                    out_channels=num_filters,
                    kernel_size=filter_size,
                    padding=filter_size // 2  # Same padding
                ),
                nn.BatchNorm1d(num_filters),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                # Second conv layer
                nn.Conv1d(
                    in_channels=num_filters,
                    out_channels=num_filters,
                    kernel_size=filter_size,
                    padding=filter_size // 2
                ),
                nn.BatchNorm1d(num_filters),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            )
            self.conv_branches.append(branch)
        
        # Global pooling: aggregate across sequence length
        # Each branch outputs (batch, num_filters, 41)
        # After pooling: (batch, num_filters)
        # With multiple branches: (batch, num_filters * len(filter_sizes))
        pooled_dim = num_filters * len(filter_sizes)
        
        # Fully connected layers
        dense_layers = []
        input_dim = pooled_dim
        
        for i in range(num_dense_layers):
            dense_layers.append(nn.Linear(input_dim, hidden_units))
            dense_layers.append(nn.BatchNorm1d(hidden_units))
            dense_layers.append(nn.ReLU())
            dense_layers.append(nn.Dropout(dropout_rate))
            input_dim = hidden_units
        
        self.dense = nn.Sequential(*dense_layers)
        
        # Final regression layer
        self.output = nn.Linear(hidden_units, 1)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, 41, 4)
        
        Returns:
            Output tensor of shape (batch, 1)
        """
        # Convert to (batch, 4, 41) for Conv1d
        x = x.transpose(1, 2)  # (batch, 4, 41)
        
        # Apply each convolutional branch
        branch_outputs = []
        for branch in self.conv_branches:
            branch_out = branch(x)  # (batch, num_filters, 41)
            # Global max pooling
            pooled = F.adaptive_max_pool1d(branch_out, 1).squeeze(-1)  # (batch, num_filters)
            branch_outputs.append(pooled)
        
        # Concatenate all branch outputs
        x = torch.cat(branch_outputs, dim=1)  # (batch, num_filters * len(filter_sizes))
        
        # Fully connected layers
        x = self.dense(x)
        
        # Final output
        output = self.output(x)  # (batch, 1)
        
        return output


class RNABindingLSTM(nn.Module):
    """
    Bidirectional LSTM for RNA binding affinity prediction.
    
    Architecture:
    - Optional embedding layer for nucleotide embeddings
    - Bidirectional LSTM layers to capture sequence context
    - Sequence aggregation (last hidden state or attention)
    - Fully connected layers for regression
    
    Args:
        lstm_hidden_size: Hidden size of LSTM units
        num_lstm_layers: Number of LSTM layers
        dropout_rate: Dropout probability
        dense_units: Number of units in dense layers
        use_embedding: Whether to use learnable embeddings (if False, uses one-hot directly)
        embedding_dim: Dimension of embeddings (if use_embedding=True)
        aggregation: 'last' or 'attention' for sequence aggregation
    """
    
    def __init__(
        self,
        lstm_hidden_size: int = 128,
        num_lstm_layers: int = 2,
        dropout_rate: float = 0.2,
        dense_units: int = 128,
        use_embedding: bool = False,
        embedding_dim: int = 32,
        aggregation: str = 'last'
    ):
        super(RNABindingLSTM, self).__init__()
        
        self.lstm_hidden_size = lstm_hidden_size
        self.num_lstm_layers = num_lstm_layers
        self.use_embedding = use_embedding
        self.aggregation = aggregation
        
        # Embedding layer (optional)
        if use_embedding:
            # Map from 4 (one-hot) to embedding_dim
            self.embedding = nn.Linear(4, embedding_dim)
            lstm_input_size = embedding_dim
        else:
            self.embedding = None
            lstm_input_size = 4  # Direct one-hot input
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=lstm_hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_rate if num_lstm_layers > 1 else 0
        )
        
        # LSTM output size is 2 * hidden_size (bidirectional)
        lstm_output_size = 2 * lstm_hidden_size
        
        # Attention mechanism (if using attention aggregation)
        if aggregation == 'attention':
            self.attention = nn.Sequential(
                nn.Linear(lstm_output_size, lstm_output_size),
                nn.Tanh(),
                nn.Linear(lstm_output_size, 1)
            )
        
        # Fully connected layers
        self.dense = nn.Sequential(
            nn.Linear(lstm_output_size, dense_units),
            nn.BatchNorm1d(dense_units),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dense_units, dense_units // 2),
            nn.BatchNorm1d(dense_units // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Final regression layer
        self.output = nn.Linear(dense_units // 2, 1)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, 41, 4)
        
        Returns:
            Output tensor of shape (batch, 1)
        """
        # Optional embedding
        if self.use_embedding:
            x = self.embedding(x)  # (batch, 41, embedding_dim)
        
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)  # lstm_out: (batch, 41, 2*hidden_size)
        
        # Sequence aggregation
        if self.aggregation == 'last':
            # Use last hidden state from both directions
            # hidden shape: (num_layers * 2, batch, hidden_size)
            # Take the last layer's forward and backward hidden states
            forward_hidden = hidden[-2]  # Last forward hidden state
            backward_hidden = hidden[-1]  # Last backward hidden state
            x = torch.cat([forward_hidden, backward_hidden], dim=1)  # (batch, 2*hidden_size)
        
        elif self.aggregation == 'attention':
            # Attention-weighted aggregation
            attention_weights = self.attention(lstm_out)  # (batch, 41, 1)
            attention_weights = F.softmax(attention_weights, dim=1)  # Normalize over sequence
            x = torch.sum(attention_weights * lstm_out, dim=1)  # (batch, 2*hidden_size)
        
        else:
            # Default: mean pooling
            x = torch.mean(lstm_out, dim=1)  # (batch, 2*hidden_size)
        
        # Fully connected layers
        x = self.dense(x)
        
        # Final output
        output = self.output(x)  # (batch, 1)
        
        return output


def create_model(model_type: str, **kwargs):
    """
    Factory function to create a model.
    
    Args:
        model_type: 'cnn' or 'lstm'
        **kwargs: Model-specific hyperparameters
    
    Returns:
        Model instance
    """
    if model_type.lower() == 'cnn':
        return RNABindingCNN(**kwargs)
    elif model_type.lower() == 'lstm':
        return RNABindingLSTM(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Use 'cnn' or 'lstm'.")

