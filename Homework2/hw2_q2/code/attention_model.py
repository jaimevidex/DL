import torch
import torch.nn as nn
import torch.nn.functional as F

class RNABindingLSTMAttention(nn.Module):
    """
    Bidirectional LSTM for RNA binding affinity prediction with Single-Head Attention.
    
    Architecture:
    - Optional embedding layer for nucleotide embeddings
    - Bidirectional LSTM layers to capture sequence context
    - Single-Head Attention Aggregation
    - Fully connected layers for regression
    """
    
    def __init__(
        self,
        lstm_hidden_size: int = 128,
        num_lstm_layers: int = 2,
        dropout_rate: float = 0.2,
        dense_units: int = 128,
        use_embedding: bool = False,
        embedding_dim: int = 32
    ):
        super(RNABindingLSTMAttention, self).__init__()
        
        self.lstm_hidden_size = lstm_hidden_size
        self.num_lstm_layers = num_lstm_layers
        self.use_embedding = use_embedding
        
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
        
        # Single-Head Attention Mechanism
        # Projects the LSTM output to a scalar score per time step
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
    
    def forward(self, x, return_attention=False):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, 41, 4)
            return_attention: If True, returns (output, attention_weights)
        
        Returns:
            output: Tensor of shape (batch, 1)
            attention_weights (optional): Tensor of shape (batch, 41, 1)
        """
        # Optional embedding
        if self.use_embedding:
            x = self.embedding(x)  # (batch, 41, embedding_dim)
        
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)  # lstm_out: (batch, 41, 2*hidden_size)
        
        # Attention-weighted aggregation
        # 1. Calculate raw scores
        attention_scores = self.attention(lstm_out)  # (batch, 41, 1)
        
        # 2. Normalize with Softmax to get probability distribution
        attention_weights = F.softmax(attention_scores, dim=1)  # (batch, 41, 1)
        
        # 3. Weighted sum of LSTM outputs
        # context_vector: (batch, 2*hidden_size)
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)
        
        # Fully connected layers
        x_dense = self.dense(context_vector)
        
        # Final output
        output = self.output(x_dense)  # (batch, 1)
        
        if return_attention:
            return output, attention_weights
        
        return output

