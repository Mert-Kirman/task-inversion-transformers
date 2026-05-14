import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """
    Injects information about the relative or absolute position of the 
    tokens in the sequence. Crucial for Transformers since they don't 
    have an inherent sense of time/order.
    """
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch_size, time_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return x


class TransformerTrajectoryEncoder(nn.Module):
    def __init__(self, input_dim, d_model=256, nhead=8, num_layers=4, dropout=0.1):
        super().__init__()
        # 1. Project physical dimensions (e.g., 3) up to the Transformer's hidden dimension
        self.input_proj = nn.Linear(input_dim, d_model)

        # The Continuous [MASK] Token: Tells the network "this coordinate is missing" during training, encouraging it to learn robust representations
        self.mask_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # 2. Add time awareness
        self.pos_encoder = PositionalEncoding(d_model)
        
        # 3. The BERT Stack
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model * 4, 
            dropout=dropout,
            batch_first=True # Expects (batch, seq, feature) instead of (seq, batch, feature)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, seq, mask_indices=None):
        # seq shape: (batch_size, time_len, input_dim)
        # mask_indices shape: (batch_size, time_len) containing booleans (True = Mask this point)

        # Project to d_model
        x = self.input_proj(seq) # (batch_size, time_len, d_model)
        
        # Apply the [MASK] token
        if mask_indices is not None:
            # Expand boolean mask to match feature dimension
            expanded_mask = mask_indices.unsqueeze(-1).expand(-1, -1, x.size(-1))
            # Replace True indices with the learnable mask token
            x = torch.where(expanded_mask, self.mask_token, x)

        # Add time awareness to the known points AND the mask tokens
        x = self.pos_encoder(x)

        # Process the whole sequence simultaneously
        encoded_seq = self.transformer(x)
        
        return encoded_seq
