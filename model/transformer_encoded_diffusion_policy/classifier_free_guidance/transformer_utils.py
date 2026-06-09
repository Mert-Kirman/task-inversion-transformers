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
        self.input_proj = nn.Linear(input_dim, d_model)

        # The Continuous [MASK] Token
        self.mask_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # NEW: A learnable Global Collection Token (CLS)
        # This token is NEVER masked and extracts the latent representation of the entire trajectory
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model * 4, 
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # NEW: Latent Bottleneck Normalization to stabilize the distribution for the MLP decoder
        self.bottleneck_norm = nn.LayerNorm(d_model)

    def forward(self, seq, mask_indices=None):
        # seq shape: (batch_size, time_len, input_dim)
        # mask_indices shape: (batch_size, time_len) containing booleans (True = Mask this point)
        batch_size, time_len, _ = seq.shape

        # 1. Project to d_model
        x = self.input_proj(seq) 
        
        # 2. Apply the [MASK] token to coordinates
        if mask_indices is not None:
            expanded_mask = mask_indices.unsqueeze(-1).expand(-1, -1, x.size(-1))
            x = torch.where(expanded_mask, self.mask_token, x)

        # 3. Add time awareness to the 200 trajectory points
        x = self.pos_encoder(x)

        # 4. NEW: Prepend the CLS token to the front of the sequence (making length 201)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1) # (batch_size, 1, d_model)
        x = torch.cat((cls_tokens, x), dim=1) # (batch_size, time_len + 1, d_model)

        # 5. NEW: Adjust the attention mask to account for the CLS token
        if mask_indices is not None:
            # The CLS token at index 0 must NEVER be masked (False = keep)
            cls_mask_flag = torch.zeros((batch_size, 1), dtype=torch.bool, device=seq.device)
            # Concat to create a mask of length 201
            extended_mask = torch.cat((cls_mask_flag, mask_indices), dim=1)
            
            # Pass the extended mask to the transformer
            encoded_seq = self.transformer(x, src_key_padding_mask=extended_mask)
        else:
            encoded_seq = self.transformer(x)
        
        # 6. NEW: Extract ONLY the CLS token output (index 0)
        # Because the CLS token is never masked, self-attention allows it to read 
        # features from all unmasked observation points and aggregate them dynamically.
        latent = encoded_seq[:, 0, :] # (batch_size, d_model)
        
        # 7. NEW: Stabilize the latent representation scale before sending to the MLP Decoder
        latent = self.bottleneck_norm(latent)
            
        return latent
