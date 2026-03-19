import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import model.loss_utils as loss_utils

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


class AttentionPooling(nn.Module):
    """
    Learns to dynamically weight the most important time steps in the trajectory.
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention_weights = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, last_hidden_state):
        # Calculate scores: (Batch, Seq_Len, 1)
        scores = self.attention_weights(last_hidden_state)
        alphas = F.softmax(scores, dim=1)
        # Weighted sum: (Batch, Hidden_Dim)
        context_vector = torch.sum(alphas * last_hidden_state, dim=1)
        return context_vector


class TransformerTrajectoryEncoder(nn.Module):
    def __init__(self, input_dim, d_model=128, nhead=4, num_layers=2, dropout=0.2):
        super().__init__()
        # 1. Project physical dimensions (e.g., 3) up to the Transformer's hidden dimension
        self.input_proj = nn.Linear(input_dim, d_model)
        
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

        # Attention Pooling (Replaces Mean Pooling)
        self.pooler = AttentionPooling(d_model)

    def forward(self, seq):
        # seq shape: (batch_size, time_len, input_dim)
        x = self.input_proj(seq) # (batch_size, time_len, d_model)
        x = self.pos_encoder(x)
        
        # Process the whole sequence simultaneously
        encoded_seq = self.transformer(x)
        
        # Apply attention pooling to get a single latent vector
        latent = self.pooler(encoded_seq) # (batch_size, d_model)
        
        return latent


class TempModel(nn.Module):
    def __init__(self, d_x, d_y1, d_y2, d_param, dropout_p=[0.2, 0.0]):
        super(TempModel, self).__init__()

        self.d_x = d_x
        self.d_y1 = d_y1
        self.d_y2 = d_y2
        self.param_dim = d_param
        self.embedding_dim = 16
        self.d_model = 128

        p_enc = dropout_p[0] # Probability for Encoder (Used in Transformer)
        p_dec = dropout_p[1] # Probability for Decoder (Used in MLP)

        # --- Parameter Embedder ---
        self.param_embedder = nn.Sequential(
            nn.Linear(d_param, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, self.embedding_dim)
        )

        # --- BERT Encoders ---
        self.encoder1 = TransformerTrajectoryEncoder(
            input_dim=d_y1, 
            d_model=self.d_model, 
            nhead=4, 
            num_layers=2, 
            dropout=p_enc
        )
        
        self.encoder2 = TransformerTrajectoryEncoder(
            input_dim=d_y2, 
            d_model=self.d_model, 
            nhead=4, 
            num_layers=2, 
            dropout=p_enc
        )

        # --- MLP Decoders ---
        self.decoder1 = nn.Sequential(
            nn.Linear(d_x + self.d_model + self.embedding_dim, 256), nn.LayerNorm(256), nn.ReLU(), nn.Dropout(p_dec), 
            nn.Linear(256, 256), nn.LayerNorm(256), nn.ReLU(), nn.Dropout(p_dec),
            nn.Linear(256, 128), nn.LayerNorm(128), nn.ReLU(), nn.Dropout(p_dec),
            nn.Linear(128, 64), nn.LayerNorm(64), nn.ReLU(), nn.Dropout(p_dec),
            nn.Linear(64, (d_y1)*2)
        )

        self.decoder2 = nn.Sequential(
            nn.Linear(d_x + self.d_model + self.embedding_dim, 256), nn.LayerNorm(256), nn.ReLU(), nn.Dropout(p_dec), 
            nn.Linear(256, 256), nn.LayerNorm(256), nn.ReLU(), nn.Dropout(p_dec),
            nn.Linear(256, 128), nn.LayerNorm(128), nn.ReLU(), nn.Dropout(p_dec),
            nn.Linear(128, 64), nn.LayerNorm(64), nn.ReLU(), nn.Dropout(p_dec),
            nn.Linear(64, (d_y2)*2)
        )

    def forward(self, y1_seq, y2_seq, params, x_tar, extra_pass, p=0):
        """
        y1_seq: (batch_size, time_len, d_y1)
        y2_seq: (batch_size, time_len, d_y2)
        params: (batch_size, 1, d_param)
        x_tar:  (batch_size, num_tar, d_x)
        """
        device = y1_seq.device
        
        # 1. Embed Task Parameters
        p_embedded = self.param_embedder(params) # (batch, 1, 16)
        p_expanded = p_embedded.expand(-1, x_tar.shape[1], -1) # (batch, num_tar, 16)

        # 2. Encode Forward Trajectory
        L_F = self.encoder1(y1_seq) # (batch, d_model)
        L_F = L_F.unsqueeze(1).expand(-1, x_tar.shape[1], -1) # (batch, num_tar, d_model)

        # 3. Encode Inverse Trajectory (Skip if extra_pass to save compute/avoid garbage data)
        if not extra_pass:
            L_I = self.encoder2(y2_seq) # (batch, d_model)
            L_I = L_I.unsqueeze(1).expand(-1, x_tar.shape[1], -1) # (batch, num_tar, d_model)
        else:
            L_I = L_F # Dummy assignment for loss calculation to return 0

        # 4. Route the Latent Vector
        latent = torch.zeros_like(L_F)
        if p == 0:
            if not extra_pass:
                # Discrete Routing for Latent Alignment
                if torch.rand(1, device=device).item() < 0.5:
                    latent = L_F
                else:
                    latent = L_I
            else:
                latent = L_F
        elif p == 1:
            latent = L_F # Inference: Conditioned on Forward
        elif p == 2:
            latent = L_I # Inference: Conditioned on Inverse

        # 5. Decode
        # concat shape: (batch_size, num_tar, d_model + 16 + d_x)
        concat = torch.cat((latent, p_expanded, x_tar), dim=-1)  
        
        output1 = self.decoder1(concat)  # (batch_size, num_tar, 2*d_y1)

        if extra_pass:
            return torch.cat((output1, output1), dim=-1), L_F, L_F, extra_pass

        output2 = self.decoder2(concat)  # (batch_size, num_tar, 2*d_y2)
        return torch.cat((output1, output2), dim=-1), L_F, L_I, extra_pass


def loss(output, target_f, target_i, d_y1, d_y2, d_param, L_F, L_I, extra_pass):
    # Standard log probability loss
    log_prob = loss_utils.log_prob_loss(output, target_f, target_i, d_y1, d_y2, d_param, extra_pass)
    
    # Raw Latent Alignment Loss (Preserves spatial magnitude)
    latent_alignment_loss = torch.mean((L_F - L_I) ** 2)

    lambda1 = 1.0  
    lambda2 = 0.01  # Scaled down to prevent overpowering log_prob

    return lambda1 * log_prob + lambda2 * latent_alignment_loss
