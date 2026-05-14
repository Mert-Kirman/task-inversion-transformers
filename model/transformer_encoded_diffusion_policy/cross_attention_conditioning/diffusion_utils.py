import torch
import torch.nn as nn
import math


class DDPMScheduler(nn.Module):
    def __init__(self, num_train_timesteps=100, beta_start=1e-4, beta_end=0.02):
        """
        Calculates the pre-computed variance schedules for the diffusion process.
        For continuous robotic trajectories, 100 timesteps is usually plenty 
        (unlike images which often require 1000).
        """
        super().__init__()
        self.num_train_timesteps = num_train_timesteps
        
        # Define the linear variance schedule (Beta)
        betas = torch.linspace(beta_start, beta_end, num_train_timesteps)
        
        # Calculate the Alphas
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        # Pre-calculate the square roots used in the forward DDPM equation
        # We register these as buffers so PyTorch handles device placement automatically
        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))
        
    def add_noise(self, original_samples, noise, timesteps):
        """
        The Forward Process (Training).
        Adds exact amounts of Gaussian noise to the clean trajectory based on the timestep 't'.
        
        original_samples: Ground truth trajectory (Batch, Seq_len, d_y)
        noise: Random Gaussian tensor of the exact same shape
        timesteps: Random integer step between 0 and num_train_timesteps-1 (Batch,)
        """
        # Extract the correct coefficients for the current batch's timesteps
        sqrt_alpha_prod = self.sqrt_alphas_cumprod[timesteps]
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod[timesteps]
        
        # Reshape to allow broadcasting over the sequence length and feature dimensions
        # Transforms shape from (Batch,) to (Batch, 1, 1)
        sqrt_alpha_prod = sqrt_alpha_prod.view(-1, 1, 1)
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.view(-1, 1, 1)
        
        # Apply the standard DDPM forward equation
        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        
        return noisy_samples


class SinusoidalPosEmb(nn.Module):
    """
    Embeds the integer diffusion timestep into a continuous vector.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        # x is a tensor of timesteps: (Batch,)
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class ConditionalResidualBlock1D(nn.Module):
    """
    A 1D Convolutional block that fuses the global conditioning (Timestep + Latent)
    into the sequence features.
    """
    def __init__(self, in_channels, out_channels, cond_dim):
        super().__init__()
        # Standard 1D convolutions for processing the trajectory
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.act = nn.Mish() # Mish is the standard activation for diffusion models
        
        # Projection layer for the condition vector
        self.cond_proj = nn.Linear(cond_dim, out_channels)
        
        # Skip connection match
        self.residual_proj = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, cond):
        # x shape: (Batch, in_channels, seq_len)
        # cond shape: (Batch, cond_dim)
        
        residual = self.residual_proj(x)

        x = self.norm1(x)
        x = self.act(x)
        x = self.conv1(x)

        # Fuse the conditioning vector by broadcasting it across the sequence length
        # cond_proj(cond) shape -> (Batch, out_channels)
        # unsqueeze(-1) -> (Batch, out_channels, 1)
        c = self.cond_proj(cond).unsqueeze(-1)
        x = x + c 

        x = self.norm2(x)
        x = self.act(x)
        x = self.conv2(x)

        return x + residual


class CrossAttention1D(nn.Module):
    """
    Allows the U-Net to look at the un-pooled Transformer sequence.
    The U-Net feature map acts as the Query, and the Transformer sequence acts as Key/Value.
    """
    def __init__(self, query_dim, context_dim, nhead=4):
        super().__init__()
        self.norm = nn.GroupNorm(8, query_dim)
        
        # kdim and vdim allow us to attend to the 256-dim context while the U-Net features are a different dimension
        self.attn = nn.MultiheadAttention(
            embed_dim=query_dim, 
            kdim=context_dim, 
            vdim=context_dim, 
            num_heads=nhead, 
            batch_first=True
        )

    def forward(self, x, context_seq):
        # x: U-Net feature map (Batch, Channels, Length) -> (Batch, Length, Channels)
        x_seq = x.transpose(1, 2)
        
        # Pre-Norm for stability
        x_norm = self.norm(x).transpose(1, 2)
        
        # Cross-Attention: 
        # Query = U-Net features. Key/Value = Transformer Output
        # Even if x_norm is length 50 (due to pooling), it can attend to the full length 200 context_seq
        attn_out, _ = self.attn(query=x_norm, key=context_seq, value=context_seq)
        
        # Residual connection
        out = x_seq + attn_out
        
        # Transpose back to U-Net channel-first format (Batch, Channels, Length)
        return out.transpose(1, 2)


class ConditionalUNet1D(nn.Module):
    def __init__(self, input_dim, global_cond_dim, context_dim, base_channels=64):
        super().__init__()
        
        self.time_emb_dim = 64
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(self.time_emb_dim),
            nn.Linear(self.time_emb_dim, self.time_emb_dim * 2),
            nn.Mish(),
            nn.Linear(self.time_emb_dim * 2, self.time_emb_dim)
        )

        # Global Cond is Time Embedding + Task Parameters (e.g., 64 + 16)
        total_global_cond_dim = global_cond_dim + self.time_emb_dim

        self.init_conv = nn.Conv1d(input_dim, base_channels, kernel_size=3, padding=1)

        # Downsampling path (Reduces sequence length, increases channels)
        # Sequence: 200 -> 100 -> 50
        self.down1 = ConditionalResidualBlock1D(base_channels, base_channels * 2, total_global_cond_dim)
        self.attn_down1 = CrossAttention1D(base_channels * 2, context_dim)
        
        self.down2 = ConditionalResidualBlock1D(base_channels * 2, base_channels * 4, total_global_cond_dim)
        self.attn_down2 = CrossAttention1D(base_channels * 4, context_dim)
        
        self.pool = nn.MaxPool1d(kernel_size=2)

        # Bottleneck (Processes the most compressed, feature-rich trajectory representation)
        self.bottleneck = ConditionalResidualBlock1D(base_channels * 4, base_channels * 4, total_global_cond_dim)
        self.attn_mid = CrossAttention1D(base_channels * 4, context_dim)

        # Upsampling path (Increases sequence length, decreases channels)
        # Sequence: 50 -> 100 -> 200
        self.up_sample = nn.Upsample(scale_factor=2, mode='nearest')
        
        # The in_channels are multiplied by 2 because of skip connection concatenation
        self.up1 = ConditionalResidualBlock1D(base_channels * 8, base_channels * 2, total_global_cond_dim)
        self.attn_up1 = CrossAttention1D(base_channels * 2, context_dim)
        
        self.up2 = ConditionalResidualBlock1D(base_channels * 4, base_channels, total_global_cond_dim)
        self.attn_up2 = CrossAttention1D(base_channels, context_dim)

        # Final Projection back to physical coordinates (e.g., 3 for X, Y, Z)
        self.final_conv = nn.Conv1d(base_channels, input_dim, kernel_size=3, padding=1)

    def forward(self, x, timesteps, global_cond, context_seq):
        """
        global_cond: (Batch, global_cond_dim) - Used for global shift addition
        context_seq: (Batch, Seq_len, context_dim) - Used for cross-attention mapping
        """
        # PyTorch Conv1d expects shape (Batch, Channels, Seq_len)
        # Transpose the input trajectory from (Batch, Seq_len, input_dim) to (Batch, input_dim, Seq_len)
        x = x.transpose(1, 2) 

        t_emb = self.time_mlp(timesteps) # (Batch, time_emb_dim)
        g_cond = torch.cat([t_emb, global_cond], dim=-1) # (Batch, total_cond_dim)

        x = self.init_conv(x) # (Batch, base_channels, 200)

        # Down-path
        skip1 = self.down1(x, g_cond)
        skip1 = self.attn_down1(skip1, context_seq)
        x_down = self.pool(skip1)
        
        skip2 = self.down2(x_down, g_cond)
        skip2 = self.attn_down2(skip2, context_seq)
        x_down = self.pool(skip2)

        # Bottleneck
        x_mid = self.bottleneck(x_down, g_cond)
        x_mid = self.attn_mid(x_mid, context_seq)

        # Up-path
        x_up = self.up_sample(x_mid)
        x_up = torch.cat([x_up, skip2], dim=1)
        x_up = self.up1(x_up, g_cond)
        x_up = self.attn_up1(x_up, context_seq)

        x_up = self.up_sample(x_up)
        x_up = torch.cat([x_up, skip1], dim=1) # (Batch, base_channels*4, 200)
        x_up = self.up2(x_up, g_cond) # (Batch, base_channels, 200)
        x_up = self.attn_up2(x_up, context_seq)

        # Transpose back to standard sequence shape (Batch, Seq_len, d_y)
        out = self.final_conv(x_up) # (Batch, d_y, 200)
        return out.transpose(1, 2)
