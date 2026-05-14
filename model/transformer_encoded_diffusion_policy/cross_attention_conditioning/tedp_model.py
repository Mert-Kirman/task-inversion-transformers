import sys
import os

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.transformer_encoded_diffusion_policy.cross_attention_conditioning.transformer_utils import TransformerTrajectoryEncoder
from model.transformer_encoded_diffusion_policy.cross_attention_conditioning.diffusion_utils import DDPMScheduler, ConditionalUNet1D


class TedpModel(nn.Module):
    def __init__(self, d_x, d_y1, d_y2, d_param, embedding_dim = 16, d_model = 256, nhead=8, num_layers=4, dropout_p=[0.1, 0.0]):
        super(TedpModel, self).__init__()

        self.d_x = d_x
        self.d_y1 = d_y1
        self.d_y2 = d_y2
        self.param_dim = d_param
        self.embedding_dim = embedding_dim
        self.d_model = d_model

        # The Noise Scheduler
        self.num_diffusion_steps = 100
        self.scheduler = DDPMScheduler(num_train_timesteps=self.num_diffusion_steps)

        # Parameter Embedder (Late Fusion Prep)
        self.param_embedder = nn.Sequential(
            nn.Linear(d_param, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, self.embedding_dim)
        )

        # BERT Encoders
        self.encoder1 = TransformerTrajectoryEncoder(
            input_dim=d_y1, 
            d_model=self.d_model, 
            nhead=nhead, 
            num_layers=num_layers, 
            dropout=dropout_p[0]
        )
        
        self.encoder2 = TransformerTrajectoryEncoder(
            input_dim=d_y2, 
            d_model=self.d_model, 
            nhead=nhead, 
            num_layers=num_layers, 
            dropout=dropout_p[0]
        )

        # The U-Net Decoder
        # cond_dim = Pure Motion Latent (256) + Task Param (16)
        cond_dim = self.d_model + self.embedding_dim
        
        self.unet1 = ConditionalUNet1D(input_dim=d_y1, global_cond_dim=self.embedding_dim, context_dim=self.d_model, base_channels=64)
        self.unet2 = ConditionalUNet1D(input_dim=d_y2, global_cond_dim=self.embedding_dim, context_dim=self.d_model, base_channels=64)

    def forward(self, y1_seq, y2_seq, params, extra_pass, p=0, mask_indices_1=None, mask_indices_2=None):
        """
        TRAINING PASS: Corrupts the target trajectory with noise and asks U-Net to predict the noise.

        y1_seq: (batch_size, time_len, d_y1)
        y2_seq: (batch_size, time_len, d_y2)
        params: (batch_size, 1, d_param)
        """
        device = y1_seq.device
        batch_size = y1_seq.shape[0]
        
        # Global Condition (Task Parameters)
        p_embedded = self.param_embedder(params).squeeze(1) # (batch, 16)

        # Sequence Condition (Full Transformer Latent Maps)
        L_F = self.encoder1(y1_seq, mask_indices_1) # (batch, 200, 256)
        if not extra_pass:
            L_I = self.encoder2(y2_seq, mask_indices_2)
        else:
            L_I = L_F

        # Route the Latent Vector
        latent_seq = torch.zeros_like(L_F)
        if p == 0:
            if not extra_pass:
                # Discrete Routing for Latent Alignment
                if torch.rand(1, device=device).item() < 0.5: latent_seq = L_F
                else: latent_seq = L_I
            else:
                latent_seq = L_F
        elif p == 1: latent_seq = L_F # Inference: Conditioned on Forward
        elif p == 2: latent_seq = L_I # Inference: Conditioned on Inverse

        # Diffusion Noise Process
        timesteps = torch.randint(0, self.num_diffusion_steps, (batch_size,), device=device).long()
        
        # Always process the Forward Trajectory (UNet1)
        noise1 = torch.randn_like(y1_seq)
        noisy_y1 = self.scheduler.add_noise(y1_seq, noise1, timesteps)
        # PASS BOTH CONDITIONS TO UNET
        noise_pred1 = self.unet1(noisy_y1, timesteps, global_cond=p_embedded, context_seq=latent_seq) 
        
        if extra_pass:
            # If extra_pass, duplicate the forward output to maintain consistent tensor dimensions
            noise_pred = torch.cat((noise_pred1, noise_pred1), dim=-1)
            noise_truth = torch.cat((noise1, noise1), dim=-1)
        else:
            # Process the Inverse Trajectory in parallel (UNet2)
            noise2 = torch.randn_like(y2_seq)
            noisy_y2 = self.scheduler.add_noise(y2_seq, noise2, timesteps)
            # PASS BOTH CONDITIONS TO UNET
            noise_pred2 = self.unet2(noisy_y2, timesteps, global_cond=p_embedded, context_seq=latent_seq) 
            
            noise_pred = torch.cat((noise_pred1, noise_pred2), dim=-1)
            noise_truth = torch.cat((noise1, noise2), dim=-1)

        return noise_pred, noise_truth, L_F, L_I, extra_pass
    
    @torch.no_grad()
    def sample(self, cond_seq, params, mask_indices=None, source_dim='y1', target_dim='y2', time_len=200):
        """
        INFERENCE PASS: Zero-Shot Task Inversion.
        Generates the trajectory iteratively from pure noise.

        cond_seq: (batch_size, time_len, d_y1)
        params: (batch_size, 1, d_param)
        """
        self.eval()
        device = cond_seq.device
        batch_size = cond_seq.shape[0]
        
        p_embedded = self.param_embedder(params).squeeze(1)
        
        encoder = self.encoder1 if source_dim == 'y1' else self.encoder2
        latent_seq = encoder(cond_seq, mask_indices) # Full Sequence Context
        
        unet = self.unet2 if target_dim == 'y2' else self.unet1
        dim_y = self.d_y2 if target_dim == 'y2' else self.d_y1
        
        # Start with pure Gaussian noise X_T
        seq = torch.randn((batch_size, time_len, dim_y), device=device)
        
        # Standard DDPM Reverse Loop
        for k in reversed(range(self.num_diffusion_steps)):
            timesteps = torch.full((batch_size,), k, device=device, dtype=torch.long)
            
            # Predict noise using Cross Attention
            noise_pred = unet(seq, timesteps, global_cond=p_embedded, context_seq=latent_seq)
            
            # DDPM Step Math
            alpha_k = (1.0 - self.scheduler.betas[k]).to(device)
            alpha_cum_k = self.scheduler.alphas_cumprod[k].to(device)
            beta_k = self.scheduler.betas[k].to(device)
            
            # Remove the predicted noise from the sequence
            # Equation: y_{k-1} = 1/sqrt(alpha) * (y_k - (1-alpha)/sqrt(1-alpha_cum) * noise_pred)
            seq = (1.0 / torch.sqrt(alpha_k)) * (seq - ((1.0 - alpha_k) / torch.sqrt(1.0 - alpha_cum_k)) * noise_pred)
            
            # Add stochastic variance back in, unless we are at the final step (k=0)
            if k > 0:
                noise = torch.randn_like(seq)
                sigma_k = torch.sqrt(beta_k) 
                seq = seq + sigma_k * noise
                
        self.train()
        return seq


def loss(noise_pred, noise_truth, L_F, L_I, extra_pass, d_y1, lambda1=1.0, lambda2=0.1):
    # Denoising Loss
    if extra_pass:
        # We only care about the forward trajectory, so we slice the first d_y1 dimensions
        # This prevents double-weighting the MSE on the duplicated y1 tensors
        pred1 = noise_pred[..., :d_y1]
        truth1 = noise_truth[..., :d_y1]
        mse_loss = F.mse_loss(pred1, truth1)
    else:
        # Calculate MSE across both the forward and inverse trajectories simultaneously
        mse_loss = F.mse_loss(noise_pred, noise_truth)
    
    # Latent Alignment Loss
    latent_alignment_loss = torch.mean((L_F - L_I) ** 2)

    return (lambda1 * mse_loss) + (lambda2 * latent_alignment_loss)
