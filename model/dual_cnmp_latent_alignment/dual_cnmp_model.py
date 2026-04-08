import torch
import torch.nn as nn
import model.loss_utils as loss_utils
import torch.nn.functional as F


class DualCNMP(nn.Module):
    def __init__(self, d_x, d_y1, d_y2, d_param, dropout_p=[0.0, 0.0]):
        super(DualCNMP, self).__init__()

        self.d_x = d_x
        self.d_y1 = d_y1
        self.d_y2 = d_y2
        self.param_dim = d_param
        self.embedding_dim = 16

        # --- Unpack dropout probabilities ---
        p_enc = dropout_p[0] # Probability for Encoder
        p_dec = dropout_p[1] # Probability for Decoder

        # --- Parameter Embedder ---
        self.param_embedder = nn.Sequential(
            nn.Linear(d_param, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, self.embedding_dim)
        )

        # --- Encoders with BatchNorm and Dropout ---
        # Linear -> BatchNorm -> Activation -> Dropout
        self.encoder1 = nn.Sequential(
            nn.Linear(d_x + d_y1, 64), nn.LayerNorm(64), nn.ReLU(), nn.Dropout(p_enc),
            nn.Linear(64, 64), nn.LayerNorm(64), nn.ReLU(), nn.Dropout(p_enc),
            nn.Linear(64, 128), nn.LayerNorm(128), nn.ReLU(), nn.Dropout(p_enc),
            nn.Linear(128, 256)
        )

        self.encoder2 = nn.Sequential(
            nn.Linear(d_x + d_y2, 64), nn.LayerNorm(64), nn.ReLU(), nn.Dropout(p_enc),
            nn.Linear(64, 64), nn.LayerNorm(64), nn.ReLU(), nn.Dropout(p_enc),
            nn.Linear(64, 128), nn.LayerNorm(128), nn.ReLU(), nn.Dropout(p_enc),
            nn.Linear(128, 256)
        )

        # --- Decoders with BatchNorm and Dropout ---
        self.decoder1 = nn.Sequential(
            nn.Linear(d_x + 256 + self.embedding_dim, 256), nn.LayerNorm(256), nn.ReLU(), nn.Dropout(p_dec), 
            nn.Linear(256, 256), nn.LayerNorm(256), nn.ReLU(), nn.Dropout(p_dec),
            nn.Linear(256, 128), nn.LayerNorm(128), nn.ReLU(), nn.Dropout(p_dec),
            nn.Linear(128, 64), nn.LayerNorm(64), nn.ReLU(), nn.Dropout(p_dec),
            nn.Linear(64, (d_y1)*2)
        )

        self.decoder2 = nn.Sequential(
            nn.Linear(d_x + 256 + self.embedding_dim, 256), nn.LayerNorm(256), nn.ReLU(), nn.Dropout(p_dec), 
            nn.Linear(256, 256), nn.LayerNorm(256), nn.ReLU(), nn.Dropout(p_dec),
            nn.Linear(256, 128), nn.LayerNorm(128), nn.ReLU(), nn.Dropout(p_dec),
            nn.Linear(128, 64), nn.LayerNorm(64), nn.ReLU(), nn.Dropout(p_dec),
            nn.Linear(64, (d_y2)*2)
        )

    def forward(self, obs, params, mask, x_tar, extra_pass, p=0):
        # obs: (batch_size, max_obs_num, 2*d_x + d_y1 + d_y2) 
        # mask: (batch_size, max_obs_num, 1)
        # x_tar: (batch_size, num_tar, d_x)
        # params: (batch_size, 1, d_param)

        mask_forward, mask_inverse = mask[0], mask[1] # (batch_size, max_obs_num, max_obs_num)

        obs_f = obs[:, :, :self.d_x+self.d_y1]  # (batch_size, max_obs_num, d_x + d_y1)
        obs_i = obs[:, :, self.d_x+self.d_y1:2*self.d_x+self.d_y1+self.d_y2]  # (batch_size, max_obs_num, d_x + d_y2)

        # Embed the parameters
        # params is (batch, 1, d_param).
        p_input = params
        
        # Pass through the MLP embedder
        p_embedded = self.param_embedder(p_input) # Output: (batch_size, num_tar, embedding_dim) (12, 1, 16)
        
        # Expand to match the number of target points
        # (batch, num_tar, embedding_dim)
        p_expanded = p_embedded.expand(-1, x_tar.shape[1], -1)

        r1 = self.encoder1(obs_f)  # (batch_size, max_obs_num, 256)
        masked_r1 = torch.bmm(mask_forward, r1) # (batch_size, max_obs_num, 256)
        sum_masked_r1 = torch.sum(masked_r1, dim=1) # (batch_size, 256)
        L_F = sum_masked_r1 / (torch.sum(mask_forward, dim=[1,2]).reshape(-1,1) + 1e-10) # (batch_size, 256)
        L_F = L_F.unsqueeze(1).expand(-1, x_tar.shape[1], -1) # (batch_size, num_tar, 128)

        if not extra_pass:
            r2 = self.encoder2(obs_i)  # (batch_size, max_obs_num, 256)
            masked_r2 = torch.bmm(mask_inverse, r2) # (batch_size, max_obs_num, 256)
            sum_masked_r2 = torch.sum(masked_r2, dim=1) # (batch_size, 256)
            L_I = sum_masked_r2 / (torch.sum(mask_inverse, dim=[1,2]).reshape(-1,1) + 1e-10) # (batch_size, 256)
            L_I = L_I.unsqueeze(1).expand(-1, x_tar.shape[1], -1) # (batch_size, num_tar, 256)

        # Get device from input tensor
        device = obs.device
        latent = torch.zeros(0, device=device)
        if p == 0:
            if not extra_pass:
                # --- Discrete Routing ---
                # 50% chance to use Forward context, 50% chance to use Inverse context
                if torch.rand(1, device=device).item() < 0.5:
                    latent = L_F
                else:
                    latent = L_I
            else:
                latent = L_F
        elif p == 1:
            latent = L_F # (1, num_tar, 256) , used for validation pass
        elif p == 2:
            latent = L_I


        # Concatenate with the EMBEDDED parameter
        # latent is (batch, num_tar, 128)
        # p_expanded is (batch, num_tar, 16)
        latent_with_par = torch.cat((latent, p_expanded), dim=-1)  # (batch_size, num_tar, 128 + 16)
        
        concat = torch.cat((latent_with_par, x_tar), dim=-1)  # (batch_size, num_tar, 128 + 16 + d_x)
        output1 = self.decoder1(concat)  # (batch_size, num_tar, 2*d_y1)

        if extra_pass:
            return torch.cat((output1, output1), dim=-1), L_F, L_F, extra_pass

        output2 = self.decoder2(concat)  # (batch_size, num_tar, 2*d_y2)
        # (batch_size, num_tar, 2*d_y1 + 2*d_y2)
        return torch.cat((output1, output2), dim=-1), L_F, L_I, extra_pass
    
def get_training_sample(extra_pass, valid_inverses, validation_indices, test_indices, demo_data, 
                        OBS_MAX, d_N, d_x, d_y1, d_y2, d_param, time_len, batch_size=1, device='cpu'):

    X1, X2, Y1, Y2, C = demo_data
    
    traj_multinom = torch.ones(d_N, device=device) # multinomial distribution for trajectories

    for i in range(d_N):
       if i in validation_indices or i in test_indices:
           traj_multinom[i] = 0

    if not extra_pass:
        for i in range(len(traj_multinom)):
            if not valid_inverses[i]:
                traj_multinom[i] = 0
    
    traj_indices = torch.multinomial(traj_multinom, batch_size, replacement=False) # random indices of trajectories

    obs_num_list = torch.randint(0, OBS_MAX, (2*batch_size,), device=device) + 1  # random number of obs. points
    max_obs_num = OBS_MAX
    observations = torch.zeros((batch_size, max_obs_num, 2*d_x + d_y1 + d_y2), device=device)
    mask_forward = torch.zeros((batch_size, max_obs_num, max_obs_num), device=device)
    mask_inverse = torch.zeros((batch_size, max_obs_num, max_obs_num), device=device)

    params = torch.zeros((batch_size, 1, d_param), device=device)
    target_X = torch.zeros((batch_size, 1, d_x), device=device)
    target_Y1 = torch.zeros((batch_size, 1, d_y1), device=device)
    target_Y2 = torch.zeros((batch_size, 1, d_y2), device=device)

    T = torch.ones(time_len, device=device)
    for i in range(batch_size):
        traj_index = int(traj_indices[i])
        obs_num_f = int(obs_num_list[i])
        obs_num_i = int(obs_num_list[batch_size + i])

        params[i] = C[traj_index]
                      
        obs_indices_f = torch.multinomial(T, obs_num_f, replacement=False)
        obs_indices_i = torch.multinomial(T, obs_num_i, replacement=False)

        for j in range(obs_num_f):
            observations[i][j][:d_x] = X1[0][obs_indices_f[j]]
            observations[i][j][d_x:d_x+d_y1] = Y1[traj_index][obs_indices_f[j]]
            mask_forward[i][j][j] = 1

        for j in range(obs_num_i):
            if valid_inverses[traj_index]:
                observations[i][j][d_x + d_y1:2*d_x + d_y1] = X2[0][obs_indices_i[j]]
                observations[i][j][2*d_x + d_y1:] = Y2[traj_index][obs_indices_i[j]]
                mask_inverse[i][j][j] = 1
        
        target_index = torch.multinomial(T, 1)
        target_X[i] = X1[0][target_index]
        target_Y1[i] = Y1[traj_index][target_index]
        target_Y2[i] = Y2[traj_index][target_index]
        
    return observations, params, [mask_forward, mask_inverse], target_X, target_Y1, target_Y2, extra_pass
    
def loss(output, target_f, target_i, d_y1, d_y2, d_param, L_F, L_I, extra_pass):
    log_prob = loss_utils.log_prob_loss(output, target_f, target_i, d_y1, d_y2, d_param, extra_pass) # scalar
    
    # Raw Latent Alignment Loss (Preserves Magnitude)
    latent_alignment_loss = torch.mean((L_F - L_I) ** 2)

    lambda1 = 1.0
    lambda2 = 0.01

    return lambda1 * log_prob + lambda2 * latent_alignment_loss


