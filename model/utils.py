import torch
import torch.distributions as D
import matplotlib.pyplot as plt
import numpy as np
import math
import os
import random

def seed_everything(seed=42):
    """
    Locks down all sources of randomness for reproducible results.
    """
    # Set Python's built-in random module
    random.seed(seed)
    
    # Set environment variable for Python hash seed (for dict/set ordering)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # Set NumPy's random seed
    np.random.seed(seed)
    
    # Set PyTorch's random seed
    torch.manual_seed(seed)
    
    # Set CUDA/GPU random seeds (Crucial for your RTX 2060)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        # Force cuDNN to be deterministic
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print(f"Global seed set to: {seed}")

def gaussian_with_offset(param, noise = 0):
    def dist(x, param, noise = 0):
        f = (math.exp(-x**2/(2.*param[0]**2))/(math.sqrt(2*math.pi)*param[0]))+param[1]
        return f+(noise*(np.random.rand()-0.5)/100.)
    return dist

def x_sinx(param, noise = 0):
    def dist(x, noise = 0):
        f =  x  + param[0] * math.sin(param[1] * x) #+ param[1]
        return f+(noise*(np.random.rand()-0.5)/100.)
    return dist

def sinx(frequency, amplitude, phase):
    def dist(x):
        return amplitude * math.sin(2 * torch.pi * frequency * x + phase) # + torch.randn(1) * 0.05
    return dist

def linear(min, max):
    def dist(x):
        return max * x + min * (1-x)
    return dist

def generate_demonstrations(num_demo, time_len = 200, params = None, plot_title = None):

    x = torch.linspace(0, 1, time_len)
    times = torch.zeros((2*(num_demo), time_len, 1))
    times[:] = x.reshape((1, time_len, 1))

    num_paired = int(num_demo * 0.5)
    num_extra = int(num_demo * 0.5)

    values = torch.zeros((2*num_paired, time_len, 1))
    extra_values = torch.zeros((2*num_extra, time_len, 1))
    
    context_params_paired = torch.zeros((num_paired, 1))
    context_params_extra = torch.ones((num_extra, 1))

    f_frequencies = [1, 1.5, 2]  # Example frequencies
    amplitudes = torch.linspace(0.1,0.5,num_paired)
    extra_amplitudes = torch.linspace(1.1,1.5,num_extra)

    phases = [0]  # Example phases
        
    for d in range(num_paired):
        f_dist1 = sinx(f_frequencies[0], amplitudes[d % len(amplitudes)], phases[d % len(phases)])

        for i in range(time_len):
            values[d, i] = f_dist1(x[i]*0.5)     

        for i in range(time_len):
           values[d+num_paired, i] = -1*(f_dist1(x[i]*0.5))

        context_params_paired[d] = amplitudes[d % len(amplitudes)]

        # Plot demonstrations
        plt.plot(times[d], values[d], color="black", alpha=0.5)
        plt.plot(times[d], values[d+num_paired], color="black", alpha=0.5)

    for d in range(num_extra):
        f_dist2 = sinx(f_frequencies[0], extra_amplitudes[d % len(extra_amplitudes)], phases[d % len(phases)])

        for i in range(time_len):
            extra_values[d, i] = f_dist2(x[i]*0.5)        

        for i in range(time_len):
           extra_values[d+num_extra, i] = -1*(f_dist2(x[i]*0.5))

        context_params_extra[d] = extra_amplitudes[d % len(extra_amplitudes)]

        # Plot demonstrations
        plt.plot(times[d], extra_values[d], color="black", alpha=0.5)
        plt.plot(times[d], extra_values[d+num_extra], color="black", alpha=0.5)

    plt.title(plot_title + ' Demonstrations')
    plt.ylabel('Y')
    plt.xlabel('time (t)')
    plt.show()

    forward = values[:num_paired]
    inverse = values[num_paired:]
    extra_forward = extra_values[:num_extra]
    extra_inverse = extra_values[num_extra:]

    Y1 = forward.reshape(num_paired, time_len, 1)
    Y2 = inverse.reshape(num_paired, time_len, 1)
    X1 = times[:1]
    X2 = times[:1]

    Y1_extra = extra_forward.reshape(num_extra, time_len, 1)
    Y2_extra = extra_inverse.reshape(num_extra, time_len, 1)

    plt.title("Forward Trajectories")
    for i in range(num_paired):
        plt.plot(times[i], forward[i], color = "blue")

    for i in range(num_extra):
        plt.plot(times[i], extra_forward[i], color = "blue")
    
    plt.grid(alpha=0.3)
    plt.show()

    plt.title("Inverse Trajectories")
    for i in range(num_paired):
        plt.plot(times[i], inverse[i], color = "red")

    for i in range(num_extra):
        plt.plot(times[i], extra_inverse[i], color = "red")
    
    plt.grid(alpha=0.3)
    plt.show()

    return X1, X2, Y1, Y2, Y1_extra, Y2_extra, context_params_paired, context_params_extra

def generate_demonstrations_multi_modality(num_demo, time_len = 200):
    x = torch.linspace(0, 1, time_len)
    times = torch.zeros((2*(num_demo), time_len, 1))
    times[:] = x.reshape((1, time_len, 1))

    num_modality_1 = int(num_demo * 0.5)
    num_modality_2 = int(num_demo * 0.5)

    values_modality_1 = torch.zeros((2*num_modality_1, time_len, 1))
    values_modality_2 = torch.zeros((2*num_modality_2, time_len, 1))

    f_frequencies = [1, 1.5, 2]  # Example frequencies
    amplitudes = torch.linspace(1.1,1.5,num_modality_1)
    extra_amplitudes = torch.linspace(2.1,2.5,num_modality_2)

    phases = [0]  # Example phases

    for d in range(num_modality_1):
        f_dist1 = sinx(f_frequencies[0], amplitudes[d % len(amplitudes)], phases[d % len(phases)])

        for i in range(time_len):
            values_modality_1[d, i] = f_dist1(x[i]*0.5)     

        for i in range(time_len):
           values_modality_1[d+num_modality_1, i] = -1*(f_dist1(x[i]*0.5))

    for d in range(num_modality_2):
        f_dist2 = sinx(f_frequencies[0], extra_amplitudes[d % len(extra_amplitudes)], phases[d % len(phases)])

        for i in range(time_len):
            values_modality_2[d, i] = f_dist2(x[i]*0.5)        

        for i in range(time_len):
           values_modality_2[d+num_modality_2, i] = -1*(f_dist2(x[i]*0.5))

    forward_modality_1 = values_modality_1[:num_modality_1]
    inverse_modality_1 = values_modality_1[num_modality_1:]
    
    forward_modality_2 = values_modality_2[:num_modality_2]
    inverse_modality_2 = values_modality_2[num_modality_2:]

    Y1 = forward_modality_1.reshape(num_modality_1, time_len, 1)
    Y1_inverse = inverse_modality_1.reshape(num_modality_1, time_len, 1)
    X1 = times[:1]
    X2 = times[:1]

    Y2 = forward_modality_2.reshape(num_modality_2, time_len, 1)
    Y2_inverse = inverse_modality_2.reshape(num_modality_2, time_len, 1)

    return X1, X2, Y1, Y1_inverse, Y2, Y2_inverse

def validate_model(means, stds, idx, demo_data, time_len, condition_points, epoch_count, d_y1, d_y2, forward, 
                   error_type='mse', plot=False):

    X1, X2, Y1, Y2 = demo_data[:4]

    if forward:
        target_demo = Y1[idx,:,:]
    else:
        target_demo = Y2[idx,:,:]

    if error_type == 'mse':
        error = torch.mean(torch.nn.functional.mse_loss(means[:,:], target_demo[:,:]))
    elif error_type == 'log_prob':
        # Calculate the log probability of the target under the predicted distribution
        dist = D.Normal(means, stds)
        error = -1 * torch.mean(dist.log_prob(target_demo))

    if plot:
        plot_test(idx, Y1, Y2, means, stds, time_len, condition_points, epoch_count)
    
    return error

def plot_test(idx, Y1, Y2, means, stds, time_len, condition_points, epoch_count):
    d_N = Y1.shape[0]
    num_dim = 8 #Y1.shape[2]
    T_forward = np.linspace(0,1,Y1.shape[1])
    T_inverse = np.linspace(0,1,Y2.shape[1])

    ## plot forward and inverse trajectories for each dimension, add subplots, 4 above, 3 below

    plt.figure(figsize=(15, 15))
    ax1 = plt.subplot(4, 3, 1)
    ax2 = plt.subplot(4, 3, 2)
    ax3 = plt.subplot(4, 3, 3)
    ax4 = plt.subplot(4, 3, 4)
    ax5 = plt.subplot(4, 3, 5)
    ax6 = plt.subplot(4, 3, 6)
    ax7 = plt.subplot(4, 3, 7)
    ax8 = plt.subplot(4, 3, 8)
    ax = [[ax1, ax2, ax3], [ax4, ax5, ax6], [ax7, ax8]]
    dim_plot_dict = {0: (0,0), 1: (0,1), 2: (0,2), 3: (1,0), 4: (1,1), 5: (1,2), 6: (2,0), 7: (2,1)}

    for dim in range(num_dim):
        plot_idx = dim_plot_dict[dim]
        ax[plot_idx[0]][plot_idx[1]].set_title(f"Joint {dim}")
        for j in range(d_N):
            if j == idx:
                ax[plot_idx[0]][plot_idx[1]].plot(T_forward, Y1[j,:,dim], color='blue', label='Forward', alpha=0.5)
                ax[plot_idx[0]][plot_idx[1]].plot(T_inverse, Y2[j,:,dim], color='red', label='Expected (Inverse)', alpha=0.5)
                continue
            ax[plot_idx[0]][plot_idx[1]].plot(T_forward, Y1[j,:,dim], color='black', alpha=0.1)
            ax[plot_idx[0]][plot_idx[1]].plot(T_inverse, Y2[j,:,dim], color='black', alpha=0.1)

        ax[plot_idx[0]][plot_idx[1]].plot(T_forward, means[:,dim].detach().numpy(), color='green', label='Prediction')
        ax[plot_idx[0]][plot_idx[1]].errorbar(T_forward, means[:,dim].detach().numpy(), yerr=stds[:,dim].detach().numpy(), color='black', alpha=0.2)
        
        for i in range(len(condition_points)):
            cd_pt_x = condition_points[i][0]
            cd_pt_y = condition_points[i][1][0][dim]
            if i == 0:
                pass
                ax[plot_idx[0]][plot_idx[1]].scatter(cd_pt_x, cd_pt_y, color='black', label='Observations')
                continue
            ax[plot_idx[0]][plot_idx[1]].scatter(cd_pt_x, cd_pt_y, color='black')

    plt.suptitle(f"Prediction for epoch {epoch_count}")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.show()

def plot_results(best_mean, best_std, Y1, Y2, idx, condition_points, errors, losses, time_len, d_N, plot_errors=True, test_dist=None):
    
    num_dim = 7

    T = np.linspace(0,1,time_len)

    plt.figure(figsize=(15, 15))
    ax1 = plt.subplot(4, 3, 1)
    ax2 = plt.subplot(4, 3, 2)
    ax3 = plt.subplot(4, 3, 3)
    ax4 = plt.subplot(4, 3, 4)
    ax5 = plt.subplot(4, 3, 5)
    ax6 = plt.subplot(4, 3, 6)
    ax7 = plt.subplot(4, 3, 7)
    ax = [[ax1, ax2, ax3], [ax4, ax5, ax6], [ax7]]
    dim_plot_dict = {0: (0,0), 1: (0,1), 2: (0,2), 3: (1,0), 4: (1,1), 5: (1,2), 6: (2,0), 7: (2,1)}

    for dim in range(num_dim):
        plot_idx = dim_plot_dict[dim]
        ax[plot_idx[0]][plot_idx[1]].set_title(f"Joint {dim}", fontsize=20)
        for j in range(d_N):
            if j == 0: 
                #ax[plot_idx[0]][plot_idx[1]].plot(T, Y1[j,:,dim], color='green', alpha=0.1, label='Forward Trajectories (Green)')
                ax[plot_idx[0]][plot_idx[1]].plot(T, Y2[j,:,dim], color='blue', alpha=0.1, label='Inverse Trajectories')
                if j == idx:
                    ax[plot_idx[0]][plot_idx[1]].plot(T, Y2[j,:,dim], color='blue', alpha=0.1, label='Ground Truth')
                continue
            if j == idx:
                ax[plot_idx[0]][plot_idx[1]].plot(T, Y2[j,:,dim], color='blue', alpha=0.1)
            #ax[plot_idx[0]][plot_idx[1]].plot(T, Y1[j,:,dim], color='green', alpha=0.1)
            ax[plot_idx[0]][plot_idx[1]].plot(T, Y2[j,:,dim], color='blue', alpha=0.1)

        ax[plot_idx[0]][plot_idx[1]].plot(T, best_mean[:,dim].detach().numpy(), color='black', label='Prediction')
        ax[plot_idx[0]][plot_idx[1]].errorbar(T, best_mean[:,dim].detach().numpy(), yerr=best_std[:,dim].detach().numpy(), color='black', alpha=0.2)
        
        for i in range(len(condition_points)):
            cd_pt_x = condition_points[i][0]
            cd_pt_y = condition_points[i][1][dim]
            if i == 0:
                pass
                ax[plot_idx[0]][plot_idx[1]].scatter(cd_pt_x, cd_pt_y, color='black', label='Observations')
                continue
            ax[plot_idx[0]][plot_idx[1]].scatter(cd_pt_x, cd_pt_y, color='black')

    for ax_ in ax[0]:
        ax_.grid(alpha=0.3)
        
    
    #plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    #plt.savefig(f"../figs/Results.png")
    plt.show()

############################################################################################################

"""
def plot_latent_space(model, observations_f, observations_i):
    with torch.no_grad():
        l_f = model.encoder1(observations_f) # condition points is (n, d_x + d_y), l is (n, 128)
        l_i = model.encoder2(observations_i)
    l_f = np.array(l_f)
    l_i = np.array(l_i)
    l = np.concat((l_f,l_i), axis=0)

    
    for i in range(len(l_f)):
        plt.scatter(i,np.linalg.norm(l_f[i]-l_i[i]), color='blue')
    plt.show()

    print(l.shape)
    l = l.squeeze(1)

    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(l)
    return pca_result


def tsne_analysis(model, observations_f, observations_i):
    with torch.no_grad():
        l_f = model.encoder1(observations_f) # condition points is (n, d_x + d_y), l is (n, 128)
        l_i = model.encoder2(observations_i)
    l_f = np.array(l_f)
    l_i = np.array(l_i)
    l = np.concat((l_f,l_i), axis=0)

    l = l.squeeze(1)

    tsne = TSNE(n_components=2, random_state=41)
    tsne_result = tsne.fit_transform(l)
    return tsne_result
"""