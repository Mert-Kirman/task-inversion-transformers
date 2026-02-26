import numpy as np
import model.utils as utils
import model.model_predict as model_predict
import torch

def val(model, validation_indices, epoch_count, demo_data, d_x, d_y1, d_y2, d_param, time_len=200):

    model.eval()
    [_, __, Y1, Y2, C] = demo_data
    error = 0
    plot_id = np.random.randint(0, len(validation_indices))
    for validation_idx in validation_indices:
        time = np.linspace(0, 1, time_len)
        # permute time
        idx = np.random.permutation(time_len)
        
        #idx = idx[:3]
        idx = [0]

        time = [time[i] for i in idx]
        f_condition_points = [[t, Y1[validation_idx, i:i+1]] for t,i in zip(time, idx)]
        i_condition_points = [[t, Y2[validation_idx, i:i+1]] for t,i in zip(time, idx)]

        context = C[validation_idx]

        fi_means, fi_stds = model_predict.predict_inverse(model, time_len, context,
                                                          f_condition_points, d_x, d_y1, d_y2)
        ff_means, ff_stds = model_predict.predict_forward_forward(model, time_len, context,
                                                                   f_condition_points, d_x, d_y1, d_y2)
        ii_means, ii_stds = model_predict.predict_inverse_inverse(model, time_len, context, 
                                                                  i_condition_points, d_x, d_y1, d_y2)
        if_means, if_stds = model_predict.predict_forward(model, time_len, context,
                                                          i_condition_points, d_x, d_y1, d_y2)
        
        if epoch_count % 200_000 == 0 and validation_idx == validation_indices[plot_id]:
            error += utils.validate_model(fi_means, fi_stds, validation_idx, demo_data, time_len, f_condition_points, epoch_count,d_y1, d_y2, forward=False, plot=True)
            error += utils.validate_model(ff_means, ff_stds, validation_idx, demo_data, time_len, f_condition_points, epoch_count, d_y1, d_y2, forward=True, plot=True)
        else:
            error += utils.validate_model(ff_means, ff_stds, validation_idx, demo_data, time_len, f_condition_points, epoch_count, d_y1, d_y2,forward=True, plot=False)
            error += utils.validate_model(fi_means, fi_stds, validation_idx, demo_data, time_len, f_condition_points, epoch_count,d_y1, d_y2, forward=False, plot=False)

        error += utils.validate_model(ii_means, ii_stds, validation_idx, demo_data, time_len, i_condition_points, epoch_count,d_y1, d_y2, forward=False, plot=False)
        error += utils.validate_model(if_means, if_stds, validation_idx, demo_data, time_len, i_condition_points, epoch_count,d_y1, d_y2, forward=True, plot=False)
    
    model.train()
    return error / (len(validation_indices) * 4)  # average error over all validation indices and both forward and inverse predictions


import numpy as np
import model.utils as utils
import model.model_predict as model_predict

def val_only_extra(model, validation_indices, epoch_count, demo_data, d_x, d_y1, d_y2, 
                   time_len=200, plot_freq=200_000, device='cpu'):

    model.eval()

    [_, __, Y1, ___, C] = demo_data
    error = 0
    plot_id = np.random.randint(0, len(validation_indices))
    for validation_idx in validation_indices:

        time = np.linspace(0, 1, time_len)
        # permute time
        idx = np.random.permutation(time_len)
        
        idx = idx[:3]
        # idx = [0]

        time = [time[i] for i in idx]
        f_condition_points = [[t, Y1[validation_idx, i:i+1]] for t,i in zip(time, idx)]
        
        context = C[validation_idx]

        ff_means, ff_stds = model_predict.predict_forward_forward(model, time_len, context,
                                                                   f_condition_points, d_x, d_y1, d_y2, device=device)
        
        if epoch_count % plot_freq == 0 and validation_idx == validation_indices[plot_id]:
            error += utils.validate_model(ff_means, ff_stds, validation_idx, demo_data, time_len, f_condition_points, epoch_count, d_y1, d_y2, forward=True, plot=False)
        else:
            error += utils.validate_model(ff_means, ff_stds, validation_idx, demo_data, time_len, f_condition_points, epoch_count, d_y1, d_y2,forward=True, plot=False)

    model.train()
    
    return error / (len(validation_indices))  # average error over all validation indices and both forward and inverse predictions



def val_features(model, validation_indices, corresponding_indices, demo_data):

    model.eval()
    [_, __, _, _, C] = demo_data
    error = 0
    for i in range(len(validation_indices)):

        validation_idx = validation_indices[i]

        # val features
        val_C = C[validation_idx, 1:]
        # corresponding features
        corresponding_C = C[corresponding_indices[i], 1:]

        val_embedding = model.mlp(val_C)
        corresponding_embedding = model.mlp(corresponding_C)

        error += torch.nn.functional.mse_loss(val_embedding, corresponding_embedding)

    model.train()
    
    return error / (len(validation_indices))  # average error over all validation indices and both forward and inverse predictions



def val_extrapolation(model, validation_indices, corresponding_indices, demo_data,d_x, d_y1, d_y2, time_len=200):

    model.eval()
    [_, __, Y1, Y2, C] = demo_data
    error = 0
    for i in range(len(validation_indices)):

        validation_idx = validation_indices[i]

        # val features
        context = C[validation_idx, :]

        time = np.linspace(0, 1, time_len)
        # permute time
        idx = np.random.permutation(time_len)
        
        #idx = idx[:3]
        idx = [0]

        time = [time[i] for i in idx]
        f_condition_points = [[t, Y1[validation_idx, i:i+1]] for t,i in zip(time, idx)]

        fi_means, fi_stds = model_predict.predict_inverse(model, time_len, context,
                                                                   f_condition_points, d_x, d_y1, d_y2)
        
        error += torch.nn.functional.mse_loss(fi_means, Y2[corresponding_indices[i], :, :])
        
    model.train()
    
    return error / (len(validation_indices))  # average error over all validation indices and both forward and inverse predictions
