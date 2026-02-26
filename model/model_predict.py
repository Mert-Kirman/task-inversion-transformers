import torch
import numpy as np

def predict_inverse(model, time_len, context, condition_points, d_x, d_y1, d_y2, device='cpu'):
    
    num_conditions = len(condition_points)
    obs = torch.zeros((1, num_conditions, d_x + d_y1), device=device)
    params = context.reshape(1, 1, -1).to(device)
    mask = torch.eye(num_conditions, device=device).repeat(1,1,1)
    mask = [mask, mask]
    for condition in condition_points:
        x_obs = torch.tensor(condition[0], device=device).reshape(1,1)
        y_obs = condition[1].reshape(1,d_y1).to(device)
        obs[0][condition_points.index(condition)] = torch.cat((x_obs, y_obs), dim=-1) 
    
    means = torch.zeros(0, device=device)
    stds = torch.zeros(0, device=device)

    with torch.no_grad():
        T = torch.linspace(0,1,time_len, device=device).reshape(1, time_len, -1)
        obs = torch.cat((obs,obs), dim=-1)
        output, _, __, ___ = model(obs, params, mask, T, False, p=1)
        mean1, std1, mean2, std2 = output.chunk(4, dim=-1)
        std2 = np.log(1+np.exp(std2.cpu()))
        std1 = np.log(1+np.exp(std1.cpu()))
        means = mean2
        stds = torch.from_numpy(std2).to(device)
    
    return means[0], stds[0]

def predict_forward_forward(model, time_len, context, condition_points, d_x, d_y1, d_y2, device='cpu'):
    
    num_conditions = len(condition_points)
    obs = torch.zeros((1, num_conditions, d_x + d_y1), device=device)
    params = context.reshape(1, 1, -1).to(device)
    mask = torch.eye(num_conditions, device=device).repeat(1,1,1)
    mask = [mask, mask]
    for condition in condition_points:
        x_obs = torch.tensor(condition[0], device=device).reshape(1,1)
        y_obs = condition[1].reshape(1, d_y1).to(device)
        obs[0][condition_points.index(condition)] = torch.cat((x_obs, y_obs), dim=-1) 
    
    means = torch.zeros(0, device=device)
    stds = torch.zeros(0, device=device)

    with torch.no_grad():
        T = torch.linspace(0,1,time_len, device=device).reshape(1, time_len, -1)
        obs = torch.cat((obs,obs), dim=-1)
        output, _, __, ___ = model(obs, params, mask, T, False, p=1)
        mean1, std1, mean2, std2 = output.chunk(4, dim=-1)
        std2 = np.log(1+np.exp(std2.cpu()))
        std1 = np.log(1+np.exp(std1.cpu()))
        means = mean1
        stds = torch.from_numpy(std1).to(device)
    
    return means[0], stds[0]


def predict_forward(model, time_len, context, condition_points, d_x, d_y1, d_y2, device='cpu'):
    
    num_conditions = len(condition_points)
    obs = torch.zeros((1, num_conditions, d_x + d_y1), device=device)
    params = context.reshape(1, 1, -1).to(device)
    mask = torch.eye(num_conditions, device=device).repeat(1,1,1)
    mask = [mask, mask]
    for condition in condition_points:
        x_obs = torch.tensor(condition[0], device=device).reshape(1,1)
        y_obs = condition[1].reshape(1, d_y2).to(device)
        obs[0][condition_points.index(condition)] = torch.cat((x_obs, y_obs), dim=-1) 

    means = torch.zeros(0, device=device)
    stds = torch.zeros(0, device=device)

    with torch.no_grad():
        T = torch.linspace(0,1,time_len, device=device).reshape(1, time_len, -1)
        obs = torch.cat((obs,obs), dim=-1)
        output, _, __, ___ = model(obs, params, mask, T, False, p=2)
        mean1, std1, mean2, std2 = output.chunk(4, dim=-1)
        std2 = np.log(1+np.exp(std2.cpu()))
        std1 = np.log(1+np.exp(std1.cpu()))
        means = mean1
        stds = torch.from_numpy(std1).to(device)
    
    return means[0], stds[0]

def predict_inverse_inverse(model, time_len, context, condition_points, d_x, d_y1, d_y2, device='cpu'):
    
    num_conditions = len(condition_points)
    obs = torch.zeros((1, num_conditions, d_x + d_y1), device=device)
    params = context.reshape(1, 1, -1).to(device)
    mask = torch.eye(num_conditions, device=device).repeat(1,1,1)
    mask = [mask, mask]
    for condition in condition_points:
        x_obs = torch.tensor(condition[0], device=device).reshape(1,1)
        y_obs = condition[1].reshape(1, d_y2).to(device)
        obs[0][condition_points.index(condition)] = torch.cat((x_obs, y_obs), dim=-1) 

    means = torch.zeros(0, device=device)
    stds = torch.zeros(0, device=device)

    with torch.no_grad():
        T = torch.linspace(0,1,time_len, device=device).reshape(1, time_len, -1)
        obs = torch.cat((obs,obs), dim=-1)
        output, _, __, ____ = model(obs, params, mask, T, False, p=2)
        mean1, std1, mean2, std2 = output.chunk(4, dim=-1)
        std2 = np.log(1+np.exp(std2.cpu()))
        std1 = np.log(1+np.exp(std1.cpu()))
        means = mean2
        stds = torch.from_numpy(std2).to(device)
    
    return means[0], stds[0]