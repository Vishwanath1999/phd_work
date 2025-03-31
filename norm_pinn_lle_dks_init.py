# %%
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as cts
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import pandas as pd
from scipy.io import loadmat,savemat
from scipy.interpolate import interp1d
from pyDOE import lhs
from IPython.display import clear_output
from torch.distributions import MultivariateNormal
import json
import copy
import wandb
import io

# %%
data = loadmat('A.mat')['A']
# data.shape

# # %%
# plt.figure(figsize=(10, 3))
# plt.imshow(np.abs(data), cmap='jet', origin='lower', aspect='auto')
# plt.colorbar()
# plt.tight_layout()
# plt.show()

data_ifft = np.fft.fft(data, axis=0)
# plt.figure(figsize=(10, 3))
# plt.imshow(np.abs(data_ifft), cmap='jet', origin='lower', aspect='auto')
# plt.colorbar()
# plt.tight_layout()
# plt.show()
# plt.close()

# %%
# plt.plot(np.real(data[:,1]))
# plt.plot(np.imag(data[:,1]))
# plt.show()

# plt.plot(np.abs(data[:,1]))
# # plt.ylim(0,)
# plt.show()

# plt.plot(np.real(data_ifft[:,3400]))
# plt.plot(np.imag(data_ifft[:,3400]))
# plt.show()

# plt.plot(np.abs(data_ifft[:,3400]))
# plt.show()

# %%
class MRR:
  def __init__(self, params):
    self.N = params['N']
    self.n0 = params['n0']
    self.n2 = params['n2']
    self.FSR = params['FSR']
    self.lambda_0 = params['lambda_0']
    self.Veff = params['Veff']
    self.D2 = params['D2']
    self.Pin = params['Pin']
    self.kappa = 2*np.pi*params['kappa']
    self.eta = params['eta']
    self.dseta_start = params['dseta_start']
    self.dseta_stop = params['dseta_stop']
    self.dseta_step = params['dseta_step']
    self.num_round_trips = 1 #number of roundtrips per tuning step
    self.R = 23e-6
    # self.num_steady_state_rt = 25000
    self.modes = np.arange(-(self.N - 1) / 2, ((self.N - 1) / 2) + 1, 1)


    self.Tr = 1/self.FSR
    self.f0 = cts.c/self.lambda_0
    self.omega_0 = 2*np.pi*self.f0
    self.g = self.g = cts.hbar * self.omega_0**2 * cts.c * self.n2 / (self.n0**2 * self.Veff) # Nonlinear coupling coefficient

    self.f_amp = np.sqrt((8 * self.g * self.eta / self.kappa**2) * (self.Pin / (cts.hbar * self.omega_0)))
    self.d2 = (1/self.kappa)*2*np.pi*self.D2
    self.mu = np.arange(-(self.N - 1) / 2, ((self.N - 1) / 2) + 1, 1) # Mode numbers relative to the pumped mode
    self.dint = (self.d2 / 2) * self.mu**2 # Normalized integrated dispersion

    if self.dseta_start == self.dseta_stop:
      self.seta = self.dseta_start*np.ones((self.dseta_step,1))
    else:
      self.seta = np.arange(self.dseta_start, self.dseta_stop, self.dseta_step)[:,np.newaxis]

  def calc_beta2(self):
    f_center = self.f0
    mode_numbers = np.arange(-(self.N - 1) / 2, ((self.N - 1) / 2) + 1, 1)

    L = 2*np.pi*self.R
    self.freq_array = f_center + mode_numbers * self.FSR

    # # Calculate beta_2 as a function of frequency
    D1 = 2 * np.pi * self.FSR  # D1 in rad/s
    theta_norm = np.sqrt(1/(2*self.d2))  
    self.theta = (np.linspace(-np.pi, np.pi, self.N))
    self.theta = (self.theta)*theta_norm # normalized frequency detuning (- D1*np.linspace(0, len(self.seta)*self.Tr, self.N))
    self.tau = 0.5*self.kappa*np.linspace(0, len(self.seta)*self.Tr, len(self.seta)) # normalized time coordinate tau = kappa*t/2
    

  def gen_tau_t(self, N0=100, Nb=100, Nf=20000, seta_idx=1):

    # initial condition
    t_0 = self.tau.min()
    theta_0 = self.theta
    T0, Theta0 = np.meshgrid(t_0, theta_0)
    T0 = T0.flatten()[:, np.newaxis]
    Theta0 = Theta0.flatten()[:, np.newaxis]
    idx = np.random.choice(len(T0), N0, replace=False)
    idx = np.sort(idx)
    T0 = T0[idx]
    Theta0 = Theta0[idx]

    # u_init = np.sqrt(2*self.g/self.kappa)*np.random.rand(len(Theta0))
    u_init = np.real(data_ifft[idx,seta_idx:seta_idx+1])
    # v_init = np.sqrt(2*self.g/self.kappa)*np.random.rand(len(Theta0))
    v_init = np.imag(data_ifft[idx,seta_idx:seta_idx+1])

    # boundary conditions
    Tb = self.tau
    theta_lb = self.theta.min()
    theta_ub = self.theta.max()

    T_lb, Theta_lb = np.meshgrid(Tb, theta_lb)
    T_ub, Theta_ub = np.meshgrid(Tb, theta_ub)

    T_lb = T_lb.flatten()[:, np.newaxis]
    Theta_lb = Theta_lb.flatten()[:, np.newaxis]
    T_ub = T_ub.flatten()[:, np.newaxis]
    Theta_ub = Theta_ub.flatten()[:, np.newaxis]

    idx = np.random.choice(len(T_lb), Nb, replace=False)
    idx = np.sort(idx)
    T_lb = T_lb[idx]
    Theta_lb = Theta_lb[idx]
    T_ub = T_ub[idx]
    Theta_ub = Theta_ub[idx]

    T,Theta = np.meshgrid(self.tau, self.theta)
    print('T shape:', T.shape)
    print('Theta shape:', Theta.shape)
    T = T.flatten()[:, np.newaxis]
    Theta = Theta.flatten()[:, np.newaxis]
    idx = np.random.choice(len(T), Nf, replace=False)
    idx = np.sort(idx)
    T = T[idx]
    Theta = Theta[idx]

    F =  self.f_amp*np.ones_like(T)#self.f_mat.flatten()[:,np.newaxis][idx]#np.sqrt((8*self.g*self.eta/self.kappa**2)*(self.Pin/(cts.hbar*self.omega_0)))*np.ones_like(T)
    seta_array = self.seta[0]*np.ones_like(T)
    return T0, Theta0, u_init, v_init, self.seta[0]*np.ones_like(T0), T_lb, Theta_lb, T_ub, Theta_ub, T, Theta, seta_array, F

  def gen_tau_t_lhs(self, N0=100, Nb=100, Nf=20000, seta_idx=1):

    t_0 = self.tau.min()
    theta_0 = self.theta

    T0, Theta0 = np.meshgrid(t_0, theta_0)
    T0 = T0.flatten()[:, np.newaxis]
    Theta0 = Theta0.flatten()[:, np.newaxis]

    # Random selection for initial condition
    idx = np.random.choice(len(T0), N0, replace=False)
    idx = np.sort(idx)
    T0 = T0[idx]
    Theta0 = Theta0[idx]

    # u_init = np.real(data_ifft[idx, seta_idx:seta_idx+1])
    # v_init = np.imag(data_ifft[idx, seta_idx:seta_idx+1])

    u_init = np.sqrt(2*self.g/self.kappa)*np.random.rand(len(Theta0))
    v_init = np.sqrt(2*self.g/self.kappa)*np.random.rand(len(Theta0))

    seta_init = self.seta[0]*np.ones_like(T0)

    # boundary conditions
    Tb = self.tau
    theta_lb = self.theta.min()
    theta_ub = self.theta.max()

    T_lb, Theta_lb = np.meshgrid(Tb, theta_lb)
    T_ub, Theta_ub = np.meshgrid(Tb, theta_ub)

    T_lb = T_lb.flatten()[:, np.newaxis]
    Theta_lb = Theta_lb.flatten()[:, np.newaxis]
    T_ub = T_ub.flatten()[:, np.newaxis]
    Theta_ub = Theta_ub.flatten()[:, np.newaxis]

    idx = np.random.choice(len(T_lb), Nb, replace=False)
    idx = np.sort(idx)
    T_lb = T_lb[idx]
    Theta_lb = Theta_lb[idx]
    T_ub = T_ub[idx]
    Theta_ub = Theta_ub[idx]

    # Use LHS for sampling collocation points from the meshgrid
    T, Theta = np.meshgrid(self.tau, self.theta)
    T = T.flatten()[:, np.newaxis]
    Theta = Theta.flatten()[:, np.newaxis]

    # LHS sampling over the meshgrid
    lhs_samples = lhs(2, samples=Nf)  # 2 dimensions for T and Theta

    # Rescale LHS samples to match the range of the meshgrid for T and Theta
    T_sampled = np.min(T) + (np.max(T) - np.min(T)) * lhs_samples[:, 0][:, np.newaxis]
    Theta_sampled = np.min(Theta) + (np.max(Theta) - np.min(Theta)) * lhs_samples[:, 1][:, np.newaxis]

    # F and seta_array for sampled collocation points
    F = self.f_amp * np.ones_like(T_sampled)
    seta_array = self.seta[0]*np.ones_like(T_sampled)

    return T0, Theta0, u_init, v_init, seta_init, T_lb, Theta_lb, T_ub, Theta_ub, T_sampled, Theta_sampled, seta_array, F


# %%
class RandomWeightFactorizationLayer(nn.Module):
  def __init__(self, in_features, out_features, mu=0.5, sigma=0.1):
    super(RandomWeightFactorizationLayer, self).__init__()

    # Step 1: Glorot initialization (Xavier initialization) for V
    V = torch.empty(out_features, in_features)
    nn.init.xavier_normal_(V)

    # Step 2: Initialize s from a normal distribution N(mu, sigma)
    self.V = nn.Parameter(V)
    self.s = nn.Parameter(torch.randn(out_features) * sigma + mu)

    self.bias = nn.Parameter(torch.zeros(out_features))

  def forward(self, x):
    # Apply random weight factorization: W = diag(exp(s)) @ V
    rwf_kernel = torch.diag(torch.exp(self.s)) @ self.V  # Apply the factorization
    return torch.matmul(x, rwf_kernel.t()) + self.bias

# class RandomWeightFactorizationLayer(nn.Module):
#   def __init__(self, in_features, out_features, mu=0.5, sigma=0.1):
#     super(RandomWeightFactorizationLayer, self).__init__()

#     # Step 1: Glorot initialization (Xavier initialization) for V
#     W = torch.empty(in_features, out_features)
#     nn.init.xavier_normal_(W)

#     # Step 2: Initialize s from a normal distribution N(mu, sigma)
#     s = torch.exp(torch.randn(out_features) * sigma + mu)
#     V = W/s
#     self.V = nn.Parameter(V)
#     self.s = nn.Parameter(s)

#     self.bias = nn.Parameter(torch.zeros(out_features))

#   def forward(self, x):
#     # Apply random weight factorization: W = diag(exp(s)) @ V
#     rwf_kernel = self.V * self.s  # Apply the factorization
#     return torch.matmul(x, rwf_kernel) + self.bias

class FourierFeatureEmbedding(nn.Module):
  def __init__(self, mapping_size, scales=[5.0], P_x=1, P_t=1):
    super(FourierFeatureEmbedding, self).__init__()
    # clip scale between 1,10
    # scale = max(1, min(10, scale))

    # Detect the available device (GPU if available, otherwise CPU)
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    self.P_x = P_x
    self.P_t = P_t

    scale_1 = scales[0]*np.ones(1)
    # Random matrix for projecting inputs to Fourier feature space, placed on the correct device
    # Instead of directly creating tensors, register them as buffers
    self.register_buffer('B_t', torch.cat([scale * torch.randn((1, mapping_size), device=self.device) for scale in scale_1], dim=1))
    self.register_buffer('B_x', torch.cat([scale * torch.randn((1, mapping_size), device=self.device) for scale in scales], dim=1))

  def forward(self, x):
    # Ensure input is on the same device as the random matrix
    x = x.to(self.device)

    t_proj = (2 * torch.pi * x[:, 0:1] / self.P_t) @ self.B_t
    x_proj = (2 * torch.pi * x[:, 1:2] / self.P_x) @ self.B_x

    # Project input using Fourier features
    # f_proj = torch.cat([torch.sin(t_proj), torch.cos(t_proj), torch.sin(x_proj), torch.cos(x_proj)], dim=1)

    return torch.cat([torch.cos(t_proj), torch.sin(t_proj)],dim=1), torch.cat([torch.cos(x_proj), torch.sin(x_proj)], dim=1)

class PINN(nn.Module):
  def __init__(self, layers, mapping_size=64, scale=[10.0], use_fourier_features=True, use_rwf=False, P_x=1, P_t=1, mu=1.0, sigma=0.1):
    super(PINN, self).__init__()

    input_dim = layers[0]
    layers = layers[1:]
    self.use_rwf = use_rwf
    self.use_fourier_features = use_fourier_features
    self.mapping_size = mapping_size
    self.scale = scale

    # Fourier Feature Embedding
    if use_fourier_features:
      self.fourier_embedding = FourierFeatureEmbedding(mapping_size, scale, P_x, P_t)
      modified_layers = [2 * mapping_size + 1] + layers[1:]#[2 * mapping_size*(len(scale)+1)] + layers[1:]
    else:
      modified_layers = layers

    print('Modified Layers: ',modified_layers)

    # Create layers with either regular Linear or RWF layers
    self.layers = nn.ModuleList()
    for i in range(len(modified_layers) - 1):
      if use_rwf:
        # if i==len(modified_layers)-2:
        #   self.layers.append(RandomWeightFactorizationLayer(2*modified_layers[i], modified_layers[i + 1], mu=mu, sigma=sigma))
        # else:
        self.layers.append(RandomWeightFactorizationLayer(modified_layers[i], modified_layers[i + 1], mu=mu, sigma=sigma))
      else:
        self.layers.append(nn.Linear(modified_layers[i], modified_layers[i + 1]))

    self.activation = nn.Tanh()

  def forward(self, x):
    # Apply Fourier feature embedding if selected
    if self.use_fourier_features:
      t_f,x_f = self.fourier_embedding(x[:,0:2])
    t_x_f = torch.cat([t_f, x_f], dim=1)
    # Pass through the network layers with activations
    for i in range(len(self.layers) - 1):
      # t_f = self.activation(self.layers[i](t_f))
      t_x_f = self.activation(self.layers[i](t_x_f))

    # x = torch.multiply(t_f,x_f)#torch.cat([t_f,x_f],dim=1)
    # Output layer (no activation)
    x = self.layers[-1](t_x_f)
    return x

def causal_aware_loss2(residual_losses, unique, inverse_indices, epsilon):
    """
    Computes the causal-aware weighted residual loss using PyTorch in a vectorized manner.

    Args:
        residual_losses (torch.Tensor): Residual losses at each time step t_i. Should be a 1D tensor of size N_t.
        unique (torch.Tensor): Unique time steps (sorted).
        inverse_indices (torch.Tensor): Indices mapping each residual_loss to its corresponding unique timestep.
        epsilon (float): Causality parameter that controls the steepness of the weights w_i.

    Returns:
        torch.Tensor: The causal-aware weighted residual loss L_r(theta).
        bool: True if the stopping condition is met, False otherwise.
        np.ndarray: The weights w_i.
        torch.Tensor: The L2 norm of the weighted residual losses.
    """
    # Sum residual losses for each unique timestep using scatter_add
    loss_t = torch.zeros(unique.size(0), dtype=residual_losses.dtype, device=residual_losses.device)
    
    # Unsqueeze inverse_indices to match the dimensions of loss_t if necessary
    inverse_indices = inverse_indices.view(-1)  # Ensure inverse_indices is 1D
    residual_losses = residual_losses.view(-1)  # Ensure residual_losses is 1D

    loss_t.scatter_add_(0, inverse_indices, residual_losses)

    # Calculate cumulative sum of losses (without the current element for each index)
    with torch.no_grad():
        loss_cumsum = torch.cumsum(loss_t, dim=0) - loss_t
        w_i = torch.exp(-epsilon * loss_cumsum)
    
    # Compute the final loss
    loss_mod = torch.mean(w_i * loss_t)
    norm_pde = torch.norm(w_i * loss_t, p=2)
    
    # Check stopping condition (if all w_i are > 0.99)
    break_ = torch.min(w_i) > 0.99
    break_ = break_.cpu().detach().numpy()

    return loss_mod, break_, w_i.cpu().numpy(), norm_pde

def lle_loss(model, T0, Theta0, u_init, v_init, seta_init, T, Theta, seta_array, F, cal=False, epsilon=100):
  # define initial condition loss
  e_init = model(torch.cat((T0, Theta0), dim=1))
  u_init_pred = e_init[:,0:1]
  v_init_pred = e_init[:,1:2]
  loss_init = torch.mean(torch.square(u_init_pred-u_init)) + torch.mean(torch.square(v_init_pred-v_init))

  norm_i = torch.norm(u_init_pred-u_init, p=2) + torch.norm(v_init_pred-v_init, p=2)

  # define pde loss w.r.t collocation points
  e = model(torch.cat((T, Theta), dim=1))
  u = e[:,0:1]
  v = e[:,1:2]
  du_dt = torch.autograd.grad(u, T, torch.ones_like(u), create_graph=True)[0]
  dv_dt = torch.autograd.grad(v, T, torch.ones_like(v), create_graph=True)[0]
  du_dtau = torch.autograd.grad(u, Theta, torch.ones_like(u), create_graph=True)[0]
  dv_dtau = torch.autograd.grad(v, Theta, torch.ones_like(v), create_graph=True)[0]
  du_dtau2 = torch.autograd.grad(du_dtau, Theta, torch.ones_like(du_dtau), create_graph=True)[0]
  dv_dtau2 = torch.autograd.grad(dv_dtau, Theta, torch.ones_like(dv_dtau), create_graph=True)[0]

  mod_e_sq = u**2 + v**2
  f_u = du_dt + 0.5*dv_dtau2 + v*mod_e_sq + u - seta_array*v - F
  f_v = dv_dt - 0.5*du_dtau2 - u*mod_e_sq + v + seta_array*u

  if cal:
    ids = torch.argsort(T,dim=0)
    loss_pde = 0.5*(torch.square(f_u) + torch.square(f_v))
    loss_pde = loss_pde[ids]
    sorted_T = T[ids]
    unique, inverse_indices = torch.unique(sorted_T, return_inverse=True)
    loss_pde, break_, w_i, norm_pde = causal_aware_loss2(loss_pde, unique, inverse_indices, epsilon=epsilon)
  else:
    loss_pde = torch.mean(torch.square(f_u)) + torch.mean(torch.square(f_v))
    w_i=None
    break_=False

  norm_pde = torch.norm(f_u, p=2) + torch.norm(f_v, p=2)

  norm_i, norm_pde = norm_i.cpu().detach().numpy(),  norm_pde.cpu().detach().numpy()
  norm_sum = norm_i + norm_pde

  lambda_pde, lambda_init = norm_sum/norm_pde, norm_sum/norm_i

  return loss_pde, loss_init, lambda_pde, lambda_init, w_i, break_

# %%
def predict_field(model, t, tau, shape):
  e = model(torch.cat((t, tau), dim=1)).cpu().detach().numpy()
  u = e[:,0:1]
  v = e[:,1:2]
  h = u + 1j*v
  h = h.reshape(shape)
  h2 = np.abs(h)
  h2 = h2.reshape(shape)
  return h2, h

# %%
seta = np.arange(-10, 45, 0.01)
seta_idx = 0
ring1_params =  {
                        'N':211, # Number of modes. It must be odd!
                        'n0': 2.4, # Refractive index
                        'n2': 2.4e-19, # Nonlinear reftactive index [m^2/W]
                        'FSR': 100e9, # Free Spectral Range [Hz]
                        'lambda_0': 1553.4e-9, # CW pump wavelength [m]
                        'kappa': 3e8, # Optical linewidth [Hz]
                        'eta': 0.5, # Coupling efficiency
                        'Veff': 1e-15, # Effective mode volume [m^3]
                        'D2': 2.5e6, # Second order dispersion [Hz]
                        'Pin': 2, # Pump power [W]
                        'dseta_start': 0,
                        'dseta_stop': 10,
                        'dseta_step': 0.01
                    }
ring1 = MRR(ring1_params)

# %%
ring1.calc_beta2()
plt.plot(ring1.theta)
plt.show()

plt.plot(ring1.tau)
plt.show()
P_t = 1*np.max(ring1.tau)
P_x = 2*np.max(np.abs(ring1.theta))
print(P_x, P_t)

# %%
# # generate tau and t
# T0, Theta0, u_init, v_init, T_lb, Theta_lb, T_ub, Theta_ub, T, Theta, seta_array, F = ring1.gen_tau_t(N0=200,Nb=200, Nf=40000)
# print(T_lb.shape, Theta_lb.shape, T_ub.shape, Theta_ub.shape, T.shape, Theta.shape, seta_array.shape, F.shape, u_init.shape, v_init.shape)

# %%
def to_tensor_device(data_dict, device):
    """
    Convert arrays to tensors, set requires_grad=True for specific tensors, and move them to the specified device.

    Parameters:
    - data_dict (dict): Dictionary containing numpy arrays to be converted.
    - device (torch.device): The device to move the tensors to (CPU/GPU).

    Returns:
    - A dictionary with tensors moved to the specified device.
    """
    data_tensors = {}

    for key, value in data_dict.items():
        if isinstance(value, np.ndarray):
            # Set requires_grad=True for 'Theta_lb', 'Theta_ub', 'T', and 'Theta'
            if key in ['T_sampled', 'Theta_sampled']:
                data_tensors[key] = torch.tensor(value, dtype=torch.float32, requires_grad=True).to(device)
            else:
                data_tensors[key] = torch.tensor(value, dtype=torch.float32).to(device)
        else:
            data_tensors[key] = value  # If it's not an array, leave it as is

    return data_tensors


# %%
training_params = {
    'layers': [2, 256, 256, 256, 256, 256, 2],  # Example architecture; modify as needed
    'mapping_size': 64, # Number of Fourier features,
    'scale': [1,10,50],  # Scale for Fourier embedding
    'use_fourier_features': True,  # Set to True if you want to use Fourier features
    'use_rwf': True, # Set to True if you want to use ramdom weight factorization
    'alpha': 0.9, # avg parameter for lamdas
    'cal': True, # Set to True if you want to use causality aware loss
    'M': 20, # Number of time splits for causality aware loss
    'f':1000, # number of time steps to update lamdas
    'epsilons':[0.01], # epsilon values for causality aware loss
    'epochs':500000
}
# modified layers in training params
# training_params['layers'] = [2*training_params['mapping_size']*(len(training_params['scale'])+1)+1] + training_params['layers'][1:]
# %%
wandb.init(project='LLE_PINN', config=training_params)

# set wandb run name
wandb.run.name = 'LLE_PINN_MSFFN_mi_init'
# %%
# # generate tau and t
T0, Theta0, u_init, v_init, seta_init, T_lb, Theta_lb, T_ub, Theta_ub, T, Theta, seta_array, F = \
  ring1.gen_tau_t(N0=200,Nb=10, Nf=50000, seta_idx=seta_idx)
print(T_lb.shape, Theta_lb.shape, T_ub.shape, Theta_ub.shape, T.shape, Theta.shape, seta_array.shape, F.shape, u_init.shape, v_init.shape)
# print number of unique elemenst in T and Theta
print(len(np.unique(T)), len(np.unique(Theta)))
# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# %%
network = PINN(layers=training_params['layers'],
               mapping_size=training_params['mapping_size'],
               scale=training_params['scale'],
               use_fourier_features=training_params['use_fourier_features'],
               use_rwf=training_params['use_rwf'],
               P_x=P_x,
               P_t=P_t,
               ).to(device)
optimizer = optim.Adam(network.parameters(), lr=1e-3)
# watch the gradients of network
# wandb.watch(network, log='all', log_freq=500)
print(network)

# %%
# put all the data as gpu tensors
data_tensors = to_tensor_device({'T0': T0, 'Theta0': Theta0, 'u_init': u_init, 'v_init': v_init, 'seta_init': seta_init,
     'T_sampled': T, 'Theta_sampled': Theta,'seta_array': seta_array, 'F': F}, device)

# %%
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=250)
# Function to reset the learning rate
def reset_learning_rate(optimizer, lr=1e-3):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    # print reste lr
    print(f"Learning rate reset to {lr}")

def lr_lambda(step):
    decay_steps = 2500
    decay_rate = 0.9
    return decay_rate ** (step / decay_steps)

# Set up the LambdaLR scheduler
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# %%
T_val = ring1.tau#np.linspace(T.min(), T.max(), 1100)
Theta_val = ring1.theta#np.linspace(Theta.min(), Theta.max(), 441)
T_val, Theta_val = np.meshgrid(T_val, Theta_val)
T_val = T_val.flatten()[:, np.newaxis]
Theta_val = Theta_val.flatten()[:, np.newaxis]
T_val = torch.Tensor(T_val).to(device)
Theta_val = torch.Tensor(Theta_val).to(device)
seta_val = ring1.seta[0][0]*torch.ones_like(T_val, device=device) #torch.Tensor(ring1.seta_array.flatten()[:, np.newaxis]).to(device)
# %%
def validate(network, T_val, Tau_val,seta_val, T_lb=None, Tau_lb=None, T_ub=None, Tau_ub=None, T0=None, Tau0=None):
    network.eval()
    with torch.no_grad():
        e = network(torch.cat((T_val, Tau_val), dim=1)).cpu().detach().numpy()
    network.train()

    u = e[:, 0:1]
    v = e[:, 1:2]
    h = u + 1j * v
    h = h.reshape(len(ring1.modes), len(ring1.seta))

    # Figure 1: Predicted Field
    fig1, ax1 = plt.subplots(figsize=(14, 3))
    if ring1.seta.max() == ring1.seta.min():
        cax1 = ax1.imshow(np.abs(h), cmap='jet', origin='lower', aspect='auto', extent=[ring1.tau.min(), ring1.tau.max(), \
                                                                                        ring1.theta.min(), ring1.theta.max()])
        ax1.set_xlabel(r"$\tau=\kappa t/2$", fontsize=14)
    else:
      cax1 = ax1.imshow(np.abs(h), cmap='jet', origin='lower', 
                        extent=[ring1.seta.min(), ring1.seta.max(), ring1.theta.min(), ring1.theta.max()], 
                        aspect='auto')
      ax1.set_xlabel(r"$\mathcal{\zeta} _0$", fontsize=14)
    fig1.colorbar(cax1, ax=ax1)
    ax1.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    ax1.set_ylabel(r"$\theta = (\varphi - D_1 t) \sqrt{\frac{\kappa}{2D_2}}$", fontsize=14)
    
    ax1.set_title(r"$|\mathcal{\psi} (\tau, \theta)|$", fontsize=16, fontweight='bold')
    ax1.tick_params(axis='both', labelsize=14)

    if T_lb is not None and Tau_lb is not None:
        ax1.scatter(T_lb, Tau_lb, c='k', marker='x', s=35)
    if T_ub is not None and Tau_ub is not None:
        ax1.scatter(T_ub, Tau_ub, c='k', marker='x', s=35)
    if T0 is not None and Tau0 is not None:
        ax1.scatter(T0, Tau0, c='m', marker='x', s=35)
    fig1.tight_layout()

    # FFT transformation and second plot
    fft_h = np.fft.fftshift(np.fft.ifft(h, axis=0), axes=0)
    abs_h = np.abs(fft_h)

    # Figure 2: FFT of Predicted Field
    fig2, ax2 = plt.subplots(figsize=(14, 3))
    if ring1.seta.max() == ring1.seta.min():
        cax2 = ax2.imshow(abs_h, cmap='jet', origin='lower', aspect='auto', extent=[ring1.tau.min(), ring1.tau.max(),\
                                                                                     ring1.modes.min(), ring1.modes.max()])
        ax2.set_xlabel(r"$\tau=\kappa t/2$", fontsize=14)
    else:
        cax2 = ax2.imshow(abs_h, cmap='jet', origin='lower', 
                          extent=[ring1.seta.min(), ring1.seta.max(), ring1.modes.min(), ring1.modes.max()], 
                          aspect='auto')
        ax2.set_xlabel(r"$\mathcal{\zeta} _0$", fontsize=14)
    fig2.colorbar(cax2, ax=ax2)
    ax2.set_ylabel(r"$\mu$", fontsize=14)
    
    ax2.set_title(r"$F \cdot T \{ \mathcal{\psi} (\tau, \theta) \} $", fontsize=16, fontweight='bold')
    ax2.tick_params(axis='both', labelsize=14)
    fig2.tight_layout()

    return fig1, fig2  # Return the figures
  # return h2

# %%
# Save dictionary as a JSON file
def save_dict(name, data):
    json_file_path = name
    with open(json_file_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)

# Load the JSON file back into a dictionary
def load_dict(name):
    with open(name, 'r') as json_file:
        loaded_data = json.load(json_file)
    return loaded_data

# %%
def plot_temporal_weights(w_i,x=None):
  fig, ax = plt.subplots(figsize=(6, 4))
  if x is not None:
    ax.plot(x, w_i, 'o')
  else:
    ax.plot(np.arange(1,len(w_i)+1), w_i, 'o')
  ax.set_xlabel(r"$\mathcal{\zeta}_0$", fontsize=14)
  ax.set_ylabel(r"$\mathcal{W}_\mathcal{i}$", fontsize=14)
  # plt.xscale('log')
  ax.grid(True)
  ax.set_ylim(-0.2, 1.2)
  # tight_layout 
  fig.tight_layout()
  # plt.show()
  return fig
    

# %%
loss_history = []
lr_hist = []
logs={}
M = training_params['M']  # Number of time marching steps
cal = training_params['cal']  # Causality-aware loss
lamda_pde, lamda_init = 1.0, 100.0  # Initial lambda values
epochs = training_params['epochs']
for epsilon in training_params['epsilons']:
    print(f"Starting training with epsilon={epsilon}\n")
    for epoch in tqdm(range(epochs), ncols=120):
        T0_, Theta0_ = data_tensors['T0'], data_tensors['Theta0']
        u_init_, v_init_, seta_init = data_tensors['u_init'], data_tensors['v_init'], data_tensors['seta_init']
        T_, Theta_ = data_tensors['T_sampled'], data_tensors['Theta_sampled']
        seta_array_, F_ = data_tensors['seta_array'], data_tensors['F']

        # Zero the gradients
        optimizer.zero_grad()

        # Calculate losses (LLE loss function)
        loss_pde, loss_init, lamda_pde_new, lamda_init_new,w_i,break_ = lle_loss(
            network, T0_, Theta0_, u_init_, v_init_, seta_init, T_, Theta_, seta_array_, F_,
            cal, epsilon=epsilon
        )

        if break_:
            print(f"Stopping training at epoch {epoch} due to the stopping condition.")
            break

        # Total loss
        loss = lamda_pde * loss_pde + lamda_init * loss_init

        # Adaptive lambda adjustments every few epochs (e.g., every 'f' epochs)
        if epoch % training_params['f'] == 0:
            lamda_pde = training_params['alpha'] * lamda_pde + (1 - training_params['alpha']) * lamda_pde_new
            lamda_init = training_params['alpha'] * lamda_init + (1 - training_params['alpha']) * lamda_init_new
            lamda_init = np.clip(lamda_init, 1e-1, 1e4)
            logs['lamda_pde'] = lamda_pde
            logs['lamda_init'] = lamda_init

        # Backpropagate the loss
        loss.backward()

        # Clip gradients to avoid exploding gradients
        torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=1.0)

        # Optimize step
        optimizer.step()

        # Save loss and learning rate history
        loss_history.append([loss_pde.item(), loss_init.item()])
        lr_hist.append(scheduler.get_last_lr()[0])
        logs['loss_pde'] = loss_pde.item()
        logs['loss_init'] = loss_init.item()
        logs['loss'] = loss.item()
        logs['lr_hist'] = scheduler.get_last_lr()[0]

        # Print validation and learning rate every 500 epochs
        if epoch % 1000 == 0:
            if cal:
                weights_fig = plot_temporal_weights(w_i, np.linspace(ring1.tau.min(), ring1.tau.max(), len(w_i)))
                logs["Temporal Weights"] =  wandb.Image(weights_fig)
                plt.close(weights_fig)

            fig1,fig2 = validate(network, T_val, Theta_val, seta_val)
            logs["Predicted Field"] =  wandb.Image(fig1)
            logs["FFT of Predicted Field"] =  wandb.Image(fig2)
            

            # Close figures to avoid memory leaks
            plt.close(fig1)
            plt.close(fig2)
            # print(f"Epoch {epoch}, Train Loss: {loss.item()}, Loss_b: {loss_b.item()}, Loss_pde: {loss_pde.item()}, Loss_init: {loss_init.item()}")
            # print(f"Current learning rate: {scheduler.get_last_lr()[0]}")
        
        wandb.log(logs)
        # Step scheduler
        scheduler.step()

        if 'FFT of Predicted Field' in logs:
            logs.pop('FFT of Predicted Field')
        if 'Predicted Field' in logs:
            logs.pop('Predicted Field') 
    # reset learning rate and scheduler
    reset_learning_rate(optimizer, lr=1e-3)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

wandb.finish()
# %%
# name = 'saved_pinn_working_v14'
# def save_model(network, name):
#     torch.save(network.state_dict(), name+'.pth')
#     save_dict(name+'.json', training_params)
#     print('Model saved successfully ... ')
# save_model(network, name)
# def load_model(name):
#     # load dict
#     loaded_dict = load_dict(name+'.json')
#     # load model
#     model = PINN(layers=loaded_dict['layers'],
#                mapping_size=loaded_dict['mapping_size'],
#                scale=loaded_dict['scale'],
#                use_fourier_features=loaded_dict['use_fourier_features'],
#                use_rwf=loaded_dict['use_rwf'],
#                P_x=P_x,
#                P_t=P_t,
#                ).to(device)
#     model.load_state_dict(torch.load(name+'.pth'))
#     model.eval()
#     print('Model loaded successfully ... ')
#     return model

# %%
# Extract loss components and learning rates
# loss_b = [row[0] for row in loss_history]
# loss_pde = [row[1] for row in loss_history]
# loss_init = [row[2] for row in loss_history]

# # Create the double y-axis plot
# fig, ax1 = plt.subplots()

# # Plot losses on the first y-axis
# ax1.set_xlabel('Epoch', fontsize=14)
# ax1.set_ylabel(r"$ \mathcal{L}_\mathrm{nn}$", fontsize=14)
# ax1.plot(loss_b, label=r"$\mathcal{L}_\mathrm{b}$", color='blue')
# ax1.plot(loss_pde, label=r"$\mathcal{L}_\mathrm{pde}$", color='green')
# ax1.plot(loss_init, label=r"$\mathcal{L}_\mathrm{init}$", color='red')
# ax1.set_yscale('log')
# ax1.legend(fontsize=14)
# ax1.grid(True)

# # Create a second y-axis
# ax2 = ax1.twinx()

# # Plot learning rates on the second y-axis
# ax2.set_ylabel('Learning Rate', fontsize=14)
# ax2.plot(lr_hist, color='black', linestyle='--', label='lr')
# ax2.legend(loc='center right',fontsize=14)
# ax2.set_yscale('log')
# # ax2.grid(True)

# # Set the title and labels
# plt.title('Loss and Learning Rate vs. Epochs')
# plt.tight_layout()
# plt.savefig(name+'_learn'+'.png')
# plt.show()

# %%
# training_params = load_dict('saved_pinn_working_v10.json')
# network2 = PINN(layers=training_params['layers'],
#                mapping_size=training_params['mapping_size'],
#                scale=training_params['scale'],
#                use_fourier_features=training_params['use_fourier_features'],
#                use_rwf=training_params['use_rwf'],
#                P_x=P_x,
#                P_t=P_t,
#                ).to(device)
# network2.load_state_dict(torch.load('saved_pinn_working_v10.pth', weights_only=False))

# %%
# validate(network, T_val, Theta_val)

# # %%
# h_pred2, h = predict_field(network, T_val, Theta_val, (len(ring1.modes),len(ring1.seta)))
# # savemat('saved_pinn_working_v4_field.mat', {'h':h})

# # %%
# # surf plot of the field h_pred\
# fig = plt.figure(figsize=(14, 7))
# ax = fig.add_subplot(111, projection='3d')
# X, Y = Theta_val.cpu().numpy().reshape(len(ring1.modes),len(ring1.seta)), T_val.cpu().numpy().reshape(len(ring1.modes),len(ring1.seta))
# # X,Y = ring1.seta_array.reshape(len(ring1.modes),len(ring1.seta)), Theta_val.reshape(len(ring1.modes),len(ring1.seta))
# ax.plot_surface(X, Y, h_pred2, cmap='jet')
# ax.set_xlabel(r"$\theta$", fontsize=14)
# ax.set_ylabel(r"$\tau$", fontsize=14)
# # ax.set_
# # ax.view_init(elev=30, azim=30)
# # plt.tight_layout()
# plt.show()

# # %%
# plt.plot(ring1.seta,h_pred2.sum(axis=0))
# plt.show()

# # %%
# def pad_array_center(arr, target_N=4096):
#   N, M = arr.shape  # Original dimensions of the array

#   if N >= target_N:
#       raise ValueError("N must be smaller than the target size (4096).")

#   # Create a new array of zeros with the desired shape (4096, M)
#   padded_array = np.zeros((target_N, M), dtype=np.complex128)

#   # Calculate the start and end indices to center the original array
#   start_idx = (target_N - N) // 2
#   end_idx = start_idx + N

#   # Place the original array in the center of the padded array
#   padded_array[start_idx:end_idx, :] = arr

#   return padded_array

# # %%
# idx = 4000
# t = np.linspace(-1/(ring1.FSR/2),1/(ring1.FSR/2), ring1.N)*1e12
# plt.plot(t,h_pred2[:,idx])
# plt.grid()
# plt.xlabel('Time (ps)',fontsize=14)
# plt.ylabel('Amplitude (a.u.)',fontsize=14)
# plt.show()

# h_padded = pad_array_center(h_pred2)
# fft_p = np.fft.fftshift(np.fft.fft(h_padded[:, idx]))
# abs_b =10*np.log10(np.abs(fft_p)) + 30
# # abs_b = np.clip(abs_b,30,55)
# # abs_p = np.clip(abs_b,-100,0)
# f = np.linspace(ring1.freq_array.min(), ring1.freq_array.max(),4096)/1e12
# plt.grid()
# plt.plot(f,abs_b)
# # plt.yscale('log')
# plt.ylabel('Amplitude (dBm)',fontsize=14)
# plt.xlabel('Freq (THz)',fontsize=14)
# plt.show()

# # %%
# plt.imshow(np.real(h), cmap='jet', origin='lower', extent=[T.min(), T.max(), Theta.min(), Theta.max()], aspect='auto')
# plt.colorbar()
# plt.show()

# plt.imshow(np.imag(h), cmap='jet', origin='lower', extent=[T.min(), T.max(), Theta.min(), Theta.max()], aspect='auto')
# plt.colorbar()
# plt.show()

# # %%
# tau = 30
# theta = ring1.theta
# Tau_val1, Theta_val1 = np.meshgrid(tau, theta)
# Tau_val1 = Tau_val1.flatten()[:, np.newaxis]
# Theta_val1 = Theta_val1.flatten()[:, np.newaxis]
# Tau_val1 = torch.Tensor(Tau_val1).to(device)
# Theta_val1 = torch.Tensor(Theta_val1).to(device)
# h_pred2, h = predict_field(network, Tau_val1, Theta_val1, (len(ring1.modes),1))
# %%
class DintFourierFeatureEmbedding(nn.Module):
  def __init__(self, dint, P_x=1):
    super(DintFourierFeatureEmbedding, self).__init__()
    # clip scale between 1,10
    # scale = max(1, min(10, scale))

    self.P_x = P_x

    # Detect the available device (GPU if available, otherwise CPU)
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    self.dint = np.transpose(dint[:,np.newaxis])
    self.register_buffer('B_x',torch.Tensor(self.dint, device=self.device))

  def forward(self, x):
    # Ensure input is on the same device as the random matrix
    x = x.to(self.device)

    x_proj = (2 * torch.pi * x[:, 1:2]/self.P_x) @ self.B_x

    # Project input using Fourier features
    # f_proj = torch.cat([torch.sin(t_proj), torch.cos(t_proj), torch.sin(x_proj), torch.cos(x_proj)], dim=1)

    return torch.cat([torch.cos(x_proj), torch.sin(x_proj)], dim=1)
  
# %%
class DintFourierFeatureEmbedding2(nn.Module):
  def __init__(self, dint, P_x=1):
    super(DintFourierFeatureEmbedding2, self).__init__()
    # clip scale between 1,10
    # scale = max(1, min(10, scale))

    self.P_x = P_x

    # Detect the available device (GPU if available, otherwise CPU)
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    self.dint = np.transpose(dint[:,np.newaxis]*self.P_x/dint.max())
    self.register_buffer('B_x',torch.Tensor(self.dint, device=self.device))

  def forward(self, x):
    # Ensure input is on the same device as the random matrix
    x = x.to(self.device)

    x_proj = (2 * torch.pi * x[:, 1:2]) @ self.B_x

    # Project input using Fourier features
    # f_proj = torch.cat([torch.sin(t_proj), torch.cos(t_proj), torch.sin(x_proj), torch.cos(x_proj)], dim=1)

    return torch.cat([torch.cos(x_proj), torch.sin(x_proj)], dim=1)

# %% Dint ff visualize
dintff = DintFourierFeatureEmbedding2(ring1.dint[0:(len(ring1.dint)+1)//2], P_x)

# %%
transformed_input = dintff(torch.cat((data_tensors['T_sampled'], data_tensors['Theta_sampled']), dim=1)).cpu().detach().numpy()
# print shape
print(transformed_input.shape)
# %%
plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
plt.scatter(T,Theta,color='b',label='Original')
plt.xlabel('T')
plt.ylabel('Theta')
plt.title('Original')

plt.subplot(1, 2, 2)
plt.scatter(T,Theta,c=transformed_input[:,-50],cmap='jet',label='Transformed')
plt.xlabel('T')
plt.ylabel('Theta')
plt.title('Transformed')
plt.colorbar()
plt.show()
# %%
