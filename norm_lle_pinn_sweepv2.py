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
import json
import wandb
import argparse

# %%
data = loadmat('A.mat')['A']
data_ifft = np.fft.fft(data, axis=0)
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

    self.seta = np.arange(self.dseta_start, self.dseta_stop, self.dseta_step)[:,np.newaxis]
    # self.seta = 24*np.ones_like(self.seta)
    self.del_omega = self.seta*self.kappa/2
    self.seta_array = np.repeat(self.seta, self.N, axis=1).T

    self.omega_mu = self.omega_0 + 2*np.pi*self.FSR*self.mu + np.pi*self.D2*self.mu**2


  def calc_beta2(self):
    f_center = self.f0
    mode_numbers = np.arange(-(self.N - 1) / 2, ((self.N - 1) / 2) + 1, 1)

    L = 2*np.pi*self.R
    self.freq_array = f_center + mode_numbers * self.FSR
    ng_estimated = cts.c / (L * np.gradient(self.freq_array))


    self.D_int = (self.D2/2)*(self.modes)**2 + 1e-6
    # # Calculate beta_2 as a function of frequency
    D1 = 2 * np.pi * self.FSR  # D1 in rad/s
    # beta_2 = -(self.n0* 2*np.pi*self.D2) / (cts.c * (2*np.pi*self.FSR)**2)
    # self.beta_2_array = beta_2_array

    # LD = self.Tr**2/np.abs(beta_2)# dispersion length
    theta_norm = np.sqrt(1/(2*self.d2))#np.sqrt((self.kappa/(2*np.pi))/(2*self.D2))
    self.theta = (np.linspace(0, 2*np.pi, len(self.seta)) - D1*np.linspace(0, len(self.seta)*self.Tr, len(self.seta)))*theta_norm # normalized frequency detuning
    # theta mod 2pi
    self.theta = np.mod(self.theta, 2*np.pi)-np.pi
    self.tau = 0.5*self.kappa*np.linspace(0, len(self.seta)*self.Tr, len(self.seta)) # normalized time coordinate tau = kappa*t/2
    

  def find_Nmax(self):
    D2 = 2*np.pi*self.D2
    self.N_max = np.floor(np.sqrt(self.kappa/D2))
    # return a random integer between 1 and N_max
    return np.random.randint(1, self.N_max)

  def gen_Psi_init(self):
    psi_0 = self.f_amp/self.seta[0]**2 - 1j*self.f_amp/self.seta[0]
    B = np.sqrt(2*self.seta[0])
    phi_0 = np.arccos(np.sqrt(8*self.seta[0])/(np.pi*self.f_amp))
    C2 = 4*self.seta[0]/(np.pi*self.f_amp) + 1j*np.sqrt( 2*self.seta[0] - 16*self.seta[0]**2/((np.pi*self.f_amp)**2) )
    self.soliton_spacing = np.ceil((8/B)*np.sqrt(2*self.d2))
    N = self.find_Nmax()
    solitons = 0
    phi_ = np.linspace(-np.pi,np.pi,self.N)
    for ii in range(N):
      phi_j = np.roll(phi_, int(self.soliton_spacing*ii))
      solitons += 1/( np.cosh( np.sqrt(self.seta[0]/self.d2) * (phi_-phi_j) ) )
    psi_init = psi_0 + C2*solitons
    return psi_init


  def gen_tau_t(self, N0=100, Nb=100, Nf=20000):

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
    # u_init = np.real(data_ifft[idx,1:2])
    # v_init = np.imag(data_ifft[idx,1:2])

    psi_init = np.fft.ifft(np.sqrt(2*self.g/self.kappa)*(np.random.randn(len(T0)) + 1j*np.random.randn(len(T0))))
    u_init = np.real(psi_init)[:,np.newaxis]
    v_init = np.imag(psi_init)[:,np.newaxis]

    # u_init = u_init[idx]#[:,np.newaxis]
    # v_init = v_init[idx]#[:,np.newaxis]

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
    return T0, Theta0, u_init, v_init, T_lb, Theta_lb, T_ub, Theta_ub, T, Theta, seta_array, F

  def gen_tau_t_lhs(self, N0=100, Nb=100, Nf=20000):

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

    # u_init = np.real(data_ifft[idx, 1:2])
    # v_init = np.imag(data_ifft[idx, 1:2])

    psi_init = np.sqrt(2*self.g/self.kappa)*(np.random.rand(len(Theta0)) + 1j*np.random.rand(len(Theta0)))
    u_init = np.real(psi_init)[:,np.newaxis]
    v_init = np.real(psi_init)[:,np.newaxis]

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
    seta_b = self.seta_array.flatten()[:, np.newaxis]

    idx = np.random.choice(len(T_lb), Nb, replace=False)
    idx = np.sort(idx)
    T_lb = T_lb[idx]
    Theta_lb = Theta_lb[idx]
    T_ub = T_ub[idx]
    Theta_ub = Theta_ub[idx]
    seta_b = seta_b[idx]

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
    # seta_array = self.seta_array.flatten()[:, np.newaxis][:Nf]
    # Sample seta_array in the same manner
    seta_array_flat = self.seta_array.flatten()[:, np.newaxis]
    seta_array = np.min(seta_array_flat) + (np.max(seta_array_flat) - np.min(seta_array_flat)) * lhs_samples[:, 0][:, np.newaxis]

    return T0, Theta0, u_init, v_init, seta_init, T_lb, Theta_lb, T_ub, Theta_ub, seta_b, T_sampled, Theta_sampled, seta_array, F

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

class FourierFeatureEmbedding(nn.Module):
  def __init__(self, mapping_size, scales=[5.0], P_x=1, P_t=1):
    super(FourierFeatureEmbedding, self).__init__()
    # clip scale between 1,10
    # scale = max(1, min(10, scale))

    # Detect the available device (GPU if available, otherwise CPU)
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    self.P_x = P_x
    self.P_t = P_t

    # Random matrix for projecting inputs to Fourier feature space, placed on the correct device
    # Instead of directly creating tensors, register them as buffers
    self.register_buffer('B_t', scales[0] * torch.randn((1, mapping_size), device=self.device))
    self.register_buffer('B_x', torch.cat([scale * torch.randn((1, mapping_size), device=self.device) for scale in scales], dim=1))

  def forward(self, x):
    # Ensure input is on the same device as the random matrix
    x = x.to(self.device)

    t_proj = (2 * torch.pi * x[:,0:1] / self.P_t) @ self.B_t
    x_proj = (2 * torch.pi * x[:,1:2] / self.P_x) @ self.B_x

    # Project input using Fourier features
    return torch.cat([torch.cos(t_proj), torch.sin(t_proj)], dim=1), torch.cat([torch.cos(x_proj), torch.sin(x_proj)], dim=1)

class PINN(nn.Module):
  def __init__(self, layers, mapping_size=64, scale=[10.0], use_fourier_features=True, use_rwf=False, \
               P_x=1, P_t=1, mu=1.0, sigma=0.1, activation=nn.Tanh()):
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
      modified_layers = [2 * mapping_size*(len(scale)+1)+1] + layers[1:]
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

    self.activation = activation

  def forward(self, x):
    # Apply Fourier feature embedding if selected
    if self.use_fourier_features:
      t_f,x_f = self.fourier_embedding(x[:,0:2])
    t_x_f = torch.cat([t_f, x_f, x[:,2:3]], dim=1)
    # Pass through the network layers with activations
    for i in range(len(self.layers) - 1):
      # t_f = self.activation(self.layers[i](t_f))
      t_x_f = self.activation(self.layers[i](t_x_f))

    # x = torch.multiply(t_f,x_f)#torch.cat([t_f,x_f],dim=1)
    # Output layer (no activation)
    x = self.layers[-1](t_x_f)
    return x


def lle_loss(model, T0, Theta0, u_init, v_init, seta_init, T, Theta, seta_array, F):
  # define initial condition loss
  e_init = model(torch.cat((T0, Theta0, seta_init), dim=1))
  u_init_pred = e_init[:,0:1]
  v_init_pred = e_init[:,1:2]
  loss_init = torch.mean(torch.square(u_init_pred-u_init)) + torch.mean(torch.square(v_init_pred-v_init))

  norm_i = torch.norm(u_init_pred-u_init, p=2) + torch.norm(v_init_pred-v_init, p=2)


  # define pde loss w.r.t collocation points
  e = model(torch.cat((T, Theta, seta_array), dim=1))
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
 
  loss_pde = torch.mean(torch.square(f_u)) + torch.mean(torch.square(f_v))

  norm_pde = torch.norm(f_u, p=2) + torch.norm(f_v, p=2)

  norm_i, norm_pde = norm_i.cpu().detach().numpy(),  norm_pde.cpu().detach().numpy()
  norm_sum = norm_i + norm_pde

  lambda_pde, lambda_init = norm_sum/norm_pde, norm_sum/norm_i

  return loss_pde, loss_init, lambda_pde, lambda_init

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
# Function to reset the learning rate
def reset_learning_rate(optimizer, lr=1e-3):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    # print reste lr
    print(f"Learning rate reset to {lr}")
def lr_lambda(step):
    decay_steps = 5000
    decay_rate = 0.9
    return decay_rate ** (step / decay_steps)

# # Set up the LambdaLR scheduler
# scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# %%


def validate(network, T_val, Tau_val,seta_val, T_lb=None, Tau_lb=None, T_ub=None, Tau_ub=None, T0=None, Tau0=None):
    network.eval()
    with torch.no_grad():
        e = network(torch.cat((T_val, Tau_val, seta_val), dim=1)).cpu().detach().numpy()
    network.train()

    u = e[:, 0:1]
    v = e[:, 1:2]
    h = u + 1j * v
    h = h.reshape(len(ring1.modes), len(ring1.seta))

    # Figure 1: Predicted Field
    fig1, ax1 = plt.subplots(figsize=(14, 3))
    cax1 = ax1.imshow(np.abs(h), cmap='jet', origin='lower', 
                      extent=[ring1.seta.min(), ring1.seta.max(), ring1.theta.min(), ring1.theta.max()], 
                      aspect='auto')
    fig1.colorbar(cax1, ax=ax1)
    ax1.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    ax1.set_ylabel(r"$\theta = (\varphi+ D_1 t) \sqrt{\frac{\kappa}{2D_2}}$", fontsize=14)
    ax1.set_xlabel(r"$\mathcal{\zeta} _0$", fontsize=14)
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
        cax2 = ax2.imshow(abs_h, cmap='jet', origin='lower', aspect='auto')
    else:
        cax2 = ax2.imshow(abs_h, cmap='jet', origin='lower', 
                          extent=[ring1.seta.min(), ring1.seta.max(), ring1.modes.min(), ring1.modes.max()], 
                          aspect='auto')
    fig2.colorbar(cax2, ax=ax2)
    ax2.set_ylabel(r"$\mu$", fontsize=14)
    ax2.set_xlabel(r"$\mathcal{\zeta} _0$", fontsize=14)
    ax2.set_title(r"$\mathbf{F} \cdot \mathbf{T} \{ \mathcal{\psi} (\tau, \theta) \} $", fontsize=16, fontweight='bold')
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
def train(config):
   with wandb.init(project='LLE_PINN',config=config):
        config = wandb.config
        layers = [3] + [config.num_nodes]*config.num_layers + [2]
        if config.activation == 'tanh':
            activation = nn.Tanh()
            use_rwf = True
        elif config.activation == 'sigmoid':
            activation = nn.Sigmoid()
            use_rwf = True
        elif config.activation == 'swish':
                activation = nn.SiLU()
                use_rwf = False
            
           
        network = PINN(layers=layers,
                    mapping_size=config.mapping_size,
                    scale=config.scale,
                    use_fourier_features=True,
                    use_rwf=use_rwf,
                    P_x=P_x,
                    P_t=P_t,
                    sigma=config.sigma,
                    activation=activation
                    ).to(device)

        if config.optimizer == 'adam':
            optimizer = optim.Adam(network.parameters(), lr=1e-3, weight_decay=1e-4)
        elif config.optimizer == 'adamw':
            optimizer = optim.AdamW(network.parameters(), lr=1e-3, weight_decay=1e-4)
        elif config.optimizer == 'rmsprop':
            optimizer = optim.RMSprop(network.parameters(), lr=1e-3, weight_decay=1e-4)
        elif config.optimizer == 'nadam':
            optimizer = optim.NAdam(network.parameters(), lr=1e-3, decoupled_weight_decay=True,weight_decay=1e-4)
        # watch the gradients of network
        wandb.watch(network, log='all', log_freq=100)
        print(network)

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=100)

        loss_history = []
        lr_hist = []
        logs={}

        lamda_pde, lamda_init = 1.0, 100.0  # Initial lambda values
        epochs = 50000

        for epoch in tqdm(range(epochs), ncols=120):
            T0_, Theta0_ = data_tensors['T0'], data_tensors['Theta0']
            u_init_, v_init_, seta_init = data_tensors['u_init'], data_tensors['v_init'], data_tensors['seta_init']
            T_, Theta_ = data_tensors['T_sampled'], data_tensors['Theta_sampled']
            seta_array_, F_ = data_tensors['seta_array'], data_tensors['F']

            # Zero the gradients
            optimizer.zero_grad()

            # Calculate losses (LLE loss function)
            loss_pde, loss_init, lamda_pde_new, lamda_init_new = lle_loss(
                network, T0_, Theta0_, u_init_, v_init_, seta_init, T_, Theta_, seta_array_, F_)

            # Total loss
            loss = lamda_pde * loss_pde + lamda_init * loss_init

            # Adaptive lambda adjustments every few epochs (e.g., every 'f' epochs)
            if epoch % training_params['f'] == 0:
                lamda_pde = training_params['alpha'] * lamda_pde + (1 - training_params['alpha']) * lamda_pde_new
                lamda_init = training_params['alpha'] * lamda_init + (1 - training_params['alpha']) * lamda_init_new
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
            logs['lr_hist'] = scheduler.get_last_lr()[0]
            logs['loss'] = loss.item()

            # Print validation and learning rate every 500 epochs
            if epoch % 1500 == 0:

                fig1,fig2 = validate(network, T_val, Theta_val, seta_val)
                logs["Predicted Field"] =  wandb.Image(fig1)
                logs["FFT of Predicted Field"] =  wandb.Image(fig2)
                
                # Close figures to avoid memory leaks
                plt.close(fig1)
                plt.close(fig2)

            
            wandb.log(logs)
            # Step scheduler
            scheduler.step(loss.item())

            if 'FFT of Predicted Field' in logs:
                logs.pop('FFT of Predicted Field')
            if 'Predicted Field' in logs:
                logs.pop('Predicted Field') 
# %%
def parse_args():
#    get the wandb parameters
    parser = argparse.ArgumentParser(description='Sweep parameters for LLE_PINN')
    parser.add_argument('--num_layers', type=int, default=3, help='Number of layers')
    parser.add_argument('--activation', type=str, default='tanh', help='Activation function')
    parser.add_argument('--num_nodes', type=int, default=128, help='Number of nodes per layer')
    parser.add_argument('--sigma', type=float, default=0.2, help='Sigma for random weight factorization')
    parser.add_argument('--scale', type=list, default=[10], help='Scale for Fourier embedding')
    parser.add_argument('--mapping_size', type=int, default=64, help='Number of Fourier features')
    parser.add_argument('--optimizer', type=str, default='adam', help='Optimizer type')
    args = parser.parse_args()
    
    return args
# %%
if __name__ == '__main__':
    args = parse_args()
    ring1_params =  {
                            'N': 256, # Number of modes. It must be odd!
                            'n0': 2.4, # Refractive index
                            'n2': 2.4e-19, # Nonlinear reftactive index [m^2/W]
                            'FSR': 100e9, # Free Spectral Range [Hz]
                            'lambda_0': 1553.4e-9, # CW pump wavelength [m]
                            'kappa': 3e8, # Optical linewidth [Hz]
                            'eta': 0.5, # Coupling efficiency
                            'Veff': 1e-15, # Effective mode volume [m^3]
                            'D2': 2.5e6, # Second order dispersion [Hz]
                            'Pin': 2, # Pump power [W]
                            'dseta_start': -5,
                            'dseta_stop': 30,
                            'dseta_step': 0.01
                        }
    ring1 = MRR(ring1_params)
    print(ring1.seta_array.shape)



    ring1.calc_beta2()
    plt.plot(ring1.theta)
    plt.show()

    plt.plot(ring1.tau)
    plt.show()
    P_t = 4*np.max(ring1.tau)
    P_x = 2*np.max(np.abs(ring1.theta))
    print(P_x, P_t)

    training_params = {
    # 'layers': [3, 256, 256, 256, 256, 2],  # Example architecture; modify as needed
    # 'mapping_size': 64, # Number of Fourier features,
    # 'scale': [1,10,20],  # Scale for Fourier embedding
    # 'use_fourier_features': True,  # Set to True if you want to use Fourier features
    # 'use_rwf': True, # Set to True if you want to use ramdom weight factorization
    'alpha': 0.9, # avg parameter for lamdas
    # 'cal': False, # Set to True if you want to use causality aware loss
    # 'M': 20, # Number of time splits for causality aware loss
    'f':1000, # number of time steps to update lamdas
    'epsilons':[0.01],
    'epochs':20000
    }
    # set wandb run name
    # wandb.run.name = 'LLE_PINN_MSFFN_RWF_no_tm'
    # # generate tau and t
    T0, Theta0, u_init, v_init, seta_init, T_lb, Theta_lb, T_ub, Theta_ub, seta_b, T, Theta, seta_array, F = \
    ring1.gen_tau_t_lhs(N0=200,Nb=400, Nf=50000)
    print(T_lb.shape, Theta_lb.shape, T_ub.shape, Theta_ub.shape, T.shape, Theta.shape, seta_array.shape, F.shape, u_init.shape, v_init.shape)
    print(seta_init.shape, seta_b.shape)
    # print number of unique elemenst in T and Theta
    print(len(np.unique(T)), len(np.unique(Theta)))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # put all the data as gpu tensors
    data_tensors = to_tensor_device({'T0': T0, 'Theta0': Theta0, 'u_init': u_init, 'v_init': v_init, 'seta_init': seta_init,
        'T_sampled': T, 'Theta_sampled': Theta,'seta_array': seta_array, 'F': F}, device)
    T_val = ring1.tau#np.linspace(T.min(), T.max(), 1100)
    Theta_val = ring1.theta#np.linspace(Theta.min(), Theta.max(), 441)
    T_val, Theta_val = np.meshgrid(T_val, Theta_val)
    T_val = T_val.flatten()[:, np.newaxis]
    Theta_val = Theta_val.flatten()[:, np.newaxis]
    T_val = torch.Tensor(T_val).to(device)
    Theta_val = torch.Tensor(Theta_val).to(device)
    seta_val = torch.Tensor(ring1.seta_array.flatten()[:, np.newaxis]).to(device)

    train(args)

# %%
# sweep_config = {
#     'method': 'bayes',
#     'metric': {
#         'name': 'loss',
#         'goal': 'minimize'
#     },
#     'parameters': {
#         'num_layers': {
#             'values': [2, 3, 4, 5, 6]
#         },
#         'mapping_size': {
#             'values': [32,64]
#         },
#         'activation': {
#             'values': ['tanh', 'sigmoid', 'swish', 'gelu']
#         },
#         'num_nodes': {
#             'values': [64, 128, 256]
#         },
#         'sigma': {
#             'values': [0.1, 0.2, 0.3, 0.4, 0.5]
#         },
#         'scale': {
#             'values': [
#                 [1, 10], 
#                 [1, 10, 20], 
#                 [1, 10, 20, 50], 
#                 [1, 20, 50, 100]
#             ]
#         },
#         'optimizer': {
#             'values': ['adam', 'rmsprop', 'adamw', 'nadam']
#         }
#     }
# }
# # Create the sweep
# sweep_id = wandb.sweep(sweep=sweep_config, project='LLE_PINN')

# # Start the sweep agent
# wandb.agent(sweep_id, function= train, count=60)  # Adjust the count as needed for the number of runs
# wandb.finish()
