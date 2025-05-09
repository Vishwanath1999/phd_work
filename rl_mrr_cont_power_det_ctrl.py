# %%
import torch
import numpy as np
from scipy.io import loadmat, savemat
from scipy import constants as cts
# import torch.types
from tqdm import tqdm
# from ipywidgets import interact, widgets
import matplotlib.pyplot as plt
import pandas as pd
# from fastdtw import fastdtw
# from scipy.spatial.distance import euclidean
# from numba import njit, prange
import wandb

DEVICE = 'cpu'
C0 = 299792458
H_BAR = cts.hbar
# %%
# @njit(fastmath=True)
# def euclidean_distance(x, y):
#     return np.abs(x - y)

# @njit(fastmath=True)
# def dtw_cost_matrix(x, y, radius):
#     """Computes DTW cost matrix using a windowed approach for efficiency."""
#     n, m = len(x), len(y)
#     cost_matrix = np.full((n + 1, m + 1), np.inf, dtype=np.float64)
#     cost_matrix[0, 0] = 0

#     for i in prange(1, n + 1):  # Parallelizing outer loop
#         start_j = max(1, i - radius)
#         end_j = min(m + 1, i + radius + 1)

#         for j in range(start_j, end_j):
#             cost = euclidean_distance(x[i - 1], y[j - 1])
#             cost_matrix[i, j] = cost + min(
#                 cost_matrix[i - 1, j],  # Insertion
#                 cost_matrix[i, j - 1],  # Deletion
#                 cost_matrix[i - 1, j - 1]  # Match
#             )

#     return cost_matrix[n, m]

# @njit(fastmath=True)
# def downsample(sequence):
#     """Downsamples a sequence by averaging adjacent elements."""
#     length = len(sequence)
#     half_size = length // 2

#     result = np.empty(half_size, dtype=np.float64)
    
#     for i in prange(half_size):
#         result[i] = (sequence[2 * i] + sequence[2 * i + 1]) / 2

#     return result

# @njit(fastmath=True)
# def fast_dtw(x, y, radius=500):
#     """Computes FastDTW with multi-core acceleration."""
#     if len(x) < radius or len(y) < radius:
#         return dtw_cost_matrix(x, y, radius)
    
#     # Downsample (coarse resolution)
#     x_shrink = downsample(x) if len(x) % 2 == 0 else downsample(x[:-1])
#     y_shrink = downsample(y) if len(y) % 2 == 0 else downsample(y[:-1])

#     # Recursive call
#     cost = fast_dtw(x_shrink, y_shrink, radius)

#     return cost
# %%
class RL_MRR_Env():

    def __init__(self, seq_len=50):
        super(RL_MRR_Env, self).__init__()

        self.step_cntr = 0
        
        self.disp = loadmat('disp.mat')
        self.res = loadmat('res.mat')
        self.sim = loadmat('sim.mat')


        self.sim['DKS_init'] = np.array([complex(x.strip()) for x in self.sim['DKS_init']])
        # Convert the complex numbers to np.complex64
        self.sim['DKS_init'] = self.sim['DKS_init'].astype(np.complex128)

        myDKS = loadmat('My_DKS_init_a3c.mat')
        self.sim['DKS_init'] =  myDKS['DKS_init'][0]

        self.sim['domega_init'] = myDKS['detuning']
        self.sim['domega_end'] = myDKS['detuning_end']

        self.disp_tensor = self.dict_to_tensor(self.disp)
        self.res_tensor = self.dict_to_tensor(self.res)
        self.sim.pop('domega', None)
        self.sim['domega'] = np.array(['None',0])
        self.sim_tensor = self.dict_to_tensor(self.sim)

        self.disp_tensor['D1'] = self.disp_tensor['D1'][0]
        self.disp_tensor['FSR'] = self.disp_tensor['FSR'][0]
        self.disp_tensor['FSR_center'] = self.disp_tensor['FSR_center'][0]

        for key in self.res_tensor:
            self.res_tensor[key] = self.res_tensor[key][0]

        # if ndim>1 then squeeze the tensor
        for key in self.sim_tensor:
            if self.sim_tensor[key].ndim > 1:
                self.sim_tensor[key] = self.sim_tensor[key][0]
        
        ng = self.disp_tensor['ng']
        R = self.res_tensor['R']
        gamma = self.res_tensor['gamma']
        self.gamma = gamma
        L = 2*torch.pi*R
        self.L = L

        Q0 = self.res_tensor['Qi']
        Qc = self.res_tensor['Qc']
        fpmp = self.sim_tensor['f_pmp']
        Ppmp = self.sim_tensor['Pin']
        phi_pmp = self.sim_tensor['phi_pmp']
        self.phi_pmp = phi_pmp
        num_probe = self.sim_tensor['num_probe']
        num_probe = num_probe[0].cpu().numpy().astype(int)
        fcenter = self.sim_tensor['f_center']

        D1 = self.disp_tensor['D1']
        FSR = D1/(2*torch.pi)
        omega0 = 2*torch.pi*fpmp
        omega_center = 2*torch.pi*fcenter

        tR = 1/FSR
        self.tR = tR
        T = 1*tR
        kext = (omega0[0]/Qc) * tR
        self.kext = kext
        k0 = (omega0[0]/Q0) * tR
        alpha = k0+kext
        self.alpha = alpha

        del_omega_init = self.sim_tensor['domega_init']
        self.del_omega_init = del_omega_init
        self.current_del_omega = del_omega_init
        del_omega_end = self.sim_tensor['domega_end']
        self.del_omega_end = del_omega_end

        del_omega_stop = self.sim_tensor['domega_stop']
        ind_sweep = self.sim_tensor['ind_pump_sweep'] 
        t_end = self.sim_tensor['Tscan']
        Dint = self.disp_tensor['Dint_new']
        self.Dint = Dint

        DKSinit_real = torch.real(self.sim_tensor['DKS_init'])
        if self.sim_tensor['DKS_init'].dtype == torch.complex128 or self.sim_tensor['DKS_init'].dtype == torch.complex64:
            DKSinit_imag = torch.imag(self.sim_tensor['DKS_init'])
        else:
            DKSinit_imag = torch.zeros_like(DKSinit_real, device=DEVICE)

        self.DKS_init = torch.complex(DKSinit_real, DKSinit_imag)
        del_omega = self.sim['domega']
        ind_pmp = [ii for ii in self.sim_tensor['ind_pmp'].int().cpu().numpy()]    
        self.ind_pmp = ind_pmp
        mu_sim = self.sim_tensor['mucenter']
        mu = torch.arange(mu_sim[0], mu_sim[1]+1, device=DEVICE)
        self.mu= mu
        # find center of mu
        mu0 = torch.where(mu == 0)[0][0].int().cpu().numpy()+1
        self.mu0 = mu0

        d_omega = 2*torch.pi*FSR * torch.arange(mu_sim[0], mu_sim[-1]+1, device=DEVICE)
        domega_pmp1 = 2*torch.pi*FSR * torch.arange(mu_sim[0]-ind_pmp[0]-1, mu_sim[-1]-ind_pmp[0], device=DEVICE)
        omega1 = omega0[0] + domega_pmp1

        self.Dint = Dint-Dint[mu0-1]
        Ptot = 0#torch.zeros(1, device=device)
        for ii in range(len(fpmp)):
            Ptot += Ppmp[ii]
        
        # fpmp = fpmp[0]
        Ain = torch.zeros(len(fpmp), len(mu),dtype=torch.complex128, device=DEVICE)
        self.Ain = Ain
        Ein = torch.zeros(len(fpmp), len(mu),dtype=torch.complex128, device=DEVICE)
        self.Ein = Ein

        for ii in range(len(fpmp)):
            Ein[ii,int(mu0+ind_pmp[ii])] = torch.sqrt(Ppmp[ii])*len(mu)
            Ain[ii] = torch.fft.ifft(torch.fft.fftshift(Ein[ii],dim=0),dim=0)*torch.exp(-1j*phi_pmp[ii])
        
        self.Ain = Ain

        self.Ptot_dist = torch.distributions.uniform.Uniform(0.01, 0.2)

        self.Dint_shift = torch.fft.ifftshift(self.Dint)

        dt = 1
        t_end  = self.sim_tensor['Tscan']*tR
        t_ramp = t_end
        self.Nt = int(3e5)
        self.max_steps = int(3.5e5)

        self.del_omega_0 = del_omega_init + (1/self.Nt)*(del_omega_end - del_omega_init)

        del_omega_tot = torch.abs(del_omega_end)+torch.abs(del_omega_init)
        del_omega_perc = -1*torch.sign(del_omega_end+del_omega_init)*(torch.abs(del_omega_end+del_omega_init)/2)/del_omega_tot
        self.t_sim = torch.linspace(-t_ramp[0]/2 + del_omega_perc[0]*t_ramp[0], t_ramp[0]/2 + del_omega_perc[0]*t_ramp[0], self.Nt, device=DEVICE, dtype=torch.float64)
        self.t_sim_start = -t_ramp[0]/2 + del_omega_perc[0]*t_ramp[0]
        self.t_sim_step = self.t_sim[1] - self.t_sim[0]

        self.pcav_ref = loadmat('ref_check.mat')['Pcomb'].T
        self.primary_sidebands = loadmat('primary_sidebands.mat')['spec'][0]
        # self.pcav_ref = loadmat('Pcomb_rl_allv2.mat')['Pcomb'].T
        self.seq_len = seq_len

    
    def Fdrive(self, del_omega_all, t_sim, Ain, ind_pmp):
        Force = torch.zeros(len(self.mu), device=DEVICE, dtype=torch.complex128)
        for ii in range(len(ind_pmp)):
            if ii > 0:
                sigma = (2*del_omega_all + self.Dint[(self.mu0)+ind_pmp[ii]] - 0.5*self.del_omega_0)*t_sim
            else:
                sigma=torch.zeros(1, device=DEVICE)
            Force = Force - 1j*Ain[ii]*torch.exp(1j*sigma)
        return Force

    def dict_to_tensor(self,dic):
        dic.pop('__header__', None)
        dic.pop('__version__', None)
        dic.pop('__globals__', None)
        dic.pop('dispfile', None)

        tensor_dic = dict()
        # convert the dictionary to a tensor
        for key in dic:
            if isinstance(dic[key], np.ndarray):
                if key == 'DKS_init':
                    tensor_dic[key] = torch.tensor(dic[key], device=DEVICE, dtype=torch.complex64)
                else:
                    dic[key] = np.where(dic[key] == 'None', None, dic[key])
                    dic[key] = dic[key].astype(np.float64)
                    tensor_dic[key] = torch.tensor(dic[key], device=DEVICE)
            else:
                tensor_dic[key] = dic[key]
        return tensor_dic
    
    def tensor_to_dict(self,tensor):
        dic = dict()
        for key in tensor:
            # print(key)
            if isinstance(tensor[key], torch.Tensor):
                if tensor[key].dtype == torch.complex128:
                    dic[key] = tensor[key].cpu().numpy().astype(np.complex128)
                else:   
                    dic[key] = tensor[key].cpu().numpy().astype(np.float64)
            else:
                dic[key] = tensor[key]
        return dic
    
    @staticmethod
    @torch.jit.script
    def FFT_Lin(alpha: torch.Tensor, Dint_shift: torch.Tensor, del_omega_all: torch.Tensor, tR: torch.Tensor) -> torch.Tensor:
        '''
        Linear operator
        Input:
            it (int) : Time index
            alpha (torch.Tensor) : Linewidth enhancement factor
            Dint_shift (torch.Tensor) : Dispersive operator
            del_omega_all (torch.Tensor) : Detuning
            tR (torch.Tensor) : Round trip time
        Output:
            torch.Tensor : Linear operator
        '''
        return (-alpha / 2) + 1j * (Dint_shift - del_omega_all) * tR


    # Function for the Nonlinear operator
    @staticmethod
    @torch.jit.script
    def NL(uu: torch.Tensor, gamma: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
        '''
        Nonlinear operator
        Input:
            uu (torch.Tensor) : Input field
            gamma (torch.Tensor) : Nonlinear coefficient
            L (torch.Tensor) : Length of the resonator
        Output:
            torch.Tensor : Nonlinear operator
        '''
        return -1j * (gamma * L * torch.square(uu) )
    
    # @staticmethod
    # @torch.jit.script
    def ssfm_step(self,A0: torch.Tensor, it: int, alpha: torch.Tensor, Dint_shift: torch.Tensor,
                del_omega_all: torch.Tensor, tR: torch.Tensor, gamma: torch.Tensor, L: torch.Tensor, 
                max_iter: int, tol: float, dt: int, kext: torch.Tensor, Fdrive_val:torch.Tensor,
                ) -> torch.Tensor:
        
        A0 = A0 + Fdrive_val * torch.sqrt(kext) * dt
        L_h_prop = self.FFT_Lin(alpha, Dint_shift, del_omega_all, tR) * dt / 2
        L_h_prop = torch.exp(L_h_prop)
        A_L_h_prop = torch.fft.ifft(torch.fft.fft(A0) * L_h_prop)
        NL_h_prop_0 = self.NL(A0, gamma, L)
        A_h_prop = A0#.clone()
        A_prop = 0*A0.clone()
        # torch.zeros_like(A0, dtype=torch.complex128, device=A0.device)

        for _ in range(max_iter):
            err=0
            NL_h_prop_1 = self.NL(torch.abs(A_h_prop), gamma, L)
            NL_prop = (NL_h_prop_0 + NL_h_prop_1) * dt / 2
            A_prop = torch.fft.ifft(torch.fft.fft(A_L_h_prop * torch.exp(NL_prop)) * L_h_prop)
            err = torch.linalg.vector_norm(A_prop - A_h_prop, ord=2, dim=0) / torch.linalg.vector_norm(A_h_prop, ord=2, dim=0)
            if err < tol:
                return A_prop
            A_h_prop = A_prop
        err = torch.linalg.vector_norm(A_prop - A_h_prop, ord=2, dim=0) / torch.linalg.vector_norm(A_h_prop, ord=2, dim=0)
        raise RuntimeError(f"Convergence Error: {err}")
    
    # @staticmethod
    # @njit(fastmath=True)
    # def dtw_numba(x, y):
    #     n, m = len(x), len(y)
    #     dtw_matrix = np.full((n + 1, m + 1), np.inf)
    #     dtw_matrix[0, 0] = 0

    #     for i in prange(1, n + 1):
    #         for j in range(1, m + 1):
    #             cost = abs(x[i - 1] - y[j - 1])
    #             dtw_matrix[i, j] = cost + min(
    #                 dtw_matrix[i - 1, j],    # Insertion
    #                 dtw_matrix[i, j - 1],    # Deletion
    #                 dtw_matrix[i - 1, j - 1] # Match
    #             )

    #     return dtw_matrix[n, m]
    
    def reset(self, steps=None):
        self.state = self.DKS_init
        self.current_del_omega = self.del_omega_init
        self.step_cntr = 0
        self.pcav_hist = []

        self.power = np.random.uniform(0.12, 0.16, size=(1,))
        # self.power = np.array([0.14])
        Ppmp = torch.tensor(self.power, dtype=torch.float64)
        # # fpmp = fpmp[0]
        # Ain = torch.zeros(1, len(self.mu),dtype=torch.complex128, device=DEVICE)
        # Ein = torch.zeros(1, len(self.mu),dtype=torch.complex128, device=DEVICE)

        for ii in range(len(Ppmp)):
            self.Ein[ii,int(self.mu0+self.ind_pmp[ii])] = torch.sqrt(Ppmp[ii])*len(self.mu)
            self.Ain[ii] = torch.fft.ifft(torch.fft.fftshift(self.Ein[ii],dim=0),dim=0)*torch.exp(-1j*self.phi_pmp[ii])
        
        # self.Ain = Ain
        if steps is not None:
            self.init_steps_ = steps
        else:
            self.init_steps_ = int(1.5e5)
        # for idx in tqdm(range(int(1.5e5)), ncols=120):
        self.ecav_state = []
        for idx in range(self.init_steps_):
            mul_factor = np.random.choice([1, -1, 0], p=[0.9,0.05,0.05])
            del_omega = self.current_del_omega + mul_factor*(1/self.Nt)*(self.del_omega_end - self.del_omega_init)

            Fdrive_val = self.Fdrive(del_omega, self.t_sim_start+self.step_cntr*self.t_sim_step, self.Ain, self.ind_pmp)
            u0 = self.ssfm_step(self.state, self.step_cntr, self.alpha, self.Dint_shift, del_omega, self.tR, self.gamma, \
                                self.L, 10, 1e-3, 1, self.kext, Fdrive_val)
            self.step_cntr += 1
            self.state = u0        
            self.current_del_omega = del_omega

            Acav = torch.sqrt(self.alpha/2)*self.state*np.exp(1j*torch.pi)/len(self.mu)
            Ecav = torch.fft.fftshift(torch.fft.fft(Acav))
            Ecav_dBm = 10*torch.log10(torch.abs(Ecav)**2)+30
            Ecav_dBm = torch.clamp(Ecav_dBm, min=-60, max=10)
            Acav_np = Acav.cpu().numpy()
            curr_pcav = np.sum(np.abs(Acav_np))
            self.pcav_hist.append(curr_pcav)
            if idx >= self.init_steps_ - self.seq_len:
                self.ecav_state.append(Ecav_dBm.cpu().numpy())

        self.primary_sidebands_flag = False
        
        print('Reset...')
        return self.state, Acav_np, np.array(self.ecav_state)
    
    def compute_reward(self, spectral_mse, pcav_mse, alpha=0.85):
        # if flip_weights:
        #     alpha = 1 - alpha
        # sum_ = abs(spectral_mse) + abs(pcav_mse)
        # # weight the components with respect to the sum
        # spectral_mse = spectral_mse/sum_
        # pcav_mse = pcav_mse/sum_
        reward = alpha*spectral_mse + (1-alpha)*pcav_mse
        return reward
    
    def step(self, state, action, power, desired_spectrum):
        Ppmp = torch.tensor(np.array([power]), dtype=torch.float64)
        
        for ii in range(1):
            self.Ein[ii,int(self.mu0+self.ind_pmp[ii])] = torch.sqrt(Ppmp[ii])*len(self.mu)
            self.Ain[ii] = torch.fft.ifft(torch.fft.fftshift(self.Ein[ii],dim=0),dim=0)*torch.exp(-1j*self.phi_pmp[ii])
        
        det_delta = action*(1/self.Nt)*(self.del_omega_end - self.del_omega_init)
        
        del_omega = self.current_del_omega + det_delta

        self.current_del_omega = del_omega

        Fdrive_val = self.Fdrive(del_omega, self.t_sim_start+self.step_cntr*self.t_sim_step, self.Ain, self.ind_pmp)
        u0 = self.ssfm_step(state, self.step_cntr, self.alpha, self.Dint_shift, del_omega, self.tR, self.gamma, \
                            self.L, 10, 1e-3, 1, self.kext, Fdrive_val)
        
        self.next_state = u0
        self.step_cntr += 1
        
        Acav = torch.sqrt(self.alpha/2)*u0*np.exp(1j*torch.pi)/len(self.mu)
        Ecav = torch.fft.fftshift(torch.fft.fft(Acav))

        Acav_np = Acav.numpy()
        curr_pcav = np.sum(np.abs(Acav_np))
        self.pcav_hist.append(curr_pcav)

        Ecav_dBm = 10*torch.log10(torch.abs(Ecav)**2)+30
        Ecav_dBm = torch.clamp(Ecav_dBm, min=-60, max=10)
        desired_spectrum_dBm = 10*torch.log10(torch.abs(desired_spectrum)**2)+30
        desired_spectrum_dBm = torch.clamp(desired_spectrum_dBm, min=-60, max=10)

        
        # pop the first element of ecav_state and append new Ecav_dBm
        self.ecav_state = np.concatenate((self.ecav_state[1:], Ecav_dBm.cpu().numpy()[np.newaxis,:]), axis=0)

        # Calculate the reward as the l2 error norm between the desired spectrum and the current spectrum
        # spectral_mse = -(torch.linalg.norm(Ecav-desired_spectrum, ord=2)/torch.linalg.norm(desired_spectrum, ord=2))
        # # rescale the spectral_mse to be between -1 and 0
        
        # pcav_mse = -(500*(curr_pcav - self.pcav_ref[self.step_cntr-1])**2)

        # if self.pcav_hist[0]-self.pcav_hist[-1]>0.1 and self.pcav_hist[-1] > 0: 
        #     self.alpha_ = 1-0.75
            # print('Flipped...')

        if self.step_cntr < 45000:
            reward = 10*np.mean(self.pcav_hist[-100:])#pcav_mse
        elif self.step_cntr >= 45000 and self.step_cntr < int(0.5*self.Nt):
            reward = 10*np.mean(self.pcav_hist[-100:]) + 1
        else:
            reward = 4*torch.corrcoef(torch.stack([desired_spectrum_dBm, Ecav_dBm]))[0,1].item() + 1
            # reward = np.array(reward).reshape(1,)
        
        if torch.linalg.vector_norm(desired_spectrum_dBm-Ecav_dBm, ord=2) < 50:
            achieved = True
            reward += 2
        else:
            achieved = False

        
        # finf correlation between pcav_hist and recent samples of pcav_ref
        # if self.step_cntr >= int(0.6*self.max_steps):
        #     corr = np.corrcoef(self.pcav_hist, self.pcav_ref[self.step_cntr-len(self.pcav_hist):self.step_cntr,0])[0,1]
        # else:
        #     corr = torch.corrcoef(torch.stack([desired_spectrum_dBm, Ecav_dBm]))[0,1]

        # pop the first element of the pcav_hist if it exceeds 10001
        done = False
        if len(self.pcav_hist) > 10000:
            self.pcav_hist.pop(0)
        
        if self.step_cntr<45000 and self.primary_sidebands_flag==False:
            self.primary_sidebands_flag = np.corrcoef(self.primary_sidebands, Ecav_dBm.cpu().numpy())[0,1]<0.5
        
        if self.step_cntr == 45000 and self.primary_sidebands_flag == False:
            done = True
            reward -= 10
            print('Primary Sidebands not formed')
            print('Corr:',np.corrcoef(self.primary_sidebands, Ecav_dBm.cpu().numpy())[0,1])
        elif self.step_cntr-self.init_steps_ >= int(0.5*self.Nt):
            if torch.corrcoef(torch.stack([desired_spectrum_dBm, Ecav_dBm]))[0,1].item() < 0.25:
                done = True
                reward -= 5
                # print('Pcav Corr:',corr)
                print('Spectral Corr:', torch.corrcoef(torch.stack([desired_spectrum_dBm, Ecav_dBm]))[0,1].item())
    
        if self.step_cntr+1 == self.max_steps:
            done = True
            # reward += 5
        
        return self.next_state, reward, done, achieved, Acav_np, self.ecav_state

# %%
# torch seed
# torch.manual_seed(0)
env = RL_MRR_Env(seq_len=50)
# %%
desired_spectrum = loadmat('desired_spec.mat')['Ecav'][0]
desired_spectrum_dBm = 10*np.log10(np.abs(desired_spectrum)**2)+30
desired_spectrum_tensor = torch.tensor(desired_spectrum, device=DEVICE, dtype=torch.complex128)
# %%
config = {
    'input_dim': [env.seq_len, 441+1],
    'n_actions': [1,1],
    'alpha': 3e-4,
    'beta': 3e-4,
    'mem_size': int(1e6),
    'run_name': 'mrr_masac_cluster',
    'batch_size': 128,
    'dist': 'normal',
    'log':True,
    'p_max':0.5,
    'p_min':0.001,
    'power_mod_steps':int(5e4)
    }
# %%

from masac import MASAC
agent = MASAC(alpha=config['alpha'], beta=config['beta'], input_dims=config['input_dim'], n_actions=config['n_actions'],
              mem_size=config['mem_size'], batch_size=config['batch_size'], 
              name=config['run_name'], eval_mode=not(torch.cuda.is_available()))
# %%
# # init wandb run
if config['log']:
    wandb.init(project='maddpg_mrr', entity='viswacolab-technical-university-of-denmark', config=config)
    wandb.watch(agent.actor, log='gradients', log_freq=1000)
    # set the wandb run name
    wandb.run.name = agent.run_name
# %% MADDPG train loop

scale_power = lambda v, x, y: x + ((v + 1) / 2) * (y - x)

def discounted_return(rewards, gamma=0.99):
    """
    Compute the full-episode discounted return:
       G = sum_{t=0..T-1} gamma^t * rewards[t]
    """
    T = len(rewards)
    # gamma powers: [1, γ, γ^2, …, γ^{T-1}]
    powers = gamma ** np.arange(T)
    return np.dot(powers, rewards)

# %%
# '''
logs={}
n_games = 40
global_n_steps = 0
scores = []
best_score = -np.inf
den = config['p_max']-config['p_min']
for i in range(n_games):
    score = 0
    done = False
    n_steps = 0
    state, acav, ecav = env.reset(10000)
    logs['env/initial pump power'] = env.power
    obs = np.concatenate((ecav/10,env.power*np.ones((50,1))/den),axis=1)
    r_hist = []
    while not done:
        action = agent.choose_action(obs)

        if n_steps%config['power_mod_steps'] == 0:
            power = scale_power(action[1], config['p_max'], config['p_min'])
            unscaled_power = action[1]
        
        next_state, reward, done, achieved, _, ecav_ = env.step(state, action[0], power, desired_spectrum_tensor)
        r_hist.append(reward)
        # log perf action
        logs['env/detuning'] = action[0]
        logs['env/pred. pump power'] = power
        if achieved==True:
            new_done = False
        else:
            new_done = done    
        logs['env/reward'] = reward  
        
        obs_ = np.concatenate((ecav_/10,env.power*np.ones((50,1))/den),axis=1)
        obs = obs_   
        agent.remember_detuning(obs, np.array([action[0], unscaled_power]), reward, obs_, new_done)
        if global_n_steps%config['power_mod_steps'] == 0:
            n_step_return = discounted_return(np.array(r_hist), gamma=0.99)
            agent.remember_power(obs, np.array([action[0], unscaled_power]), reward, obs_, new_done)
            r_hist=[]
        state = next_state
        ecav = ecav_
        score += reward
        n_steps += 1
        
        if agent.det_memory_ready:
            cl, al, ent_loss, ent_coeff = agent.learn()
            logs['model/critic_loss'] = cl
            logs['model/actor_loss'] = al
            logs['model/entropy_loss'] = ent_loss
            logs['model/entropy_coeff'] = ent_coeff
            # print('Critic loss:', cl, 'Actor loss:', al, 'Entropy loss:', ent_loss, 'Entropy coeff:', ent_coeff)

        if n_steps>int(0.5*env.Nt) and done==True:
            
            fig=plt.figure(figsize=(14,4))
            plt.vlines(np.arange(-220,221, 1), -60*np.ones(len(ecav[-1])), ecav[-1], \
                    label='Obtained Spectrum')
            plt.vlines(np.arange(-220,221, 1), -60*np.ones(len(desired_spectrum)),\
                        desired_spectrum_dBm, color='red', label='Desired Spectrum',alpha=0.5)
            plt.xlabel('Rel. Mode no.', fontsize=14)
            plt.ylabel('Power(dBm)', fontsize=14)
            plt.grid()
            plt.ylim(-90,5)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            plt.legend(fontsize=14)
            plt.title('Correlation '+str(np.round(np.corrcoef(ecav[-1], np.clip(desired_spectrum_dBm,-60,10))[0,1],2)), fontsize=14)
            plt.tight_layout()
            wandb.log({"ecav": wandb.Image(fig)})
            plt.close(fig)


        if global_n_steps%100 == 0:
            if config['log']:
                wandb.log(logs)

        global_n_steps += 1
    scores.append(score)
    avg_score = np.mean(scores[-5:])
    
    if avg_score > best_score:
        best_score = avg_score
        agent.save_models()

    print('episode ', i, 'score %.2f' % score, 'average score %.2f' % avg_score,'best score %.2f' % best_score, 'n_steps', n_steps)

if config['log']:
    wandb.finish()
# '''
# %%
# plt.figure(figsize=(10,6))
# plt.plot(scores)
# plt.grid()
# plt.xlabel('Episodes')
# plt.ylabel('Scores')
# plt.title('Scores vs Episodes')
# plt.savefig('./maddpg_results/'+agent.run_name+'_scores_vs_episodes.png')
# plt.show()
# %%
# agent_frozen = agent
# agent.load_models()
# # freeze the actor network
# for param in agent_frozen.actor.parameters():
#     param.requires_grad = False
# %%
state, acav, ecav = env.reset(10000)
den = config['p_max']-config['p_min']
obs = np.concatenate((ecav/10,env.power*np.ones((50,1))/den),axis=1)
print('Chosen power:', env.power)
r_hist = []
action_hist = []
acav_hist = []
score = 0
done = False
pcav_hist = []
pbar = tqdm(total=env.max_steps-env.init_steps_, ncols=120)
idx = 0
done = False
ecav_hist = []
achieved = False
power_change_flag = True
while not done:
# for idx in tqdm(range(env.init_steps_, int(1.5*env.max_steps)), ncols=120):
    # perform random actions
    # try:
        action = agent.choose_action(obs, True)
        if power_change_flag == True:
            power = scale_power(action[1], config['p_max'], config['p_min'])
            unscaled_power = action[1]
            power_change_flag = False
        # action = np.random.uniform(-1,1,size=(1,))
        # if achieved==True:
        #     action = np.array([0])
        # else:
        #     action = np.array([0.5])#np.random.choice([0, 1, 2], p=[1/3, 1/3, 1/3])

        next_state, reward, done, achieved, acav_, ecav_ = env.step(state, action[0], power, desired_spectrum_tensor)
        state = next_state
        ecav = ecav_
        obs_ = np.concatenate((ecav_/10,env.power*np.ones((50,1))/0.04),axis=1)
        obs = obs_
        score += reward
        curr_pcav = np.sum(np.abs(acav_))
        pcav_hist.append(curr_pcav)
        r_hist.append(reward)
        action_hist.append(action)
        # ecav_hist.append(ecav_[-1])
        if idx %100 == 0:        
            acav_hist.append(acav_)
        idx += 1
        pbar.update(1)
    # except KeyboardInterrupt:
    #     pbar.close()
    #     break
    # if keyboard interrupt then close the progress bar
pbar.close()

print('Test score %.2f' % score)
# %%
# plt.figure(figsize=(14, 4))
# plt.imshow(np.array(ecav_hist).T, aspect='auto', cmap='jet',\
#             extent=[0, len(ecav_hist), -1e12*env.tR.item()/2, 1e12*env.tR.item()/2])
# plt.colorbar(label='Power(dBm)')
# plt.xlabel('Tuning Steps')
# plt.ylabel(r'$\mu$' +'(rel)', fontsize=14)
# # plt.xticks(fontsize=14)
# # plt.yticks(fontsize=14)
# # plt.xlim(0,40000)
# plt.show()
env.power = np.round(env.power, 3)
# %%
import os

# Create save directory if not exists
save_dir = os.path.join('./results', agent.run_name)
os.makedirs(save_dir, exist_ok=True)
# %%
# find correlation between the obtained pcav and r_hist[:,-1]
plt.figure(figsize=(10, 6))
plt.plot(pcav_hist, label='Obtained')
plt.plot(env.pcav_ref[10000:], label='Reference')
# plt.xlim(25000,50000)
plt.grid()
plt.legend()
plt.xlabel('Iteration')
plt.ylabel('Pcav')
plt.title('Pump Power: '+str(env.power[0])+'mW', fontsize=16, fontweight='bold')
plt.tight_layout()
mod_pow = str(env.power[0]).replace('.','_')
if idx > int(0.5*env.max_steps):
    plt.savefig(os.path.join(save_dir, mod_pow + '_pcav_spec_all_ctrl.png'))
plt.show()

# %%
import matplotlib.ticker as ticker

plt.figure(figsize=(14,4))
plt.imshow(np.abs(1e3*np.array(acav_hist).T), aspect='auto', cmap='jet',\
            extent=[0, len(acav_hist), -1e12*env.tR.item()/2, 1e12*env.tR.item()/2])
plt.colorbar(label='Power(dBm)')
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
# Set x-ticks to exponent format
formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(True)
formatter.set_powerlimits((-1, 1))
plt.gca().xaxis.set_major_formatter(formatter)
plt.xlabel('Tuning Steps', fontsize=14)
plt.ylabel(r'$t_R (ps)$', fontsize=14)
plt.title('Pump Power: '+str(env.power[0])+'mW', fontsize=16, fontweight='bold')
mod_pow = str(env.power[0]).replace('.','_')
plt.tight_layout()
if idx > int(0.5*env.max_steps):
    plt.savefig(os.path.join(save_dir, mod_pow + '_ecav_hist_spec_all_ctrl.png'))
plt.show()

# %%
plt.figure(figsize=(14,4))
spectrum = np.fft.fftshift(np.fft.fft(np.array(acav_hist).T, axis=0), axes=0)
spectrum_dBm = 10*np.log10(np.abs(spectrum)**2)+30
spectrum_dBm = np.clip(spectrum_dBm, -60, 10)
plt.imshow(spectrum_dBm, aspect='auto', cmap='jet'\
            ,extent=[0, len(acav_hist), env.mu.min().item(), env.mu.max().item()])
plt.xlabel('Tuning Steps', fontsize=14)
plt.ylabel(r'$\mu$' +'(rel)', fontsize=14)
plt.colorbar(label='Power(dBm)')
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.title('Pump Power: '+str(env.power[0])+'mW', fontsize=16, fontweight='bold')
mod_pow = str(env.power[0]).replace('.','_')
plt.tight_layout()
if idx > int(0.5*env.max_steps):
    plt.savefig(os.path.join(save_dir, mod_pow + '_ecav_hist_ifft_spec_all_ctrl.png'))
plt.show()
# %% Reward Plot
plt.figure(figsize=(10, 6))
plt.plot(r_hist)
plt.xlabel('Iteration', fontsize=14)
plt.ylabel('Reward ', fontsize=14)
plt.grid()
plt.title('Pump Power: '+str(env.power[0])+'mW', fontsize=16, fontweight='bold')
plt.tight_layout()
mod_pow = str(env.power[0]).replace('.','_')
if idx > int(0.5*env.max_steps):
    plt.savefig(os.path.join(save_dir, mod_pow + '_rewards_spec_all_ctrl.png'))
plt.show()
# %%
# desired_spectrum_dBm = 10*torch.log10(torch.abs(desired_spectrum)**2)+30
plt.figure(figsize=(14,4))
plt.vlines(np.arange(-220,221, 1), -60*np.ones(len(ecav[-1])), ecav[-1], \
           label='Obtained Spectrum')
plt.vlines(np.arange(-220,221, 1), -60*np.ones(len(desired_spectrum)),\
            desired_spectrum_dBm, color='red', label='Desired Spectrum',alpha=0.5)
plt.xlabel('Rel. Mode no.', fontsize=14)
plt.ylabel('Power(dBm)', fontsize=14)
plt.grid()
plt.ylim(-90,5)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=14)
plt.title('Pump Power: '+str(env.power[0])+'mW', fontsize=16, fontweight='bold')
mod_pow = str(env.power[0]).replace('.','_')
plt.tight_layout()
if idx > int(0.5*env.max_steps):
    plt.savefig(os.path.join(save_dir, mod_pow + '_ecav_spec_all_ctrl.png'))
plt.show()
# %%
action_hist = np.array(action_hist)
plt.figure(figsize=(10, 6))
plt.plot(action_hist, label='GHz')
plt.xlabel('Iteration', fontsize=14)
plt.ylabel('Action', fontsize=14)
plt.legend()
plt.grid()
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.gca().xaxis.set_major_formatter(formatter)
plt.title('Pump Power: '+str(env.power[0])+'mW', fontsize=16, fontweight='bold')
mod_pow = str(env.power[0]).replace('.','_')
plt.tight_layout()
if idx > int(0.5*env.max_steps):
    plt.savefig(os.path.join(save_dir, mod_pow + '_actions_spec_all_ctrl.png'))
plt.show()

# plt.figure(figsize=(10, 6))
# plt.plot(action_hist[:,1])
# plt.xlabel('Iteration', fontsize=14)
# plt.ylabel('Pump Power', fontsize=14)
# plt.grid()
# plt.xticks(fontsize=14)
# plt.yticks(fontsize=14)
# plt.savefig('./maddpg_results/'+agent.run_name+'_power_all_ctrl.png')
# plt.show()

# %%
# import numpy as np
# import matplotlib.pyplot as plt

# x = np.random.rand(128)
# # take softmax of x
# x_softmax = np.exp(x) / np.sum(np.exp(x), axis=0)
# # plot histogram of x_softmax using numpy histogram
# x_hist = np.histogram(x_softmax, bins=10, density=True)
# plt.figure(figsize=(10, 6))
# plt.bar(x_hist[1][:-1], x_hist[0], width=np.diff(x_hist[1]), align='edge', edgecolor='black')
# plt.xlabel('Value', fontsize=14)
# plt.ylabel('Frequency', fontsize=14)
# plt.title('Histogram of Softmax Values', fontsize=16, fontweight='bold')
# plt.show()
# %%
