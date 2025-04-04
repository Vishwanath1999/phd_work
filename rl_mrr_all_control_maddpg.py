# %%
import torch
import numpy as np
from scipy.io import loadmat, savemat
from scipy import constants as cts
import torch.types
from tqdm import tqdm
from ipywidgets import interact, widgets
import matplotlib.pyplot as plt
import pandas as pd
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from numba import njit, prange
import wandb

DEVICE = 'cpu'
C0 = 299792458
H_BAR = cts.hbar
# %%
@njit(fastmath=True)
def euclidean_distance(x, y):
    return np.abs(x - y)

@njit(fastmath=True)
def dtw_cost_matrix(x, y, radius):
    """Computes DTW cost matrix using a windowed approach for efficiency."""
    n, m = len(x), len(y)
    cost_matrix = np.full((n + 1, m + 1), np.inf, dtype=np.float64)
    cost_matrix[0, 0] = 0

    for i in prange(1, n + 1):  # Parallelizing outer loop
        start_j = max(1, i - radius)
        end_j = min(m + 1, i + radius + 1)

        for j in range(start_j, end_j):
            cost = euclidean_distance(x[i - 1], y[j - 1])
            cost_matrix[i, j] = cost + min(
                cost_matrix[i - 1, j],  # Insertion
                cost_matrix[i, j - 1],  # Deletion
                cost_matrix[i - 1, j - 1]  # Match
            )

    return cost_matrix[n, m]

@njit(fastmath=True)
def downsample(sequence):
    """Downsamples a sequence by averaging adjacent elements."""
    length = len(sequence)
    half_size = length // 2

    result = np.empty(half_size, dtype=np.float64)
    
    for i in prange(half_size):
        result[i] = (sequence[2 * i] + sequence[2 * i + 1]) / 2

    return result

@njit(fastmath=True)
def fast_dtw(x, y, radius=500):
    """Computes FastDTW with multi-core acceleration."""
    if len(x) < radius or len(y) < radius:
        return dtw_cost_matrix(x, y, radius)
    
    # Downsample (coarse resolution)
    x_shrink = downsample(x) if len(x) % 2 == 0 else downsample(x[:-1])
    y_shrink = downsample(y) if len(y) % 2 == 0 else downsample(y[:-1])

    # Recursive call
    cost = fast_dtw(x_shrink, y_shrink, radius)

    return cost
# %%
class RL_MRR_Env():

    def __init__(self):
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
        Nt = int(3e5)
        self.max_steps = Nt

        self.del_omega_0 = del_omega_init + (1/Nt)*(del_omega_end - del_omega_init)

        del_omega_tot = torch.abs(del_omega_end)+torch.abs(del_omega_init)
        del_omega_perc = -1*torch.sign(del_omega_end+del_omega_init)*(torch.abs(del_omega_end+del_omega_init)/2)/del_omega_tot
        self.t_sim = torch.linspace(-t_ramp[0]/2 + del_omega_perc[0]*t_ramp[0], t_ramp[0]/2 + del_omega_perc[0]*t_ramp[0], Nt, device=DEVICE, dtype=torch.float64)

        self.pcav_ref = loadmat('Pcomb_rl_allv2.mat')['Pcomb'].T

    
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
    def FFT_Lin(it: int, alpha: torch.Tensor, Dint_shift: torch.Tensor, del_omega_all: torch.Tensor, tR: torch.Tensor) -> torch.Tensor:
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
        return -1j * (gamma * L * torch.square(torch.abs(uu)) )
    
    # @staticmethod
    # @torch.jit.script
    def ssfm_step(self,A0: torch.Tensor, it: int, alpha: torch.Tensor, Dint_shift: torch.Tensor,
                del_omega_all: torch.Tensor, tR: torch.Tensor, gamma: torch.Tensor, L: torch.Tensor, 
                max_iter: int, tol: float, dt: int, kext: torch.Tensor, Fdrive_val:torch.Tensor,
                ) -> torch.Tensor:
        
        A0 = A0 + Fdrive_val * torch.sqrt(kext) * dt
        L_h_prop = torch.exp(self.FFT_Lin(it, alpha, Dint_shift, del_omega_all, tR) * dt / 2)
        A_L_h_prop = torch.fft.ifft(torch.fft.fft(A0) * L_h_prop)
        NL_h_prop_0 = self.NL(A0, gamma, L)
        A_h_prop = A0#.clone()
        A_prop = 0*A0.clone()
        # torch.zeros_like(A0, dtype=torch.complex128, device=A0.device)

        for _ in range(max_iter):
            err=0
            NL_h_prop_1 = self.NL(A_h_prop, gamma, L)
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


        Ppmp = self.Ptot_dist.sample().unsqueeze(0)
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
        for _ in range(self.init_steps_):
            mul_factor = np.random.choice([1, -1, 0], p=[1/3, 1/3, 1/3])
            del_omega = self.current_del_omega + mul_factor*(1/self.max_steps)*(self.del_omega_end - self.del_omega_init)

            Fdrive_val = self.Fdrive(del_omega, self.t_sim[self.step_cntr], self.Ain, self.ind_pmp)
            u0 = self.ssfm_step(self.state, self.step_cntr, self.alpha, self.Dint_shift, del_omega, self.tR, self.gamma, \
                                self.L, 10, 1e-3, 1, self.kext, Fdrive_val)
            self.step_cntr += 1
            self.state = u0        
            self.current_del_omega = del_omega

            Acav = torch.sqrt(self.alpha/2)*self.state*np.exp(1j*torch.pi)/len(self.mu)
            Ecav = torch.fft.fftshift(torch.fft.fft(Acav))
            Ecav_dBm = 10*torch.log10(torch.abs(Ecav)**2)+30
            Ecav_dBm = torch.clamp(Ecav_dBm, min=-60, max=10)
            Acav_np = Acav.numpy()
            curr_pcav = np.sum(np.abs(Acav_np))
            self.pcav_hist.append(curr_pcav)

        self.grad_0 = False
        self.grad_neg= False
        
        print('Reset...')
        return self.state, Acav_np, Ecav_dBm.numpy()

    def compute_reward(self,dtw_distance, mse, x1=50, gamma=0.1, k=10):

        # Dynamic weight transition
        if dtw_distance > x1:
            alpha = 1.0  # Focus more on DTW initially
            beta = 0.35  # MSE has lower importance
        else:
            weight_shift = 1 - (dtw_distance / x1)  # Smooth transition
            alpha = 1 - weight_shift  # Reduce DTW importance
            beta = 0.35 + 0.65 * weight_shift  # Increase MSE importance

        # Stability term: variance over last k steps
        stability = -np.var(self.pcav_hist[-k:]) if len(self.pcav_hist[-k:]) >= k else 0
        # Final reward
        reward = - (alpha * dtw_distance + beta * mse) + gamma * stability
        return reward
    
    def compute_reward_grad(self,grad_neg,grad_0,mse,corr,alpha=0.25):

        # ig grad_neg and grad_0 are True then alpha=1-alpha
        if grad_neg==True and grad_0==True:
            alpha = 1-alpha
        
        reward = (1-alpha)*corr + alpha*mse
        return reward
    
    def step(self, state, action, desired_spectrum):
        Ppmp = torch.tensor(action[1:], dtype=torch.float64)
        

        for ii in range(1):
            self.Ein[ii,int(self.mu0+self.ind_pmp[ii])] = torch.sqrt(Ppmp[ii])*len(self.mu)
            self.Ain[ii] = torch.fft.ifft(torch.fft.fftshift(self.Ein[ii],dim=0),dim=0)*torch.exp(-1j*self.phi_pmp[ii])
        
        # self.Ain = Ain

        if int(action[0]) == 0:
            del_omega = self.current_del_omega - (1/self.max_steps)*(self.del_omega_end - self.del_omega_init)
        elif int(action[0]) == 1:
            del_omega = self.current_del_omega + (1/self.max_steps)*(self.del_omega_end - self.del_omega_init)
        else:
            del_omega = self.current_del_omega

        Fdrive_val = self.Fdrive(del_omega, self.t_sim[self.step_cntr], self.Ain, self.ind_pmp)
        u0 = self.ssfm_step(state, self.step_cntr, self.alpha, self.Dint_shift, del_omega, self.tR, self.gamma, \
                            self.L, 10, 1e-3, 1, self.kext, Fdrive_val)
        
        
        self.current_del_omega = del_omega
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

        if torch.allclose(Ecav_dBm, desired_spectrum_dBm, atol=1):
            achieved = True
        else:
            achieved = False
        

        # Calculate the reward as the l2 error norm between the desired spectrum and the current spectrum
        mse = -(torch.linalg.norm(Ecav-desired_spectrum, ord=2)/torch.linalg.norm(desired_spectrum, ord=2))
        # seq_len = 3000
        # if self.step_cntr > self.init_steps_ + 10:
        #     if self.step_cntr - self.init_steps_ < seq_len:
        #         distance = self.dtw_numba(self.pcav_hist, self.pcav_ref[self.init_steps_:self.step_cntr,0])
        #         distance  = distance/10
        #     else:
        #         # downsample the self.pcav_hist and self.pcav_ref[self.init_steps_:self.step_cntr] to 3000 points
        #         # and then compute the distance
        #         # Downsample to 3000 points
        #         num_points = seq_len
        #         indices = np.linspace(0, len(self.pcav_hist) - 1, num_points).astype(int)
        #         downsampled_pcav_hist = np.array(self.pcav_hist)[indices]
        #         # downsampled_pcav_hist = downsampled_pcav_hist[:,np.newaxis]
        #         downsampled_pcav_ref = np.array(self.pcav_ref[self.init_steps_:self.step_cntr,0])[indices]
                
        #         # Compute the distance
        #         distance = self.dtw_numba(downsampled_pcav_hist, downsampled_pcav_ref)
        #         distance = distance / 10
        # else:
        #     distance = -5
        #self.compute_reward(distance/10, mse.numpy(), x1=10, gamma=0)
        # reward = - torch.linalg.norm((Ecav_dBm-desired_spectrum_dBm)/500, ord=2)
        recent_samples = min(20000, len(self.pcav_hist))
        corrcoef = np.corrcoef(self.pcav_hist[-recent_samples:],self.pcav_ref[min(self.step_cntr,self.max_steps)-recent_samples:min(self.step_cntr,self.max_steps),0])[0,1]

        # if self.step_cntr <= 0.50*self.max_steps:
        #     reward = 0.75*2*corrcoef + 0.25*mse.numpy()
        # elif self.step_cntr > 0.50*self.max_steps and self.step_cntr <= 0.75*self.max_steps:
        #     reward = 0.5*2*corrcoef + 0.5*mse.numpy()
        # else:
        #     reward = 0.25*2*corrcoef + 0.75*mse.numpy()
        
        # if self.step_cntr == 0.5*self.max_steps or self.step_cntr == 0.75*self.max_steps:
        #     distance = fast_dtw(self.pcav_hist/np.max(self.pcav_hist),\
        #                         self.pcav_ref[self.init_steps_:self.step_cntr,0]/np.max(self.pcav_ref[self.init_steps_:self.step_cntr,0]))
        #     if distance > 20:
        #         achieved =True
        #     else:
        #         achieved = False

        # if self.step_cntr >= self.max_steps//3:
        grad = np.gradient(self.pcav_hist[-recent_samples:])
        if np.mean(grad) < 0:
            self.grad_neg = True
            # print(self.step_cntr, 'Negative gradient')
        else:
            self.grad_neg = False
        
        if self.grad_neg == True and np.allclose(np.mean(grad), 0, atol=1e-2) and curr_pcav>0.05*np.max(self.pcav_hist):
            self.grad_0 = True
            # print(self.step_cntr, 'Zero gradient')
        else:
            self.grad_0 = False        
        reward = self.compute_reward_grad(self.grad_neg, self.grad_0, 1*mse.numpy(), 2*corrcoef)
        if (self.step_cntr > self.max_steps):
            print('Done...')
            done = True
        else:
            done = False
        
        if self.step_cntr >= int(0.6*self.max_steps):
            if Ecav_dBm.max() < -40 or self.grad_neg == False:
                done = True
            else:
                done = False
        else:
            done = False
        
        return self.next_state, reward, done, achieved, Acav_np, desired_spectrum_dBm.numpy()

# %%
# torch seed
# torch.manual_seed(0)
env = RL_MRR_Env()

# state, acav,ecav = env.reset()
# plt.plot(ecav)
# plt.show()
# %%
# state, acav = env.reset()
# s = (torch.sqrt(env.alpha/2)*state).cpu().numpy()*np.exp(1j*torch.pi)/len(env.mu)
# del_omega=[]
# del_omega.append(env.current_del_omega.cpu().numpy())
# %%
desired_spectrum = loadmat('desired_spec.mat')['Ecav'][0]
desired_spectrum = torch.tensor(desired_spectrum, device=DEVICE, dtype=torch.complex128)
# %%

from maddpg import MADDPG
agent = MADDPG(alpha=1e-4, beta=1e-4, input_dims=[441], disc_n_actions=3, cont_n_actions=1)
# %%
# init wandb run
wandb.init(project='maddpg_mrr', entity='viswacolab-technical-university-of-denmark')
wandb.watch(agent.agents[0].actor, log='all', log_freq=1000)
wandb.watch(agent.agents[0].critic, log='all', log_freq=1000) 
# %% MADDPG train loop
logs={}
n_games = 1000
# r_hist = []
# scaled_r_hist = []
global_n_steps = 0
scores = []
best_score = -np.inf
for i in range(n_games):
    score = 0
    done = False
    n_steps = 0
    state, acav, ecav = env.reset(10000)
    agent.reset_action_noise()
    while not done:
        action = agent.choose_action(ecav/10)
        # print(action)
        a, b = 0.001, 0.5  # Define action bounds
        scaled_action = a + 0.5 * (action[0] + 1) * (b - a)
        scaled_action = np.clip(scaled_action, a, b)
        perf_action = np.array([np.argmax(action[1]), scaled_action[0]])
        next_state, reward, done, achieved, _, ecav_ = env.step(state, perf_action, desired_spectrum)
        # log perf action
        logs['detuning'] = perf_action[0]
        logs['power'] = perf_action[1]
        if achieved==True:
            new_done = False
            done = True
        else:
            new_done = done    
        logs['reward'] = reward      
        agent.remember(ecav/10,np.concatenate((action[0], action[1])), reward, ecav_/10, new_done)
        state = next_state
        ecav = ecav_
        score += reward
        # r_hist.append(reward)
        n_steps += 1
        global_n_steps += 1
        if global_n_steps > 4*agent.batch_size:
            # print('Training...')
            al, cl = agent.learn()
            for idx, loss in enumerate(al):
                logs[f'actor_loss_{idx}'] = loss
            for idx, loss in enumerate(cl):
                logs[f'critic_loss_{idx}'] = loss

        if global_n_steps%100 == 0:
            wandb.log(logs)

    scores.append(score)
    avg_score = np.mean(scores[-100:])
    
    print('episode ', i, 'score %.2f' % score, 'average score %.2f' % avg_score, 'n_steps', n_steps)
    if avg_score > best_score:
        best_score = avg_score
        agent.save_models()
# %% save model
# torch.save(agent.actor.state_dict(), './rl_results/actor_power_det_ctrl.pth')
# save the r_hist into a csv file
# ...existing code...

# import pandas as pd

# # Save the r_hist and Pcav_optimum into a CSV file
# r_hist_df = pd.DataFrame({
#     'reward': r_hist[:,-1],
#     'Pcav_optimum': np.array(pcav_hist)  # Ensure both columns have the same length
# })
# r_hist_df.to_csv('r_hist2.csv', index=False)

# %%
plt.figure(figsize=(10,6))
plt.plot(scores)
plt.grid()
plt.xlabel('Episodes')
plt.ylabel('Scores')
plt.title('Scores vs Episodes')
plt.savefig('./maddpg_results/scores_vs_episodes.png')
plt.show()
# %%
agent_frozen = agent
# # freeze the actor network
# for param in agent_frozen.actor.parameters():
#     param.requires_grad = False
state, acav, ecav = env.reset(10000)
r_hist = []
action_hist = []
acav_hist = []
score = 0
done = False
pcav_hist = []
pbar = tqdm(total=env.max_steps-env.init_steps_, ncols=120)
idx = 0
done = False
while not done:
# for idx in tqdm(range(env.init_steps_, env.max_steps), ncols=120):
    # perform random actions
    try:
        det = 1#np.random.choice([0, 1, 2], p=[1/3, 1/3, 1/3])
        pow_ = 0.14#np.random.uniform(0.001, 0.5)
        perf_action = np.array([det, pow_])

        # action=agent_frozen.choose_action(ecav/10, deterministic=True)
        # a, b = 0.001, 0.5  # Define action bounds
        # scaled_action = a + 0.5 * (action[0] + 1) * (b - a)
        # scaled_action = np.clip(scaled_action, a, b)
        # perf_action = np.array([action[1], scaled_action[0]])

        next_state, reward, done, achieved, acav_, ecav_ = env.step(state, perf_action, desired_spectrum)
        state = next_state
        ecav = ecav_
        score += reward
        curr_pcav = np.sum(np.abs(acav_))
        pcav_hist.append(curr_pcav)
        r_hist.append(reward)
        action_hist.append(perf_action)
        if idx %100 == 0:        
            acav_hist.append(acav_)
        idx += 1
        pbar.update(1)
    except KeyboardInterrupt:
        pbar.close()
        break
    # if keyboard interrupt then close the progress bar
pbar.close()

print('Test score %.2f' % score)
# %%
# find correlation between the obtained pcav and r_hist[:,-1]
plt.figure(figsize=(10, 6))
plt.plot(env.pcav_hist, label='Obtained')
plt.plot(env.pcav_ref, label='Reference')
plt.grid()
plt.legend()
plt.xlabel('Iteration')
plt.ylabel('Pcav')
plt.show()

# %%
# from fastdtw import fastdtw
# from scipy.spatial.distance import euclidean
# pcav_ref = loadmat('Pcomb_opt_rl.mat')['Pcomb'].T
# mag = np.abs(np.array(acav_hist).T)
# pcav = np.sum(mag, axis=0, keepdims=True).T
# # print(pcav.shape, pcav_ref.shape)
# idx = -1
# distance, path = fastdtw(pcav[0:idx]/np.max(pcav[0:idx]), pcav_ref[0:idx]/np.max(pcav_ref[0:idx]), dist=euclidean)
# # print(np.corrcoef(pcav[:,0]/np.max(pcav[:,0]), pcav_ref[:,0]/np.max(pcav_ref[:,0])))
# print(distance, np.exp(-distance))
# plt.figure(figsize=(10, 6))
# plt.plot(pcav_ref, label='Reference')
# plt.plot(pcav[0:idx], label='Obtained')
# plt.grid()
# plt.legend()
# plt.show()
# # cosine_sim = np.dot(pcav[:,0], pcav_ref[:,0]) / (np.linalg.norm(pcav[:,0]) * np.linalg.norm(pcav_ref[:,0]))
# # print(cosine_sim)
# distance = fast_dtw(pcav[0:idx,0]/np.max(pcav[0:idx,0]), pcav_ref[0:idx,0]/np.max(pcav_ref[0:idx,0]))
# print(distance)
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
# plt.savefig('./maddpg_results/ac_ecav_hist_spec_all_ctrl.png')
plt.show()

# %%
plt.figure(figsize=(14,4))
plt.imshow(np.abs(np.fft.fftshift(np.fft.fft(1e3*np.array(acav_hist)))).T, aspect='auto', cmap='jet'\
            ,extent=[0, len(acav_hist), env.mu.min().item(), env.mu.max().item()])
plt.xlabel('Tuning Steps', fontsize=14)
plt.ylabel(r'$\mu (rel)$', fontsize=14)
plt.colorbar(label='Power(dBm)')
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
# plt.savefig('./maddpg_results/ac_ecav_hist_ifft_spec_all_ctrl.png')
plt.show()
# %%
plt.figure(figsize=(10, 6))
plt.plot(r_hist)
plt.xlabel('Iteration', fontsize=14)
plt.ylabel('Reward '+ r'$(1-\alpha)(corr) + \alpha (-mse)$', fontsize=14)
plt.grid()
# plt.savefig('./maddpg_results/ac_rewards_spec_all_ctrl.png')
plt.show()
# %%
desired_spectrum_dBm = 10*torch.log10(torch.abs(desired_spectrum)**2)+30
plt.figure(figsize=(14,4))
plt.vlines(np.arange(len(ecav)), -60*np.ones(len(ecav)), ecav, \
           label='Obtained Spectrum')
plt.vlines(np.arange(len(desired_spectrum)), -60*np.ones(len(desired_spectrum)),\
            desired_spectrum_dBm, color='red', label='Desired Spectrum',alpha=0.5)
plt.xlabel('Mode no.', fontsize=14)
plt.ylabel('Power(dBm)', fontsize=14)
plt.grid()
plt.ylim(-90,5)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=14)
plt.savefig('./maddpg_results/ac_ecav_spec_all_ctrl.png')
plt.show()
# %%
action_hist = np.array(action_hist)
plt.figure(figsize=(10, 6))
actions = np.array(action_hist[:,0]).astype(int)
plt.plot(actions, label='Actions', color='gray', alpha=0.5)

# Highlight different actions with different colors
plt.scatter(np.where(actions == 0), actions[actions == 0], color='red', label='Decrease Detuning', marker='o')
plt.scatter(np.where(actions == 1), actions[actions == 1], color='green', label='Increase Detuning', marker='x')
plt.scatter(np.where(actions == 2), actions[actions == 2], color='blue', label='No Update', marker='s')

plt.xlabel('Iteration', fontsize=14)
plt.ylabel('Action', fontsize=14)
plt.legend()
plt.grid()
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
# plt.savefig('./maddpg_results/ac_actions_spec_all_ctrl.png')
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(action_hist[:,1])
plt.xlabel('Iteration', fontsize=14)
plt.ylabel('Pump Power', fontsize=14)
plt.grid()
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
# plt.savefig('./maddpg_results/ac_power_all_ctrl.png')
plt.show()
# %%
