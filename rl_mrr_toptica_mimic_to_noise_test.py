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
class RL_MRR_Env():

    def __init__(self, seq_len=50, p_max=0.3, p_min=0.1, ctrl_freq=100):
        super(RL_MRR_Env, self).__init__()

        self.step_cntr = 0
        
        self.disp = loadmat('disp_copy.mat')
        self.res = loadmat('res_copy.mat')
        self.sim = loadmat('sim_copy.mat')


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
        self.FSR = FSR
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

        self.tau0 = 100e-9
        self.xi = -4.5e4
        self.delta_theta = torch.tensor(0.0, device=DEVICE, dtype=torch.float64)

        self.un_norm_kappa = 2*torch.pi*(fpmp[0]/Q0 + fpmp[0]/Qc)

        # del_omega_init = self.sim_tensor['domega_init']
        del_omega_init = 2*self.un_norm_kappa
        self.del_omega_init = del_omega_init
        self.current_del_omega = del_omega_init
        # del_omega_end = self.sim_tensor['domega_end']
        del_omega_end = -10*self.un_norm_kappa
        self.del_omega_end = del_omega_end

        # del_omega_stop = self.sim_tensor['domega_stop']
        del_omega_stop = -12*self.un_norm_kappa
        self.del_omega_stop = del_omega_stop
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

        self.Dint_shift = torch.fft.ifftshift(self.Dint)

        dt = 1
        self.max_steps = int(4e5)
        t_end  = self.max_steps*tR.cpu().numpy()
        t_ramp = t_end
        tr = tR.cpu().numpy()
        self.Nt = np.round(t_ramp/tr/dt).astype(int)

        self.del_omega_0 = del_omega_init + (1/self.Nt)*(del_omega_end - del_omega_init)

        del_omega_tot = torch.abs(del_omega_end)+torch.abs(del_omega_init)
        del_omega_perc = -1*torch.sign(del_omega_end+del_omega_init)*(torch.abs(del_omega_end+del_omega_init)/2)/del_omega_tot
        self.t_sim = torch.linspace(-t_ramp/2 + del_omega_perc[0]*t_ramp, t_ramp/2 + del_omega_perc[0]*t_ramp, self.Nt, device=DEVICE, dtype=torch.float64)
        self.t_sim_start = -t_ramp/2 + del_omega_perc[0]*t_ramp
        self.t_sim_step = self.t_sim[1] - self.t_sim[0]

        self.pcav_ref = loadmat('ref_check.mat')['Pcomb'].T
        self.primary_sidebands = loadmat('primary_sidebands.mat')['spec'][0]
        # self.pcav_ref = loadmat('Pcomb_rl_allv2.mat')['Pcomb'].T
        self.seq_len = seq_len
        self.p_max = p_max
        self.p_min = p_min
        self.ctrl_freq = ctrl_freq

    
    @staticmethod
    @torch.jit.script
    def Noise(h_bar: float, omega1: torch.Tensor, mu_len: int, device: torch.device) -> torch.Tensor:
        Ephoton = torch.tensor(h_bar, device=device) * omega1
        phase = 2 * torch.pi * torch.rand(mu_len,  device=device)
        array = torch.rand(mu_len,  device=device)
        Enoise = array * torch.sqrt(Ephoton / 2) * torch.exp(1j * phase) * mu_len
        return torch.fft.ifftshift(torch.fft.ifft(Enoise)).squeeze()
    
    def Fdrive(self, del_omega_all, t_sim, Ain):
        Force = torch.zeros(len(self.mu), device=DEVICE, dtype=torch.complex128)
        for ii in range(len(self.ind_pmp)):
            if ii > 0:
                sigma = (2*del_omega_all[ii] + self.Dint[(self.mu0)+self.ind_pmp[ii]] - (0.5*del_omega_all[0] + self.delta_theta))*t_sim
            else:
                sigma=torch.zeros(1, device=DEVICE)
            Force = Force - 1j*Ain[ii]*torch.exp(1j*sigma)
        return Force + self.Noise(h_bar=H_BAR, omega1=2*torch.pi*self.sim_tensor['f_pmp'], mu_len=len(self.mu), device=DEVICE)

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
    def FFT_Lin(it: int, alpha: torch.Tensor, Dint_shift: torch.Tensor, del_omega_all: torch.Tensor, tR: torch.Tensor, delta_theta:torch.Tensor) -> torch.Tensor:
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
        return (-alpha / 2) + 1j * (Dint_shift - del_omega_all + delta_theta) * tR


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
        L_h_prop = torch.exp(self.FFT_Lin(it, alpha, Dint_shift, del_omega_all, tR, self.delta_theta) * dt / 2)
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
    
    def reset(self, steps=None):
        print('Detuning range:', self.del_omega_init.item()/(2*np.pi*1e9), ' to ', self.del_omega_end.item()/(2*np.pi*1e9), ' GHz')
        self.state = self.DKS_init
        self.current_del_omega = self.del_omega_0
        self.delta_theta = torch.tensor(0.0, device=DEVICE, dtype=torch.float64)
        self.step_cntr = 0
        self.pcav_hist = []

        # self.power = np.random.uniform(self.p_min, self.p_max, size=(1,))
        self.power = np.array([0.1])
        Ppmp = torch.tensor(self.power, dtype=torch.float64)

        for ii in range(len(Ppmp)):
            self.Ein[ii,int(self.mu0+self.ind_pmp[ii])] = torch.sqrt(Ppmp[ii])*len(self.mu)
            self.Ain[ii] = torch.fft.ifft(torch.fft.fftshift(self.Ein[ii],dim=0),dim=0)*torch.exp(-1j*self.phi_pmp[ii])
        
        # self.Ain = Ain
        if steps is not None:
            self.init_steps_ = steps
        else:
            self.init_steps_ = int(1.5e5)

        self.ecav_state = []
        for idx in range(self.init_steps_):
            mul_factor = np.random.choice([1, -1, 0], p=[1/3, 1/3, 1/3])
            del_omega = self.current_del_omega + mul_factor*(1/self.Nt)*(self.del_omega_end - self.del_omega_init)

            Fdrive_val = self.Fdrive(del_omega, self.t_sim_start+self.step_cntr*self.t_sim_step, self.Ain)
            u0 = self.ssfm_step(self.state, self.step_cntr, self.alpha, self.Dint_shift, del_omega, self.tR, self.gamma, \
                                self.L, 10, 1e-3, 1, self.kext, Fdrive_val)
            self.step_cntr += 1
            self.state = u0
            P_avg = torch.mean(torch.abs(u0)**2)  # Compute average power
            d_delta_theta_dt = -self.delta_theta / self.tau0 + self.xi * P_avg
            self.delta_theta += (1 * self.tR) * d_delta_theta_dt  # Euler step
            self.current_del_omega = del_omega + self.delta_theta

            Acav = torch.sqrt(self.alpha/2)*self.state*np.exp(1j*torch.pi)/len(self.mu)
            Ecav = torch.fft.fftshift(torch.fft.fft(Acav))
            Ecav_dBm = 10*torch.log10(torch.abs(Ecav)**2)+30
            Ecav_dBm = torch.clamp(Ecav_dBm, min=-60, max=10)
            Acav_np = Acav.cpu().numpy()
            curr_pcav = np.sum(np.abs(Acav_np))
            self.pcav_hist.append(curr_pcav)
            if idx % self.ctrl_freq == 0:
                self.ecav_state.append(Ecav_dBm.cpu().numpy())
                if len(self.ecav_state) > self.seq_len:
                    self.ecav_state.pop(0)

        self.primary_sidebands_flag = False
        self.ecav_state = np.array(self.ecav_state)

        self.env_p_hist = []
        
        print('Reset...')
        return self.state, Acav_np, self.ecav_state

    def rescale_and_quantize(self,action, lower_limit=-1e6, upper_limit=1e6, step_size=1e4):
        """
        Rescale input in [-1, 1] to [lower_limit, upper_limit] and quantize to step_size.

        Parameters:
            action (float): Input value in [-1, 1]
            lower_limit (float): Lower bound in Hz (default -0.5 GHz)
            upper_limit (float): Upper bound in Hz (default 0.5 GHz)
            step_size (float): Quantization step in Hz (default 10 kHz)

        Returns:
            float: Quantized output in Hz
        """
        # Clip the input to ensure it's within [-1, 1]
        action = np.clip(action, -1, 1)
        # Rescale to [lower_limit, upper_limit]
        value = lower_limit + (action + 1) * (upper_limit - lower_limit) / 2
        # Quantize to nearest step_size
        quantized_value = np.round(value / step_size) * step_size
        return quantized_value
    
    def rescale_power(self, power, lower_limit=0.12, upper_limit=0.16, step_size=0.01):
        """
        Rescale input power in [-1, 1] to [lower_limit, upper_limit] and quantize to step_size.

        Parameters:
            power (float): Input value in [-1, 1]
            lower_limit (float): Lower bound in W (default 0.12 W)
            upper_limit (float): Upper bound in W (default 0.16 W)
            step_size (float): Quantization step in W (default 0.001 W)

        Returns:
            float: Quantized output in W
        """
        # Clip the input to ensure it's within [-1, 1]
        power = np.clip(power, -1, 1)
        # Rescale to [lower_limit, upper_limit]
        value = lower_limit + (power + 1) * (upper_limit - lower_limit) / 2
        # Quantize to nearest step_size
        quantized_value = np.round(value / step_size) * step_size
        return quantized_value
    
    def is_terminal(self, Ecav_dBm, desired_spectrum_dBm, achieved):
        terminal = False
        reward_penalty = 0
        corr = torch.corrcoef(torch.stack([desired_spectrum_dBm, Ecav_dBm]))[0,1].item()
        if self.step_cntr<45000 and self.primary_sidebands_flag==False:
            self.primary_sidebands_flag = np.corrcoef(self.primary_sidebands, Ecav_dBm.cpu().numpy())[0,1]<0.5
        
        if self.step_cntr == 45000 and self.primary_sidebands_flag == False:
            terminal = True
            reward_penalty = -10
            print('Primary Sidebands not formed')
            print('Corr:',np.corrcoef(self.primary_sidebands, Ecav_dBm.cpu().numpy())[0,1])
        elif self.step_cntr-self.init_steps_ >= int(0.5*self.Nt) and corr < 0.25: #and self.step_cntr-self.init_steps_ <= self.Nt:
            terminal = True
            reward_penalty = -5
            print('Did not form soliton ...')
            print('Spectral Corr:', corr)
        # elif self.step_cntr > self.Nt and achieved == False:
        #     terminal = True
        #     reward_penalty = -5
        #     print('Did not achieve desired spectrum ...')
        #     print('Spectral Corr:', corr)
        return terminal, reward_penalty
    
    def step(self, state, action, desired_spectrum):
        env.power = self.rescale_power(action[0:1], lower_limit=self.p_min, upper_limit=self.p_max, step_size=0.001)
        self.env_p_hist.append(env.power[0])
        if len(self.env_p_hist) > env.seq_len:
            self.env_p_hist.pop(0)
        
        Ppmp = torch.tensor(env.power, dtype=torch.float64)
        
        for ii in range(1):
            self.Ein[ii,int(self.mu0+self.ind_pmp[ii])] = torch.sqrt(Ppmp[ii])*len(self.mu)
            self.Ain[ii] = torch.fft.ifft(torch.fft.fftshift(self.Ein[ii],dim=0),dim=0)*torch.exp(-1j*self.phi_pmp[ii])
        
        # det_delta = action*(2/self.Nt)*(self.del_omega_end - self.del_omega_init)
        det_delta = self.rescale_and_quantize(action[1])*(2*np.pi)  # Convert GHz to rad/s
        
        for _ in range(self.ctrl_freq):
            del_omega = self.current_del_omega + det_delta + self.delta_theta

            del_omega = torch.clamp(del_omega, min=self.del_omega_stop, max=self.del_omega_init)

            self.current_del_omega = torch.clamp(del_omega, min=self.del_omega_end, max=self.del_omega_init)

            Fdrive_val = self.Fdrive(del_omega, self.t_sim_start+self.step_cntr*self.t_sim_step, self.Ain)
            u0 = self.ssfm_step(state, self.step_cntr, self.alpha, self.Dint_shift, del_omega, self.tR, self.gamma, \
                                self.L, 10, 1e-3, 1, self.kext, Fdrive_val)
            state = u0
            P_avg = torch.mean(torch.abs(u0)**2)  # Compute average power
            d_delta_theta_dt = -self.delta_theta / self.tau0 + self.xi * P_avg
            self.delta_theta += (1 * self.tR) * d_delta_theta_dt  # Euler step
            self.step_cntr += 1
        self.next_state = u0
        
        Acav = torch.sqrt(self.alpha/2)*u0*np.exp(1j*torch.pi)/len(self.mu)
        Ecav = torch.fft.fftshift(torch.fft.fft(Acav))

        Acav_np = Acav.numpy()
        curr_pcav = np.sum(np.abs(Acav_np))
        self.pcav_hist.append(curr_pcav)
        if len(self.pcav_hist) > 10000:
            self.pcav_hist.pop(0)

        Ecav_dBm = 10*torch.log10(torch.abs(Ecav)**2)+30
        Ecav_dBm = torch.clamp(Ecav_dBm, min=-60, max=10)
        desired_spectrum_dBm = 10*torch.log10(torch.abs(desired_spectrum)**2)+30
        desired_spectrum_dBm = torch.clamp(desired_spectrum_dBm, min=-60, max=10)

        
        # pop the first element of ecav_state and append new Ecav_dBm
        # self.ecav_state = np.concatenate((self.ecav_state[1:], Ecav_dBm.cpu().numpy()[np.newaxis,:]), axis=0)
        if self.ecav_state.shape[0] >= self.seq_len:
            self.ecav_state = np.delete(self.ecav_state, 0, axis=0)
        self.ecav_state = np.concatenate((self.ecav_state, Ecav_dBm.cpu().numpy()[np.newaxis,:]), axis=0)

        reward = 4*torch.corrcoef(torch.stack([desired_spectrum_dBm, Ecav_dBm]))[0,1].item() #+ 1
        # penalize for high variance in power
        if len(self.env_p_hist) > 1:
            power_var = np.std(self.env_p_hist)
            if power_var > 0.001:
                reward -= 2*len(self.env_p_hist) * (power_var - 0.001)
        
        if torch.linalg.vector_norm(desired_spectrum_dBm-Ecav_dBm, ord=2) < 50 or torch.corrcoef(torch.stack([desired_spectrum_dBm, Ecav_dBm]))[0,1].item() > 0.9:
            achieved = True
            reward += 2
        else:
            achieved = False

        done = False
        terminal, reward_penalty = self.is_terminal(Ecav_dBm, desired_spectrum_dBm, achieved)
        reward += reward_penalty

        if self.step_cntr+1 >= self.max_steps:
            done = True            
        else:
            done = terminal
            
        
        return self.next_state, reward, done, terminal, achieved, Acav_np, self.ecav_state

# %%
# torch seed
# torch.manual_seed(0)
env = RL_MRR_Env(seq_len=100, p_max=0.16, p_min=0.12, ctrl_freq=100)
fpmp = env.sim_tensor['f_pmp'].item()
freq = (fpmp + np.arange(-220,221)*env.FSR.item())*1e-12
# %%
desired_spectrum = loadmat('desired_spec.mat')['Ecav'][0]
desired_spectrum_dBm = 10*np.log10(np.abs(desired_spectrum)**2)+30
desired_spectrum_tensor = torch.tensor(desired_spectrum, device=DEVICE, dtype=torch.complex128)
# %%
config = {
    'input_dim': [env.seq_len, 441+2],
    'n_actions': 2,
    'alpha': 3e-4,
    'beta': 3e-4,
    'mem_size': int(1e6),
    'run_name': 'mrr_sac_cluster_delayed_toptica_pow_ton',
    'batch_size': 128,
    'dist': 'beta',
    'train':False,
    'p_max': env.p_max,
    'p_min': env.p_min,
    'fc_dim':128,
    'use_per':True
    }
# %%

from sac import SACAgent
agent = SACAgent(input_dim=config['input_dim'], n_actions=config['n_actions'], alpha=config['alpha'], beta=config['beta'],
                mem_size=config['mem_size'], batch_size=config['batch_size'], dist=config['dist'], run_name=config['run_name'],
                eval_mode=True, fc_dim=config['fc_dim'], use_per=config['use_per'])
print(agent.actor)
print(agent.critic_1)
agent.load_models()
# %%
# '''
state, acav, ecav = env.reset(10000)
den = env.p_max - env.p_min
obs = np.concatenate((ecav/10,env.power*np.ones((env.seq_len,1))/den,np.zeros((env.seq_len,1))),axis=1)
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
while not done:
# for idx in tqdm(range(env.init_steps_, int(env.max_steps)), ncols=120):
    # perform random actions
    # try:
        action = agent.choose_action(obs, True)

        next_state, reward, done, terminal, achieved, acav_, ecav_ = env.step(state, action, desired_spectrum_tensor)
        state = next_state
        ecav = ecav_
        ecav_obs = np.concatenate((ecav_[-1]/10, env.power/den, env.rescale_and_quantize(action[1:])*1e-6), axis=0)
        obs_ = np.concatenate((obs[1:], ecav_obs[np.newaxis,:]), axis=0)
        obs = obs_ 
        score += reward
        curr_pcav = np.sum(np.abs(acav_))
        pcav_hist.append(curr_pcav)
        r_hist.append(reward)
        action_hist.append(action)
       
        acav_hist.append(acav_)
        idx += env.ctrl_freq
        pbar.update(env.ctrl_freq)
pbar.close()

print('Test score %.2f' % score)
# %%
# %%
import os

# Create save directory if not exists
save_dir = os.path.join('./results', agent.run_name)
os.makedirs(save_dir, exist_ok=True)
plt.style.use('physrev.mplstyle')
# %%
# find correlation between the obtained pcav and r_hist[:,-1]
plt.figure(figsize=(10, 6))
plt.plot(pcav_hist, linewidth=1.5)
plt.grid()
plt.xlabel('Steps', fontsize=16)
plt.ylabel('Pcav', fontsize=16)
plt.title('Pump Power: '+str(env.power[0])+'mW', fontsize=16, fontweight='bold')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
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
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=16)
cbar.set_label(r'Power $(dBm)$', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
# Set x-ticks to exponent format
formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(True)
formatter.set_powerlimits((-1, 1))
plt.gca().xaxis.set_major_formatter(formatter)
plt.xlabel('Tuning Steps', fontsize=14)
plt.ylabel(r'$t_R (ps)$', fontsize=14)
plt.title('Pump Power: '+str(env.power[0])+' mW', fontsize=16, fontweight='bold')
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
plt.xlabel('Tuning Steps', fontsize=18)
plt.ylabel(r'$\mu$' +'(rel)', fontsize=18)
cbar = plt.colorbar()
# set colorbar ticks size
cbar.ax.tick_params(labelsize=16)
cbar.set_label(r'Power $(dBm)$', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.title('Pump Power: '+str(env.power[0])+' mW', fontsize=16, fontweight='bold')
mod_pow = str(env.power[0]).replace('.','_')
plt.tight_layout()
if idx > int(0.5*env.max_steps):
    plt.savefig(os.path.join(save_dir, mod_pow + '_ecav_hist_ifft_spec_all_ctrl.png'))
plt.show()
# %% Reward Plot
plt.figure(figsize=(10, 6))
plt.plot(r_hist)
plt.xlabel('Tuning Steps', fontsize=16)
plt.ylabel('Reward ', fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.grid()
plt.title('Pump Power: '+str(env.power[0])+' mW', fontsize=16, fontweight='bold')
plt.tight_layout()
mod_pow = str(env.power[0]).replace('.','_')
if idx > int(0.5*env.max_steps):
    plt.savefig(os.path.join(save_dir, mod_pow + '_rewards_spec_all_ctrl.png'))
plt.show()
# %%
desired_spectrum_dBm_ = np.clip(desired_spectrum_dBm, -60, 10)
mse = np.linalg.norm(desired_spectrum_dBm_-ecav[-1], ord=2)
corr = np.corrcoef(desired_spectrum_dBm_, ecav_[-1])[0, 1]
print('MSE:', mse, 'Corr:', corr)
# %%
# desired_spectrum_dBm = 10*torch.log10(torch.abs(desired_spectrum)**2)+30
plt.figure(figsize=(14,4))
plt.vlines(np.arange(-220,221, 1), -60*np.ones(len(ecav[-1])), ecav[-1], \
           label='Obtained Spectrum',alpha=1, linewidth=1.5)
plt.vlines(np.arange(-220,221, 1), -60*np.ones(len(desired_spectrum)),\
            desired_spectrum_dBm, color='red', label='Desired Spectrum',alpha=0.5, linewidth=1.5)
plt.xlabel('Rel. Mode no.', fontsize=18)
plt.ylabel('Power(dBm)', fontsize=18)
plt.grid()
plt.ylim(-90,5)
plt.xlim(-150, 150)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.legend(fontsize=18,loc='lower center')
plt.title('Pump Power: '+str(env.power[0])+' mW', fontsize=16, fontweight='bold')
mod_pow = str(env.power[0]).replace('.','_')
plt.tight_layout()
if idx > int(0.5*env.max_steps):
    plt.savefig(os.path.join(save_dir, mod_pow + '_ecav_spec_all_ctrl_modes.png'))
plt.show()

plt.figure(figsize=(14,4))
plt.vlines(freq, -60*np.ones(len(ecav[-1])), ecav[-1], \
           label='Obtained Spectrum',alpha=1, linewidth=1.5)
plt.vlines(freq, -60*np.ones(len(desired_spectrum)),\
            desired_spectrum_dBm, color='red', label='Desired Spectrum',alpha=0.5, linewidth=1.5)
plt.xlabel('Freq. (THz)', fontsize=18)
plt.ylabel('Power(dBm)', fontsize=18)
plt.grid()
plt.ylim(-90,5)
plt.xlim(freq[220-150], freq[220+150])
plt.xticks(fontsize=18, fontweight='bold')
plt.yticks(fontsize=18, fontweight='bold')
plt.legend(fontsize=18, loc='lower center')
plt.title('Pump Power: '+str(env.power[0])+' mW', fontsize=16, fontweight='bold')
mod_pow = str(env.power[0]).replace('.','_')
plt.tight_layout()
if idx > int(0.5*env.max_steps):
    plt.savefig(os.path.join(save_dir, mod_pow + '_ecav_spec_all_ctrl_freq.png'))
plt.show()
# %%
action_hist = np.array(action_hist)
del_detuning = env.rescale_and_quantize(action_hist[:,1])
det_start = env.del_omega_init.item()/(2*np.pi)
detuning_array = det_start + np.cumsum(del_detuning)
detuning_array = np.clip(detuning_array, env.del_omega_end.item()/(2*np.pi), env.del_omega_init.item()/(2*np.pi))
plt.figure(figsize=(10, 6))
plt.plot(detuning_array*1e-9)
plt.xlabel('Tuning Steps', fontsize=18)
plt.ylabel('Pump detuning (GHz)', fontsize=18)
plt.grid()
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.gca().xaxis.set_major_formatter(formatter)
plt.title('Pump Power: '+str(env.power[0])+'mW', fontsize=16, fontweight='bold')
mod_pow = str(env.power[0]).replace('.','_')
plt.tight_layout()
if idx > int(0.5*env.max_steps):
    plt.savefig(os.path.join(save_dir, mod_pow + '_actions_spec_all_ctrl.png'))
plt.show()

# %%
plt.figure(figsize=(10, 6))
plt.plot(env.rescale_power(action_hist[:,0]))
plt.xlabel('Tuning Steps', fontsize=18)
plt.ylabel('Pump Power (mW)', fontsize=18)
plt.grid()
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.title('Pump Power: '+str(env.power[0])+' mW', fontsize=16, fontweight='bold')
mod_pow = str(env.power[0]).replace('.','_')
plt.tight_layout()
if idx > int(0.5*env.max_steps):
    plt.savefig(os.path.join(save_dir, mod_pow + '_actions_power_spec_all_ctrl.png'))
plt.show()
'''
# %%
def run_test_processes(run_id, save_dir):
    # Re-create environment and agent inside the process
    env = RL_MRR_Env(seq_len=100, p_max=0.16, p_min=0.12)
    desired_spectrum = loadmat('desired_spec.mat')['Ecav'][0]
    desired_spectrum_tensor = torch.tensor(desired_spectrum, device=DEVICE, dtype=torch.complex128)
    from sac import SACAgent
    config = {
        'input_dim': [env.seq_len, 441+2],
        'n_actions': 2,
        'alpha': 3e-4,
        'beta': 3e-4,
        'mem_size': int(1e6),
        'run_name': 'mrr_sac_cluster_delayed_toptica_pow_ton',
        'batch_size': 128,
        'dist': 'beta',
        'train':False,
        'p_max': env.p_max,
        'p_min': env.p_min,
        'fc_dim': 128
    }
    agent = SACAgent(input_dim=config['input_dim'], n_actions=config['n_actions'], alpha=config['alpha'], beta=config['beta'],
                    mem_size=config['mem_size'], batch_size=config['batch_size'], dist=config['dist'], run_name=config['run_name'],
                    eval_mode=not(torch.cuda.is_available()), fc_dim=config['fc_dim'])
    agent.load_models()
    state, _, ecav = env.reset(10000)
    den = env.p_max - env.p_min
    obs = np.concatenate((ecav/10,env.power*np.ones((env.seq_len,1))/den,np.zeros((env.seq_len,1))),axis=1)
    print('Chosen power:', env.power,'\n')
    r_hist = []
    score = 0
    done = False
    pbar = tqdm(total=env.max_steps-env.init_steps_, ncols=120, position=run_id, desc=f'Run {run_id}')
    idx = 0
    done = False
    while not done:
    # for idx in tqdm(range(env.init_steps_, int(env.max_steps)), ncols=120):
        # perform random actions
        # try:
            action = agent.choose_action(obs, True)

            next_state, reward, done, terminal, achieved, acav_, ecav_ = env.step(state, action, desired_spectrum_tensor)
            state = next_state
            # ecav = ecav_
            ecav_obs = np.concatenate((ecav_[-1]/10, env.power/den, env.rescale_and_quantize(action[1:])*1e-6), axis=0)
            obs_ = np.concatenate((obs[1:], ecav_obs[np.newaxis,:]), axis=0)
            obs = obs_ 
            score += reward

            r_hist.append(reward)
            idx += env.ctrl_freq
            pbar.update(env.ctrl_freq)
    pbar.close()

    print('Test score %.2f' % score)
    mod_pow = str(np.round(env.power[0],4)).replace('.','_')
    
    # if env.step_cntr >= env.max_steps-env.init_steps_-2:
    print('Idx:', idx)
    # save the reward history
    np.save(os.path.join(save_dir, mod_pow + '_reward_hist_spec_all_ctrl_v2.npy'), r_hist)
# %%
# write a function to load the reward history and plot it
def plot_reward_histories_sigma(files, N=100, S=0, label='Reward', color='C0'):
    """
    Plot rolling mean and std of rewards from multiple runs, handling different lengths.
    Optionally leave the last S samples from max_len and then plot.

    Args:
        files (list): List of file paths to .npy reward histories.
        N (int): Window size for rolling mean/std.
        S (int): Number of samples to leave from the end.
        label (str): Label for the mean line.
        color (str): Color for the plot.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    plt.style.use('physrev.mplstyle')

    # Load all reward histories
    rewards = [np.load(f) for f in files]
    max_len = max(len(r) for r in rewards)
    # Pad with np.nan to align lengths
    rewards_padded = np.full((len(rewards), max_len), np.nan)
    for i, r in enumerate(rewards):
        rewards_padded[i, :len(r)] = r

    # Optionally leave the last S samples
    plot_len = max_len - S if S > 0 else max_len

    # Compute rolling mean and std, ignoring nan
    rolling_mean = []
    rolling_std = []
    for t in range(plot_len):
        window = rewards_padded[:, max(0, t-N+1):t+1]
        vals = window[~np.isnan(window)]
        if len(vals) > 0:
            rolling_mean.append(np.mean(vals))
            rolling_std.append(np.std(vals))
        else:
            rolling_mean.append(np.nan)
            rolling_std.append(np.nan)

    steps = np.linspace(0, 100*plot_len, plot_len)*1e-5
    mu = np.array(rolling_mean)
    sigma = np.array(rolling_std)

    plt.figure(figsize=(7, 5))
    plt.plot(steps, mu, color=color, linewidth=1.5)
    plt.fill_between(steps, mu - sigma, mu + sigma, color=color, alpha=0.3)
    plt.xlabel(r'Steps $(\times 10^5)$', fontsize=14)
    plt.ylabel('Reward', fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    # set x-ticks to exponent format
    # formatter = ticker.ScalarFormatter(useMathText=True)
    # formatter.set_scientific(True)
    # formatter.set_powerlimits((-1, 1))
    # plt.gca().xaxis.set_major_formatter(formatter)
    # plt.title('Reward Rolling Mean ± Std', fontsize=16, fontweight='bold')
    # plt.legend(fontsize=14)
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'reward_histories_sigma.png'))
    plt.show()
# %%
def plot_reward_histories_min_max(files, N=100, S=0, label='Reward', color='C0'):
    """
    Plot rolling mean and min/max of rewards from multiple runs, handling different lengths.
    Optionally leave the last S samples from max_len and then plot.

    Args:
        files (list): List of file paths to .npy reward histories.
        N (int): Window size for rolling stats.
        S (int): Number of samples to leave from the end.
        label (str): Label for the mean line.
        color (str): Color for the plot.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    plt.style.use('physrev.mplstyle')

    # Load all reward histories
    rewards = [np.load(f) for f in files]
    max_len = max(len(r) for r in rewards)
    # Pad with np.nan to align lengths
    rewards_padded = np.full((len(rewards), max_len), np.nan)
    for i, r in enumerate(rewards):
        rewards_padded[i, :len(r)] = r

    # Optionally leave the last S samples
    plot_len = max_len - S if S > 0 else max_len

    # Compute rolling mean, min, and max, ignoring nan
    rolling_mean = []
    rolling_min = []
    rolling_max = []
    for t in range(plot_len):
        window = rewards_padded[:, max(0, t-N+1):t+1]
        vals = window[~np.isnan(window)]
        if len(vals) > 0:
            rolling_mean.append(np.mean(vals))
            rolling_min.append(np.min(vals))
            rolling_max.append(np.max(vals))
        else:
            rolling_mean.append(np.nan)
            rolling_min.append(np.nan)
            rolling_max.append(np.nan)

    steps = np.linspace(0, 100*plot_len, plot_len)*1e-5
    mu = np.array(rolling_mean)
    minv = np.array(rolling_min)
    maxv = np.array(rolling_max)

    plt.figure(figsize=(7, 5))
    plt.plot(steps, mu, color=color, linewidth=1.5)
    plt.fill_between(steps, minv, maxv, color=color, alpha=0.3)
    plt.xlabel(r'Steps $(\times 10^5)$', fontsize=14)
    plt.ylabel('Reward', fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid()
    plt.tight_layout()
    # plt.legend(fontsize=14)
    plt.savefig(os.path.join(save_dir, 'reward_histories_min_max.png'))
    plt.show()
# %%
import torch.multiprocessing as mp
import os
import glob

if __name__ == '__main__':
    # Create save directory if not exists
    save_dir = os.path.join('./results', agent.run_name)
    os.makedirs(save_dir, exist_ok=True)
    print('Save dir:', save_dir)
    mp.set_start_method('spawn', force=True)  # safer for PyTorch
    num_runs = 10
    processes = []
    for run_id in range(num_runs):
        p = mp.Process(target=run_test_processes, args=(run_id, save_dir))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    # get the list of all npy files in the directory
    npy_files = glob.glob(os.path.join(save_dir, '*.npy'))
    # # # Example usage: plot all reward histories with a rolling window of 100
    plot_reward_histories_sigma(npy_files, N=5, S=0, label='Reward', color='C0')
    plot_reward_histories_min_max(npy_files, N=5, S=0, label='Reward', color='C0')
'''