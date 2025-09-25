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

    def __init__(self, seq_len=50, p_max=0.3, p_min=0.1, ctrl_freq=100, thermal_effect='moderate', 
                 delta_omega_min=-1e6, delta_omega_max=1e6, delta_omega_step=1e4):
        

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

        self.thermal_effect = thermal_effect

        self.tau0 = 100e-9
        if thermal_effect == 'low':
            self.xi = -1.2e4
        elif thermal_effect == 'moderate':
            self.xi = -(4.5e4)
        elif thermal_effect == 'high':
            self.xi = -1.2e5
        else:
            raise ValueError("Invalid thermal effect. Choose from 'low', 'moderate', or 'high'.")
        
        self.delta_theta = torch.tensor(0.0, device=DEVICE, dtype=torch.float64)

        self.un_norm_kappa = 2*torch.pi*(fpmp[0]/Q0 + fpmp[0]/Qc)

        # del_omega_init = self.sim_tensor['domega_init']
        del_omega_init = 4*self.un_norm_kappa
        self.del_omega_init = del_omega_init
        self.current_del_omega = del_omega_init
        # del_omega_end = self.sim_tensor['domega_end']
        del_omega_end = -5*self.un_norm_kappa
        self.del_omega_end = del_omega_end

        # del_omega_stop = self.sim_tensor['domega_stop']
        del_omega_stop = -5*self.un_norm_kappa
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

        self.primary_sidebands = loadmat('primary_sidebands.mat')['spec'][0]
        # self.pcav_ref = loadmat('Pcomb_rl_allv2.mat')['Pcomb'].T
        self.seq_len = seq_len
        self.p_max = p_max
        self.p_min = p_min
        self.ctrl_freq = ctrl_freq
        self.delta_omega_min = delta_omega_min  
        self.delta_omega_max = delta_omega_max
        self.delta_omega_step = delta_omega_step

    
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
                sigma = (2*del_omega_all[ii] + self.Dint[(self.mu0)+self.ind_pmp[ii]] - 0.5*del_omega_all[0])*t_sim
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
            self.current_del_omega = del_omega

            Fdrive_val = self.Fdrive(del_omega + (self.delta_theta/self.tR), self.t_sim_start+self.step_cntr*self.t_sim_step, self.Ain)
            u0 = self.ssfm_step(self.state, self.step_cntr, self.alpha, self.Dint_shift, del_omega + (self.delta_theta/self.tR), self.tR, self.gamma, \
                                self.L, 10, 1e-3, 1, self.kext, Fdrive_val)
            self.step_cntr += 1
            self.state = u0
            P_avg = torch.mean(torch.abs(u0)**2)  # Compute average power
            d_delta_theta_dt = -self.delta_theta / self.tau0 + self.xi * P_avg
            self.delta_theta += (1 * self.tR) * d_delta_theta_dt  # Euler step
             

            Acav = torch.sqrt(self.alpha/2)*self.state*np.exp(1j*torch.pi)/np.sqrt(len(self.mu))
            Ecav = torch.fft.fftshift(torch.fft.fft(Acav))/np.sqrt(len(self.mu))
            cav = Fdrive_val*torch.sqrt(1-self.kext)
            wg = torch.sqrt(self.kext)*u0*np.exp(1j*np.pi)
            Awg = (wg + cav)/np.sqrt(len(self.mu))
            Ewg = torch.fft.fftshift(torch.fft.fft(Awg))/np.sqrt(len(self.mu))
            Ecav_dBm = 10*torch.log10(torch.abs(Ewg)**2)+30
            Ecav_dBm = torch.clamp(Ecav_dBm, min=-60, max=None)
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

        self.det_out_cntr = 0
        
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
        
        # if self.step_cntr == int(0.4*self.Nt) and self.primary_sidebands_flag == False:
        #     terminal = True
        #     reward_penalty = -5
        #     print('Primary Sidebands not formed')
            # print('Corr:',np.corrcoef(self.primary_sidebands, Ecav_dBm.cpu().numpy())[0,1])

        if self.step_cntr-self.init_steps_ >= int(0.5*self.Nt) and corr < 0.25: #and self.step_cntr-self.init_steps_ <= self.Nt:
            terminal = True
            reward_penalty = -5
            print('Did not form soliton ...')
            print('Spectral Corr:', corr)
        # elif self.step_cntr > self.Nt and achieved == False:
        #     terminal = True
        #     reward_penalty = -5
        #     print('Did not achieve desired spectrum ...')
        #     print('Spectral Corr:', corr)
        if self.step_cntr > int(0.4*self.Nt):
            if self.current_del_omega < self.del_omega_end or self.current_del_omega > self.del_omega_init:
                self.det_out_cntr += self.ctrl_freq
        
        if self.det_out_cntr > int(0.2*self.Nt) and corr < 0.25:
            terminal = True
            reward_penalty = -5
            print('Detuning out of range ...')
            print('Current Detuning:', self.current_del_omega.item()/(2*np.pi*1e9), 'GHz')
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
        delta_omega = self.rescale_and_quantize(action[1], self.delta_omega_min, self.delta_omega_max, self.delta_omega_step)*(2*np.pi)  # Convert GHz to rad/s
        
        for _ in range(self.ctrl_freq):
            del_omega = self.current_del_omega + delta_omega #+ self.delta_theta/(self.tR)

            self.current_del_omega = torch.clamp(del_omega, min=min(self.del_omega_end, self.del_omega_init), max=max(self.del_omega_end, self.del_omega_init))


            Fdrive_val = self.Fdrive(self.current_del_omega + self.delta_theta/(self.tR), self.t_sim_start+self.step_cntr*self.t_sim_step, self.Ain)
            u0 = self.ssfm_step(state, self.step_cntr, self.alpha, self.Dint_shift, self.current_del_omega + self.delta_theta/(self.tR), self.tR, self.gamma, \
                                self.L, 10, 1e-3, 1, self.kext, Fdrive_val)
            state = u0
            P_avg = torch.mean(torch.abs(u0)**2)  # Compute average power
            d_delta_theta_dt = -self.delta_theta / self.tau0 + self.xi * P_avg
            self.delta_theta += (1 * self.tR) * d_delta_theta_dt  # Euler step
            self.step_cntr += 1
        self.next_state = u0
        
        
        Acav = torch.sqrt(self.alpha/2)*u0*np.exp(1j*torch.pi)/np.sqrt(len(self.mu))
        Ecav = torch.fft.fftshift(torch.fft.fft(Acav))/np.sqrt(len(self.mu))

        cav = Fdrive_val*torch.sqrt(1-self.kext)
        wg = torch.sqrt(self.kext)*u0*np.exp(1j*np.pi)
        Awg = (wg + cav)/np.sqrt(len(self.mu))
        Ewg = torch.fft.fftshift(torch.fft.fft(Awg))/np.sqrt(len(self.mu))

        Acav_np = Acav.numpy()
        curr_pcav = np.sum(np.abs(Acav_np))
        self.pcav_hist.append(curr_pcav)
        if len(self.pcav_hist) > 10000:
            self.pcav_hist.pop(0)

        Ecav_dBm = 10*torch.log10(torch.abs(Ewg)**2)+30
        Ecav_dBm = torch.clamp(Ecav_dBm, min=-60, max=None) # -60 dBm is the minimum power level we want to consider
        desired_spectrum_dBm = 10*torch.log10(torch.abs(desired_spectrum)**2)+30
        desired_spectrum_dBm = torch.clamp(desired_spectrum_dBm, min=-60, max=None) # -60 dBm is the minimum power level we want to consider

        
        # pop the first element of ecav_state and append new Ecav_dBm
        # self.ecav_state = np.concatenate((self.ecav_state[1:], Ecav_dBm.cpu().numpy()[np.newaxis,:]), axis=0)
        if self.ecav_state.shape[0] >= self.seq_len:
            self.ecav_state = np.delete(self.ecav_state, 0, axis=0)
        self.ecav_state = np.concatenate((self.ecav_state, Ecav_dBm.cpu().numpy()[np.newaxis,:]), axis=0)

        reward = 4*torch.corrcoef(torch.stack([desired_spectrum_dBm, Ecav_dBm]))[0,1].item() #+ curr_pcav#+ 
        # self.primary_sidebands_flag = torch.corrcoef(torch.stack([desired_spectrum_dBm, Ecav_dBm]))[0,1]
        # penalize for high variance in power
        # if len(self.env_p_hist) > 1:
        #     power_var = np.std(self.env_p_hist)
        #     if power_var > 0.001:
        #         reward -= 2*len(self.env_p_hist) * (power_var - 0.001)
        
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

        # Fdrive_un_comp = self.Fdrive(self.delta_theta/(self.tR), self.t_sim_start+self.step_cntr*self.t_sim_step, self.Ain).cpu().numpy()
        # Fdrive_comp = Fdrive_val.cpu().numpy()
            
        
        return self.next_state, reward, done, terminal, achieved, Acav_np, self.ecav_state, Ewg.cpu().numpy()#, Fdrive_un_comp, Fdrive_comp
# %%
# torch seed
# torch.manual_seed(0)
env = RL_MRR_Env(seq_len=100, p_max=0.2, p_min=0.05, ctrl_freq=100, thermal_effect='low',\
                  delta_omega_min=-1e6, delta_omega_max=1e6, delta_omega_step=1e4)
fpmp = env.sim_tensor['f_pmp'].item()
freq = (fpmp + np.arange(-220,221)*env.FSR.item())*1e-12
# %%
desired_spectrum = loadmat('desired_spec2.mat')['Ewg'][0]
desired_spectrum_dBm = 10*np.log10(np.abs(desired_spectrum)**2)+30 
desired_spectrum_dBm = np.clip(desired_spectrum_dBm, -60, None)
desired_spectrum_tensor = torch.tensor(desired_spectrum, device=DEVICE, dtype=torch.complex128)
# %%
config = {
    'input_dim': [env.seq_len, 300+2],
    'n_actions': 2,
    'alpha': 3e-4,
    'beta': 3e-4,
    'mem_size': int(1e6),
    'run_name': 'mrr_sac_cluster_delayed_toptica_pow_ton_un_norm_mod_v3',
    'batch_size': 128,
    'dist': 'beta', # 'beta' or 'normal'
    'train':False,
    'p_max': env.p_max,
    'p_min': env.p_min,
    'fc_dim':256,
    'use_per':True,
    'delta_omega_min': env.delta_omega_min,  # Minimum detuning in Hz
    'delta_omega_max': env.delta_omega_max,   # Maximum detuning in Hz
    'delta_omega_step': env.delta_omega_step,   # Step size for detuning in
    'bidirectional': False,  # Whether to use bidirectional GRU
    }
# %%

from rl_codes.sac_v3 import SACAgent
agent = SACAgent(input_dim=config['input_dim'], n_actions=config['n_actions'], alpha=config['alpha'], beta=config['beta'],
                mem_size=config['mem_size'], batch_size=config['batch_size'], dist=config['dist'], run_name=config['run_name'],
                eval_mode=not(torch.cuda.is_available()), fc_dim=config['fc_dim'], use_per=config['use_per'], bidir=config['bidirectional'])
print(agent.actor)
print(agent.critic_1)
agent.load_models()
# %%
# '''
state, acav, ecav = env.reset(10000)
den = env.p_max - env.p_min
obs = np.concatenate((ecav[:,len(env.mu)//2-150:len(env.mu)//2+150]/10,env.power*np.ones((env.seq_len,1))/den,np.zeros((env.seq_len,1))),axis=1)
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
delta_theta = []
e_wg_hist = []
det_hist = []
# Fdrive_uncomp_hist = []
# Fdrive_comp_hist = []
while not done:
# for idx in tqdm(range(env.init_steps_, int(env.max_steps)), ncols=120):
    # perform random actions
    # try:
        action = agent.choose_action(obs, True)

        next_state, reward, done, terminal, achieved, acav_, ecav_, e_wg = env.step(state, action, desired_spectrum_tensor)
        state = next_state
        ecav = ecav_
        ecav_obs = np.concatenate((ecav_[-1,len(env.mu)//2-150:len(env.mu)//2+150]/10, env.power/den, 5*env.current_del_omega/(env.del_omega_init - env.del_omega_end)), axis=0)
        obs_ = np.concatenate((obs[1:], ecav_obs[np.newaxis,:]), axis=0)
        obs = obs_ 
        score += reward
        curr_pcav = np.sum(np.abs(acav_)**2)
        pcav_hist.append(curr_pcav)
        r_hist.append(reward)
        action_hist.append(action)
        delta_theta.append(env.delta_theta.item())
        det_hist.append(env.current_del_omega.item()/(2*np.pi))  # Convert rad/s to GHz
        e_wg_hist.append(e_wg)
       
        acav_hist.append(acav_)
        idx += env.ctrl_freq
        pbar.update(env.ctrl_freq)
pbar.close()

print('Test score %.2f' % score, 'at step %d' % idx)
# %%
# %%
import os

# Create save directory if not exists
save_dir = os.path.join('./results', agent.run_name, env.thermal_effect, 'new3')
os.makedirs(save_dir, exist_ok=True)
plt.style.use('physrev.mplstyle')
# %%
# …existing code above…

# import matplotlib.animation as animation

# # stack histories into arrays (T time-steps × M modes)
# f_uncomp = np.array(Fdrive_uncomp_hist)
# f_comp   = np.array(Fdrive_comp_hist)

# # build frequency axis in THz
# fpmp      = env.sim_tensor['f_pmp'].item()
# num_modes = f_uncomp.shape[1]
# freq      = (fpmp + np.arange(-num_modes//2, num_modes//2 + num_modes%2)*env.FSR.item())*1e-12

# # compute FFT and shift zero-freq to center
# FFT_uncomp = np.fft.fftshift(np.fft.fft(f_uncomp, axis=1), axes=1)
# FFT_comp   = np.fft.fftshift(np.fft.fft(f_comp,   axis=1), axes=1)

# # power in dBm = 10·log10(|FFT|²) + 30
# P_uncomp = 10*np.log10(np.abs(FFT_uncomp)**2 + 1e-12) + 30
# P_comp   = 10*np.log10(np.abs(FFT_comp)**2   + 1e-12) + 30

# if idx > int(0.5*env.max_steps):
# # create figure
#     fig, (ax1, ax2) = plt.subplots(2,1, sharex=True, figsize=(8,6))
#     line1, = ax1.plot(freq, P_uncomp[0], color='C0')
#     ax1.set_ylabel('Uncomp. Power (dBm)')
#     ax1.set_ylim(P_uncomp.min(), P_uncomp.max())
#     line2, = ax2.plot(freq, P_comp[0], color='C1')
#     ax2.set_ylabel('Comp. Power (dBm)')
#     ax2.set_xlabel('Frequency (THz)')
#     ax2.set_ylim(P_comp.min(), P_comp.max())

#     def update(frame):
#         line1.set_ydata(P_uncomp[frame])
#         line2.set_ydata(P_comp[frame])
#         ax1.set_title(f'Time step {frame+1}/{P_uncomp.shape[0]}')
#         return line1, line2

#     ani = animation.FuncAnimation(
#         fig, update,
#         frames=P_uncomp.shape[0],
#         interval=100,
#         blit=True
#     )

#     # save to mp4 (requires ffmpeg)

#     out_path = os.path.join(save_dir, 'fdrive_evolution_freq.mp4')
#     ani.save(out_path, fps=10, dpi=200)
#     print(f'Animation saved to {out_path}')
# %%
# find correlation between the obtained pcav and r_hist[:,-1]
plt.figure(figsize=(10, 6))
plt.plot(1e3*np.array(pcav_hist), linewidth=1.5)
plt.grid()
plt.xlabel('Tuning Steps', fontsize=16)
plt.ylabel(r'$P_{cav}$ (mW)', fontsize=16)
plt.title(env.thermal_effect + ' thermal effect', fontsize=16, fontweight='bold')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.tight_layout()
mod_pow = str(env.power[0]).replace('.','_')
if idx > int(0.5*env.max_steps):
    plt.savefig(os.path.join(save_dir, mod_pow + '_'+ env.thermal_effect + '_pcav_spec_all_ctrl.png'))
    # save as svg also
    plt.savefig(os.path.join(save_dir, mod_pow + '_'+ env.thermal_effect + '_pcav_spec_all_ctrl.svg'), format='svg')
plt.show()
# %%
import matplotlib.ticker as ticker

plt.figure(figsize=(14,4))
plt.imshow(1e3*np.abs(np.array(acav_hist).T)**2, aspect='auto', cmap='jet',\
            extent=[0, len(acav_hist), -1e12*env.tR.item()/2, 1e12*env.tR.item()/2])
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=16)
cbar.set_label(r'Power $(mW)$', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
# Set x-ticks to exponent format
formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(True)
formatter.set_powerlimits((-1, 1))
plt.gca().xaxis.set_major_formatter(formatter)
plt.xlabel('Tuning Steps', fontsize=14)
plt.ylabel(r'$t_R (ps)$', fontsize=14)
plt.title(env.thermal_effect + ' thermal effect', fontsize=16, fontweight='bold')
mod_pow = str(env.power[0]).replace('.','_')
plt.tight_layout()
if idx > int(0.5*env.max_steps):
    plt.savefig(os.path.join(save_dir, mod_pow + '_'+ env.thermal_effect + '_ecav_hist_spec_all_ctrl.png'))
    plt.savefig(os.path.join(save_dir, mod_pow + '_'+ env.thermal_effect + '_ecav_hist_spec_all_ctrl.svg'), format='svg')
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
plt.title(env.thermal_effect + ' thermal effect', fontsize=16, fontweight='bold')
mod_pow = str(env.power[0]).replace('.','_')
plt.tight_layout()
if idx > int(0.5*env.max_steps):
    plt.savefig(os.path.join(save_dir, mod_pow + '_'+ env.thermal_effect + '_ecav_hist_ifft_spec_all_ctrl.png'))
    # save as svg also
    plt.savefig(os.path.join(save_dir, mod_pow + '_'+ env.thermal_effect + '_ecav_hist_ifft_spec_all_ctrl.svg'), format='svg')
plt.show()
# %%
plt.figure(figsize=(14,4))
e_wg_hist = np.array(e_wg_hist).T
ewg_dBm = 10*np.log10(np.abs(e_wg_hist)**2)+30
ewg_dBm = np.clip(ewg_dBm, -60, 50)
plt.imshow(ewg_dBm, aspect='auto', cmap='jet'\
            ,extent=[0, len(acav_hist), env.mu.min().item(), env.mu.max().item()])
plt.xlabel('Tuning Steps', fontsize=18)
plt.ylabel(r'$\mu$' +'(rel)', fontsize=18)
cbar = plt.colorbar()
# set colorbar ticks size
cbar.ax.tick_params(labelsize=16)
cbar.set_label(r'Power $(dBm)$', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.title(env.thermal_effect + ' thermal effect', fontsize=16, fontweight='bold')
mod_pow = str(env.power[0]).replace('.','_')
plt.tight_layout()
if idx > int(0.5*env.max_steps):
    plt.savefig(os.path.join(save_dir, mod_pow + '_'+ env.thermal_effect + '_ewg_hist_spec_all_ctrl.png'))
    # save as svg also
    plt.savefig(os.path.join(save_dir, mod_pow + '_'+ env.thermal_effect + '_ewg_hist_spec_all_ctrl.svg'), format='svg')
plt.show()
# %% Reward Plot
plt.figure(figsize=(10, 6))
plt.plot(r_hist)
plt.xlabel('Tuning Steps', fontsize=16)
plt.ylabel('Reward ', fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.grid()
plt.title(env.thermal_effect + ' thermal effect', fontsize=16, fontweight='bold')
plt.tight_layout()
mod_pow = str(env.power[0]).replace('.','_')
if idx > int(0.5*env.max_steps):
    plt.savefig(os.path.join(save_dir, mod_pow + '_'+ env.thermal_effect + '_rewards_spec_all_ctrl.png'))
    # save as svg also
    plt.savefig(os.path.join(save_dir, mod_pow + '_'+ env.thermal_effect + '_rewards_spec_all_ctrl.svg'), format='svg')
plt.show()
# %%
desired_spectrum_dBm_ = np.clip(desired_spectrum_dBm, -60, 10)
mse = np.linalg.norm(desired_spectrum_dBm_-ecav[-1], ord=2)
corr = np.corrcoef(desired_spectrum_dBm_, ecav_[-1])[0, 1]
print('MSE:', mse, 'Corr:', corr)
# %%
# desired_spectrum_dBm = 10*torch.log10(torch.abs(desired_spectrum)**2)+30
obtained_spectrum = 10*np.log10(np.abs(e_wg)**2) + 30
desired_spectrum = loadmat('desired_spec2.mat')['Ewg'][0]
desired_spectrum_dBm = 10*np.log10(np.abs(desired_spectrum)**2)+30

plt.figure(figsize=(14,4))
plt.vlines(np.arange(-220,221, 1), -60*np.ones(len(ecav[-1])), obtained_spectrum, \
           label='Obtained Spectrum',alpha=1, linewidth=1.5)
plt.vlines(np.arange(-220,221, 1), -60*np.ones(len(desired_spectrum)),\
            desired_spectrum_dBm, color='red', label='Desired Spectrum',alpha=0.5, linewidth=1.5)
plt.xlabel('Rel. Mode no.', fontsize=18)
plt.ylabel('Power(dBm)', fontsize=18)
plt.grid()
plt.ylim(-90,30)
plt.xlim(-150, 150)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.legend(fontsize=18,loc='lower center')
plt.title(env.thermal_effect + ' thermal effect', fontsize=16, fontweight='bold')
mod_pow = str(env.power[0]).replace('.','_')
plt.tight_layout()
if idx > int(0.5*env.max_steps):
    plt.savefig(os.path.join(save_dir, mod_pow + '_'+ env.thermal_effect + '_ecav_spec_all_ctrl_modes.png'))
    # save as svg also
    plt.savefig(os.path.join(save_dir, mod_pow + '_'+ env.thermal_effect + '_ecav_spec_all_ctrl_modes.svg'), format='svg')
plt.show()

plt.figure(figsize=(14,4))
plt.vlines(freq, -60*np.ones(len(ecav[-1])), obtained_spectrum, \
           label='Obtained Spectrum',alpha=1, linewidth=1.5)
plt.vlines(freq, -60*np.ones(len(desired_spectrum)),\
            desired_spectrum_dBm, color='red', label='Desired Spectrum',alpha=0.5, linewidth=1.5)
plt.xlabel('Freq. (THz)', fontsize=18)
plt.ylabel('Power(dBm)', fontsize=18)
plt.grid()
plt.ylim(-90,30)
plt.xlim(freq[220-150], freq[220+150])
plt.xticks(fontsize=18, fontweight='bold')
plt.yticks(fontsize=18, fontweight='bold')
plt.legend(fontsize=18, loc='lower center')
plt.title(env.thermal_effect + ' thermal effect', fontsize=16, fontweight='bold')
mod_pow = str(env.power[0]).replace('.','_')
plt.tight_layout()
if idx > int(0.5*env.max_steps):
    plt.savefig(os.path.join(save_dir, mod_pow + '_'+ env.thermal_effect + '_ecav_spec_all_ctrl_freq.png'))
    # save as svg also
    plt.savefig(os.path.join(save_dir, mod_pow + '_'+ env.thermal_effect + '_ecav_spec_all_ctrl_freq.svg'), format='svg')
plt.show()
# %%
action_hist = np.array(action_hist)
# del_detuning = env.rescale_and_quantize(action_hist[:,1])
# det_start = env.del_omega_init.item()/(2*np.pi)
# detuning_array = det_start + np.cumsum(del_detuning)
# detuning_array = np.clip(detuning_array, env.del_omega_end.item()/(2*np.pi), env.del_omega_init.item()/(2*np.pi))
detuning_array = np.array(det_hist)
plt.figure(figsize=(10, 6))
plt.plot(detuning_array*1e-9, linewidth=1.5)
plt.xlabel('Tuning Steps', fontsize=18)
plt.ylabel('Pump detuning (GHz)', fontsize=18)
plt.grid()
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.gca().xaxis.set_major_formatter(formatter)
plt.title(env.thermal_effect + ' thermal effect', fontsize=16, fontweight='bold')
mod_pow = str(env.power[0]).replace('.','_')
plt.tight_layout()
if idx > int(0.5*env.max_steps):
    plt.savefig(os.path.join(save_dir, mod_pow + '_'+ env.thermal_effect + '_actions_spec_all_ctrl.png'))
    # save as svg also
    plt.savefig(os.path.join(save_dir, mod_pow + '_'+ env.thermal_effect + '_actions_spec_all_ctrl.svg'), format='svg')
plt.show()
# %%
import fractions

# Convert to units of kappa
kappa = env.un_norm_kappa.item()/(2*np.pi)
detuning_kappa = detuning_array / kappa

plt.figure(figsize=(10, 6))
plt.plot(detuning_kappa, linewidth=1.5)
plt.xlabel('Tuning Steps', fontsize=18)
plt.ylabel(r'Pump detuning ($\kappa$ units)', fontsize=18)
plt.grid()
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)

# Set y-ticks at reasonable intervals (e.g., 0.5)
ymin, ymax = plt.ylim()
yticks = np.arange(np.floor(ymin*2)/2, np.ceil(ymax*2)/2 + 0.01, 0.5)

def frac_label(val):
    frac = fractions.Fraction(val).limit_denominator(8)
    if frac.numerator == 0:
        return r"$0$"
    elif frac.denominator == 1:
        return rf"${frac.numerator}\,\kappa$"
    elif frac.numerator == 1:
        return rf"$\frac{{1}}{{{frac.denominator}}}\,\kappa$"
    elif frac.numerator == -1:
        return rf"$-\frac{{1}}{{{frac.denominator}}}\,\kappa$"
    else:
        return rf"$\frac{{{frac.numerator}}}{{{frac.denominator}}}\,\kappa$"

ytick_labels = [frac_label(y) for y in yticks]
plt.yticks(yticks, ytick_labels, fontsize=18)

plt.title(env.thermal_effect + ' thermal effect', fontsize=16, fontweight='bold')
mod_pow = str(env.power[0]).replace('.','_')
plt.tight_layout()
if idx > int(0.5*env.max_steps):
    plt.savefig(os.path.join(save_dir, mod_pow + '_'+ env.thermal_effect + '_actions_spec_all_ctrl_kappa.png'))
    plt.savefig(os.path.join(save_dir, mod_pow + '_'+ env.thermal_effect + '_actions_spec_all_ctrl_kappa.svg'), format='svg')
plt.show()
# %%
plt.figure(figsize=(10, 6))
plt.plot(env.rescale_power(action_hist[:,0]), linewidth=1.5)
plt.xlabel('Tuning Steps', fontsize=18)
plt.ylabel('Pump Power (mW)', fontsize=18)
plt.grid()
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.title(env.thermal_effect + ' thermal effect', fontsize=16, fontweight='bold')
mod_pow = str(env.power[0]).replace('.','_')
plt.tight_layout()
if idx > int(0.5*env.max_steps):
    plt.savefig(os.path.join(save_dir, mod_pow + '_'+ env.thermal_effect + '_actions_power_spec_all_ctrl.png'))
    # save as svg also
    plt.savefig(os.path.join(save_dir, mod_pow + '_'+ env.thermal_effect + '_actions_power_spec_all_ctrl.svg'), format='svg')
plt.show()
# %%
plt.figure(figsize=(10, 6))
plt.plot(np.array(delta_theta)*1e-6/(2*np.pi*env.tR.item()), linewidth=1.5)
plt.xlabel('Tuning Steps', fontsize=18)
plt.ylabel(r'$f _{\Theta}$ (MHz)', fontsize=18)
plt.grid()
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.title(env.thermal_effect + ' thermal effect', fontsize=16, fontweight='bold')
mod_pow = str(env.power[0]).replace('.','_')
plt.tight_layout()
if idx > int(0.5*env.max_steps):
    plt.savefig(os.path.join(save_dir, mod_pow + '_'+ env.thermal_effect + '_delta_theta_spec_all_ctrl.png'))
    # save as svg also
    plt.savefig(os.path.join(save_dir, mod_pow + '_'+ env.thermal_effect + '_delta_theta_spec_all_ctrl.svg'), format='svg')
plt.show()

# %%
plt.figure(figsize=(10, 6))
plt.plot(detuning_array*1e-9, np.array(delta_theta)*1e-9/(2*np.pi*env.tR.item()), linewidth=1.5)
plt.xlabel(r'$f_{det}$ (GHz)', fontsize=18)
plt.ylabel(r'$f _{\Theta}$ (GHz)', fontsize=18)
plt.grid()
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.title(env.thermal_effect + ' thermal effect', fontsize=16, fontweight='bold')
mod_pow = str(env.power[0]).replace('.','_')
plt.tight_layout()
if idx > int(0.5*env.max_steps):
    plt.savefig(os.path.join(save_dir, mod_pow + '_'+ env.thermal_effect + '_detuning_delta_theta_spec_all_ctrl.png'))
    # save as svg also
    plt.savefig(os.path.join(save_dir, mod_pow + '_'+ env.thermal_effect + '_detuning_delta_theta_spec_all_ctrl.svg'), format='svg')
plt.show()
'''
# %%
def plot_all_results(env, save_dir, idx, pcav_hist, acav_hist, e_wg_hist, r_hist, det_hist, delta_theta, action_hist, freq, ecav, obtained_spectrum, desired_spectrum_dBm, thermal_effect):
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    import os
    import numpy as np
    import fractions

    plt.style.use('physrev.mplstyle')
    mod_pow = str(env.power[0]).replace('.','_')
    # Convert tuning steps to time in microseconds
    time_axis = np.arange(len(pcav_hist)) * 100 * env.tR.item() * 1e6  # microseconds

    # 1. Pcav history
    plt.figure(figsize=(10, 6))
    plt.plot(time_axis, 1e3*np.array(pcav_hist), linewidth=1.5)
    plt.grid()
    plt.xlabel(r'Time ($\mu$s)', fontsize=16)
    plt.ylabel(r'$P_{cav}$ (mW)', fontsize=16)
    plt.title(f'{thermal_effect} thermal effect', fontsize=16, fontweight='bold')
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()    
    plt.savefig(os.path.join(save_dir, mod_pow + f'_{thermal_effect}_run_{idx}_pcav_spec_all_ctrl.png'))
    plt.savefig(os.path.join(save_dir, mod_pow + f'_{thermal_effect}_run_{idx}_pcav_spec_all_ctrl.svg'), format='svg')
    plt.show()

    # 2. Acav time evolution
    plt.figure(figsize=(14,4))
    plt.imshow(1e3*np.abs(np.array(acav_hist).T)**2, aspect='auto', cmap='jet',
               extent=[time_axis[0], time_axis[-1], -1e12*env.tR.item()/2, 1e12*env.tR.item()/2])
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=16)
    cbar.set_label(r'Power $(mW)$', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-1, 1))
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.xlabel(r'Time ($\mu$s)', fontsize=14)
    plt.ylabel(r'$t_R (ps)$', fontsize=14)
    plt.title(f'{thermal_effect} thermal effect', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, mod_pow + f'_{thermal_effect}_run_{idx}_ecav_hist_spec_all_ctrl.png'))
    plt.savefig(os.path.join(save_dir, mod_pow + f'_{thermal_effect}_run_{idx}_ecav_hist_spec_all_ctrl.svg'), format='svg')
    plt.show()

    # 3. Spectrum evolution (FFT of acav_hist)
    plt.figure(figsize=(14,4))
    spectrum = np.fft.fftshift(np.fft.fft(np.array(acav_hist).T, axis=0), axes=0)
    spectrum_dBm = 10*np.log10(np.abs(spectrum)**2)+30
    spectrum_dBm = np.clip(spectrum_dBm, -60, 10)
    plt.imshow(spectrum_dBm, aspect='auto', cmap='jet',
               extent=[time_axis[0], time_axis[-1], env.mu.min().item(), env.mu.max().item()])
    plt.xlabel(r'Time ($\mu$s)', fontsize=18)
    plt.ylabel(r'$\mu$' +'(rel)', fontsize=18)
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=16)
    cbar.set_label(r'Power $(dBm)$', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.title(f'{thermal_effect} thermal effect', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, mod_pow + f'_{thermal_effect}_run_{idx}_ecav_hist_ifft_spec_all_ctrl.png'))
    plt.savefig(os.path.join(save_dir, mod_pow + f'_{thermal_effect}_run_{idx}_ecav_hist_ifft_spec_all_ctrl.svg'), format='svg')
    plt.show()

    # 4. Ewg spectrum evolution
    plt.figure(figsize=(14,4))
    e_wg_hist = np.array(e_wg_hist).T
    ewg_dBm = 10*np.log10(np.abs(e_wg_hist)**2)+30
    ewg_dBm = np.clip(ewg_dBm, -60, 50)
    plt.imshow(ewg_dBm, aspect='auto', cmap='jet',
               extent=[time_axis[0], time_axis[-1], env.mu.min().item(), env.mu.max().item()])
    plt.xlabel(r'Time ($\mu$s)', fontsize=18)
    plt.ylabel(r'$\mu$' +'(rel)', fontsize=18)
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=16)
    cbar.set_label(r'Power $(dBm)$', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.title(f'{thermal_effect} thermal effect', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, mod_pow + f'_{thermal_effect}_run_{idx}_ewg_hist_spec_all_ctrl.png'))
    plt.savefig(os.path.join(save_dir, mod_pow + f'_{thermal_effect}_run_{idx}_ewg_hist_spec_all_ctrl.svg'), format='svg')
    plt.show()

    # 5. Reward plot
    plt.figure(figsize=(10, 6))
    plt.plot(time_axis, r_hist)
    plt.xlabel(r'Time ($\mu$s)', fontsize=16)
    plt.ylabel('Reward ', fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid()
    plt.title(f'{thermal_effect} thermal effect', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, mod_pow + f'_{thermal_effect}_run_{idx}_rewards_spec_all_ctrl.png'))
    plt.savefig(os.path.join(save_dir, mod_pow + f'_{thermal_effect}_run_{idx}_rewards_spec_all_ctrl.svg'), format='svg')
    plt.show()

    # 8. Detuning array (GHz)
    detuning_array = np.array(det_hist)
    plt.figure(figsize=(10, 6))
    plt.plot(time_axis, detuning_array*1e-9, linewidth=1.5, label='Pump detuning')
    plt.plot(time_axis, np.array(delta_theta)*1e-9/(2*np.pi*env.tR.item()), linewidth=1.5 , label=r'$f_{\Theta}$')
    plt.xlabel(r'Time ($\mu$s)', fontsize=18)
    plt.ylabel('Freq (GHz)', fontsize=18)
    plt.grid()
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(fontsize=18)
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-1, 1))
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.title(f'{thermal_effect} thermal effect', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, mod_pow + f'_{thermal_effect}_run_{idx}_actions_spec_all_ctrl.png'))
    plt.savefig(os.path.join(save_dir, mod_pow + f'_{thermal_effect}_run_{idx}_actions_spec_all_ctrl.svg'), format='svg')
    plt.show()

    # 9. Detuning in kappa units
    kappa = env.un_norm_kappa.item()/(2*np.pi)
    detuning_kappa = detuning_array / kappa
    plt.figure(figsize=(10, 6))
    plt.plot(time_axis, detuning_kappa, linewidth=1.5)
    plt.xlabel(r'Time ($\mu$s)', fontsize=18)
    plt.ylabel(r'Pump detuning ($\kappa$ units)', fontsize=18)
    plt.grid()
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    ymin, ymax = plt.ylim()
    yticks = np.arange(np.floor(ymin*2)/2, np.ceil(ymax*2)/2 + 0.01, 0.5)
    def frac_label(val):
        frac = fractions.Fraction(val).limit_denominator(8)
        if frac.numerator == 0:
            return r"$0$"
        elif frac.denominator == 1:
            return rf"${frac.numerator}\,\kappa$"
        elif frac.numerator == 1:
            return rf"$\frac{{1}}{{{frac.denominator}}}\,\kappa$"
        elif frac.numerator == -1:
            return rf"$-\frac{{1}}{{{frac.denominator}}}\,\kappa$"
        else:
            return rf"$\frac{{{frac.numerator}}}{{{frac.denominator}}}\,\kappa$"
    ytick_labels = [frac_label(y) for y in yticks]
    plt.yticks(yticks, ytick_labels, fontsize=18)
    plt.title(f'{thermal_effect} thermal effect', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, mod_pow + f'_{thermal_effect}_run_{idx}_actions_spec_all_ctrl_kappa.png'))
    plt.savefig(os.path.join(save_dir, mod_pow + f'_{thermal_effect}_run_{idx}_actions_spec_all_ctrl_kappa.svg'), format='svg')
    plt.show()

    # 10. Pump power
    plt.figure(figsize=(10, 6))
    plt.plot(time_axis, env.rescale_power(action_hist[:,0]), linewidth=1.5)
    plt.xlabel(r'Time ($\mu$s)', fontsize=18)
    plt.ylabel('Pump Power (mW)', fontsize=18)
    plt.grid()
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.title(f'{thermal_effect} thermal effect', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, mod_pow + f'_{thermal_effect}_run_{idx}_actions_power_spec_all_ctrl.png'))
    plt.savefig(os.path.join(save_dir, mod_pow + f'_{thermal_effect}_run_{idx}_actions_power_spec_all_ctrl.svg'), format='svg')
    plt.show()

    # # 11. Delta theta
    # plt.figure(figsize=(10, 6))
    # plt.plot(time_axis, np.array(delta_theta)*1e-6/(2*np.pi*env.tR.item()), linewidth=1.5)
    # plt.xlabel(r'Time ($\mu$s)', fontsize=18)
    # plt.ylabel(r'$f _{\Theta}$ (MHz)', fontsize=18)
    # plt.grid()
    # plt.xticks(fontsize=18)
    # plt.yticks(fontsize=18)
    # plt.title(f'{thermal_effect} thermal effect', fontsize=16, fontweight='bold')
    # plt.tight_layout()
    # plt.savefig(os.path.join(save_dir, mod_pow + f'_{thermal_effect}_delta_theta_spec_all_ctrl.png'))
    # plt.savefig(os.path.join(save_dir, mod_pow + f'_{thermal_effect}_delta_theta_spec_all_ctrl.svg'), format='svg')
    # plt.show()

    # 12. Detuning vs delta theta
    plt.figure(figsize=(10, 6))
    plt.plot(detuning_array*1e-9, np.array(delta_theta)*1e-9/(2*np.pi*env.tR.item()), linewidth=1.5)
    plt.xlabel(r'Pump detuning (GHz)', fontsize=18)
    plt.ylabel(r'$f _{\Theta}$ (GHz)', fontsize=18)
    plt.grid()
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.title(f'{thermal_effect} thermal effect', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, mod_pow + f'_{thermal_effect}_run_{idx}_detuning_delta_theta_spec_all_ctrl.png'))
    plt.savefig(os.path.join(save_dir, mod_pow + f'_{thermal_effect}_run_{idx}_detuning_delta_theta_spec_all_ctrl.svg'), format='svg')
    plt.show()

    # 6. Obtained vs desired spectrum (mode index)
    plt.figure(figsize=(14,4))
    plt.vlines(np.arange(-220,221, 1), -60*np.ones(len(ecav[-1])), obtained_spectrum,
               label='Obtained Spectrum',alpha=1, linewidth=1.5)
    plt.vlines(np.arange(-220,221, 1), -60*np.ones(len(desired_spectrum_dBm)),
               desired_spectrum_dBm, color='red', label='Desired Spectrum',alpha=0.5, linewidth=1.5)
    plt.xlabel('Rel. Mode no.', fontsize=18)
    plt.ylabel('Power(dBm)', fontsize=18)
    plt.grid()
    plt.ylim(-90,30)
    plt.xlim(-150, 150)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(fontsize=18,loc='lower center')
    plt.title(f'{thermal_effect} thermal effect', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, mod_pow + f'_{thermal_effect}_run_{idx}_ecav_spec_all_ctrl_modes.png'))
    plt.savefig(os.path.join(save_dir, mod_pow + f'_{thermal_effect}_run_{idx}_ecav_spec_all_ctrl_modes.svg'), format='svg')
    plt.show()

    # 7. Obtained vs desired spectrum (frequency)
    plt.figure(figsize=(14,4))
    plt.vlines(freq, -60*np.ones(len(ecav[-1])), obtained_spectrum,
               label='Obtained Spectrum',alpha=1, linewidth=1.5)
    plt.vlines(freq, -60*np.ones(len(desired_spectrum_dBm)),
               desired_spectrum_dBm, color='red', label='Desired Spectrum',alpha=0.5, linewidth=1.5)
    plt.xlabel('Freq. (THz)', fontsize=18)
    plt.ylabel('Power(dBm)', fontsize=18)
    plt.grid()
    plt.ylim(-90,30)
    plt.xlim(freq[220-150], freq[220+150])
    plt.xticks(fontsize=18, fontweight='bold')
    plt.yticks(fontsize=18, fontweight='bold')
    plt.legend(fontsize=18, loc='lower center')
    plt.title(f'{thermal_effect} thermal effect', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, mod_pow + f'_{thermal_effect}_run_{idx}_ecav_spec_all_ctrl_freq.png'))
    plt.savefig(os.path.join(save_dir, mod_pow + f'_{thermal_effect}_run_{idx}_ecav_spec_all_ctrl_freq.svg'), format='svg')
    plt.show()
    print('All plots saved successfully in', save_dir)
# %%
def run_test_processes(run_id, save_dir):
    # Re-create environment and agent inside the process
    env = RL_MRR_Env(seq_len=100, p_max=0.2, p_min=0.05, ctrl_freq=100, thermal_effect='low')
    desired_spectrum = loadmat('desired_spec.mat')['Ecav'][0]
    desired_spectrum_tensor = torch.tensor(desired_spectrum, device=DEVICE, dtype=torch.complex128)
    from rl_codes.sac_v3 import SACAgent
    config = {
    'input_dim': [env.seq_len, 300+2],
    'n_actions': 2,
    'alpha': 3e-4,
    'beta': 3e-4,
    'mem_size': int(1e6),
    'run_name': 'mrr_sac_cluster_delayed_toptica_pow_ton_un_norm_mod_v3',
    'batch_size': 256,
    'dist': 'beta', # 'beta' or 'normal'
    'train':False,
    'p_max': env.p_max,
    'p_min': env.p_min,
    'fc_dim':256,
    'use_per':True,
    'delta_omega_min': env.delta_omega_min,  # Minimum detuning in Hz
    'delta_omega_max': env.delta_omega_max,   # Maximum detuning in Hz
    'delta_omega_step': env.delta_omega_step,   # Step size for detuning in
    'bidirectional': False,  # Whether to use bidirectional detuning
    }
    agent = SACAgent(input_dim=config['input_dim'], n_actions=config['n_actions'], alpha=config['alpha'], beta=config['beta'],
                    mem_size=config['mem_size'], batch_size=config['batch_size'], dist=config['dist'], run_name=config['run_name'],
                    eval_mode=not(torch.cuda.is_available()), fc_dim=config['fc_dim'], use_per=config['use_per'], bidir=config['bidirectional'])

    agent.load_models()
    # 

    state, _, ecav = env.reset(10000)
    den = env.p_max - env.p_min
    obs = np.concatenate((ecav[:,len(env.mu)//2-150:len(env.mu)//2+150]/10,env.power*np.ones((env.seq_len,1))/den,np.zeros((env.seq_len,1))),axis=1)
    print('Chosen power:', env.power)
    r_hist = []
    action_hist = []
    score = 0
    done = False
    pcav_hist = []
    acav_hist = []
    e_wg_hist = []
    pbar = tqdm(total=env.max_steps-env.init_steps_, ncols=120, position=run_id, desc=f'Run {run_id}')
    idx = 0
    delta_theta = []
    det_hist = []
    terminal = False
    while not done:
        action = agent.choose_action(obs, True)
        next_state, reward, done, terminal, achieved, acav_, ecav_, e_wg = env.step(state, action, desired_spectrum_tensor)
        state = next_state
        ecav = ecav_
        ecav_obs = np.concatenate((ecav_[-1,len(env.mu)//2-150:len(env.mu)//2+150]/10,
                                   env.power/den,
                                   5*env.current_del_omega/(env.del_omega_init - env.del_omega_end)), axis=0)
        obs = np.concatenate((obs[1:], ecav_obs[np.newaxis,:]), axis=0)
        score += reward
        curr_pcav = np.sum(np.abs(acav_)**2)
        pcav_hist.append(curr_pcav)
        acav_hist.append(acav_)
        e_wg_hist.append(e_wg)
        r_hist.append(reward)
        action_hist.append(action)
        delta_theta.append(env.delta_theta.item())
        det_hist.append(env.current_del_omega.item()/(2*np.pi))  # Convert rad/s to GHz
        idx += env.ctrl_freq
        pbar.update(env.ctrl_freq)
    pbar.close()

    
    if terminal ==False:
        print(f'Run {run_id} completed with score {score} at step {idx}')
        freq = (env.sim_tensor['f_pmp'].item() + np.arange(-220,221)*env.FSR.item())*1e-12
        obtained_spectrum = 10*np.log10(np.abs(e_wg_hist[-1])**2) + 30
        desired_spectrum = loadmat('desired_spec2.mat')['Ewg'][0]
        desired_spectrum_dBm = 10*np.log10(np.abs(desired_spectrum)**2)+30
        # desired_spectrum_dBm = np.clip(desired_spectrum_dBm, -60, None)
        # Call plot_all_results
        plot_all_results(
            env=env,
            save_dir=save_dir,
            idx=run_id,
            pcav_hist=pcav_hist,
            acav_hist=acav_hist,
            e_wg_hist=e_wg_hist,
            r_hist=r_hist,
            det_hist=det_hist,
            delta_theta=delta_theta,
            action_hist=np.array(action_hist),
            freq=freq,
            ecav=ecav,
            obtained_spectrum=obtained_spectrum,
            desired_spectrum_dBm=desired_spectrum_dBm,
            thermal_effect=env.thermal_effect
        )
        # np.save(os.path.join(save_dir, str(run_id) + '_p_cav.npy'), np.array(pcav_hist))
        # np.save(os.path.join(save_dir, str(run_id) + '_detuning_theta_sum.npy'), -1*np.array(det_hist) + np.array(delta_theta))
# %%
# def run_test_processes(run_id, save_dir, result_queue):
#     # Re-create environment and agent inside the process
#     env = RL_MRR_Env(seq_len=100, p_max=0.2, p_min=0.05, ctrl_freq=100, thermal_effect='low')
#     desired_spectrum = loadmat('desired_spec.mat')['Ecav'][0]
#     desired_spectrum_tensor = torch.tensor(desired_spectrum, device=DEVICE, dtype=torch.complex128)
#     from rl_codes.sac_v3 import SACAgent
#     config = {
#         'input_dim': [env.seq_len, 300+2],
#         'n_actions': 2,
#         'alpha': 3e-4,
#         'beta': 3e-4,
#         'mem_size': int(1e6),
#         'run_name': 'mrr_sac_cluster_delayed_toptica_pow_ton_un_norm',
#         'batch_size': 256,
#         'dist': 'beta',
#         'train': False,
#         'p_max': env.p_max,
#         'p_min': env.p_min,
#         'fc_dim': 256,
#         'use_per': True,
#         'delta_omega_min': env.delta_omega_min,
#         'delta_omega_max': env.delta_omega_max,
#         'delta_omega_step': env.delta_omega_step,
#         'bidirectional': False,
#     }
#     agent = SACAgent(input_dim=config['input_dim'], n_actions=config['n_actions'], alpha=config['alpha'], beta=config['beta'],
#                      mem_size=config['mem_size'], batch_size=config['batch_size'], dist=config['dist'], run_name=config['run_name'],
#                      eval_mode=not(torch.cuda.is_available()), fc_dim=config['fc_dim'], use_per=config['use_per'], bidir=config['bidirectional'])

#     agent.load_models()

#     state, _, ecav = env.reset(10000)
#     den = env.p_max - env.p_min
#     obs = np.concatenate((ecav[:, len(env.mu)//2-150:len(env.mu)//2+150]/10,
#                           env.power*np.ones((env.seq_len, 1))/den,
#                           np.zeros((env.seq_len, 1))), axis=1)
#     r_hist = []
#     action_hist = []
#     score = 0
#     done = False
#     pcav_hist = []
#     pbar = tqdm(total=env.max_steps-env.init_steps_, ncols=120, position=run_id, desc=f'Run {run_id}')
#     idx = 0
#     delta_theta = []
#     det_hist = []
#     terminal = False  # ensure defined
#     while not done:
#         action = agent.choose_action(obs, True)

#         next_state, reward, done, terminal, _, acav_, ecav_, _ = env.step(state, action, desired_spectrum_tensor)
#         state = next_state
#         ecav = ecav_
#         ecav_obs = np.concatenate((ecav_[-1, len(env.mu)//2-150:len(env.mu)//2+150]/10,
#                                    env.power/den,
#                                    5*env.current_del_omega/(env.del_omega_init - env.del_omega_end)), axis=0)
#         obs = np.concatenate((obs[1:], ecav_obs[np.newaxis, :]), axis=0)
#         score += reward
#         curr_pcav = np.sum(np.abs(acav_)**2)
#         pcav_hist.append(curr_pcav)
#         r_hist.append(reward)
#         action_hist.append(action)
#         delta_theta.append(env.delta_theta.item())
#         det_hist.append(env.current_del_omega.item()/(2*np.pi))

#         idx += env.ctrl_freq
#         pbar.update(env.ctrl_freq)
#     pbar.close()

#     # Only send back non-None results
#     if terminal==False:
#         pcav_arr = np.array(pcav_hist)
#         freq_arr = np.array(det_hist) + np.array(delta_theta)  # your "freq"
#         result_queue.put((run_id, pcav_arr, freq_arr))
# %%
# # write a function to load the reward history and plot it
# def plot_reward_histories_sigma(files, N=100, S=0, label='Reward', color='C0'):
#     """
#     Plot rolling mean and std of rewards from multiple runs, handling different lengths.
#     Optionally leave the last S samples from max_len and then plot.

#     Args:
#         files (list): List of file paths to .npy reward histories.
#         N (int): Window size for rolling mean/std.
#         S (int): Number of samples to leave from the end.
#         label (str): Label for the mean line.
#         color (str): Color for the plot.
#     """
#     import numpy as np
#     import matplotlib.pyplot as plt
#     import matplotlib.ticker as ticker

#     plt.style.use('physrev.mplstyle')

#     # Load all reward histories
#     rewards = [np.load(f) for f in files]
#     max_len = max(len(r) for r in rewards)
#     # Pad with np.nan to align lengths
#     rewards_padded = np.full((len(rewards), max_len), np.nan)
#     for i, r in enumerate(rewards):
#         rewards_padded[i, :len(r)] = r

#     # Optionally leave the last S samples
#     plot_len = max_len - S if S > 0 else max_len

#     # Compute rolling mean and std, ignoring nan
#     rolling_mean = []
#     rolling_std = []
#     for t in range(plot_len):
#         window = rewards_padded[:, max(0, t-N+1):t+1]
#         vals = window[~np.isnan(window)]
#         if len(vals) > 0:
#             rolling_mean.append(np.mean(vals))
#             rolling_std.append(np.std(vals))
#         else:
#             rolling_mean.append(np.nan)
#             rolling_std.append(np.nan)

#     steps = np.linspace(0, 100*plot_len, plot_len)*1e-5
#     mu = np.array(rolling_mean)
#     sigma = np.array(rolling_std)

#     plt.figure(figsize=(7, 5))
#     plt.plot(steps, mu, color=color, linewidth=1.5)
#     plt.fill_between(steps, mu - sigma, mu + sigma, color=color, alpha=0.3)
#     plt.xlabel(r'Steps $(\times 10^5)$', fontsize=14)
#     plt.ylabel('Reward', fontsize=16)
#     plt.xticks(fontsize=16)
#     plt.yticks(fontsize=16)
#     # set x-ticks to exponent format
#     # formatter = ticker.ScalarFormatter(useMathText=True)
#     # formatter.set_scientific(True)
#     # formatter.set_powerlimits((-1, 1))
#     # plt.gca().xaxis.set_major_formatter(formatter)
#     # plt.title('Reward Rolling Mean ± Std', fontsize=16, fontweight='bold')
#     # plt.legend(fontsize=14)
#     plt.grid()
#     plt.tight_layout()
#     plt.savefig(os.path.join(save_dir, 'reward_histories_sigma.png'))
#     plt.show()
# # %%
# def plot_reward_histories_min_max(files, N=100, S=0, label='Reward', color='C0'):
#     """
#     Plot rolling mean and min/max of rewards from multiple runs, handling different lengths.
#     Optionally leave the last S samples from max_len and then plot.

#     Args:
#         files (list): List of file paths to .npy reward histories.
#         N (int): Window size for rolling stats.
#         S (int): Number of samples to leave from the end.
#         label (str): Label for the mean line.
#         color (str): Color for the plot.
#     """
#     import numpy as np
#     import matplotlib.pyplot as plt

#     plt.style.use('physrev.mplstyle')

#     # Load all reward histories
#     rewards = [np.load(f) for f in files]
#     max_len = max(len(r) for r in rewards)
#     # Pad with np.nan to align lengths
#     rewards_padded = np.full((len(rewards), max_len), np.nan)
#     for i, r in enumerate(rewards):
#         rewards_padded[i, :len(r)] = r

#     # Optionally leave the last S samples
#     plot_len = max_len - S if S > 0 else max_len

#     # Compute rolling mean, min, and max, ignoring nan
#     rolling_mean = []
#     rolling_min = []
#     rolling_max = []
#     for t in range(plot_len):
#         window = rewards_padded[:, max(0, t-N+1):t+1]
#         vals = window[~np.isnan(window)]
#         if len(vals) > 0:
#             rolling_mean.append(np.mean(vals))
#             rolling_min.append(np.min(vals))
#             rolling_max.append(np.max(vals))
#         else:
#             rolling_mean.append(np.nan)
#             rolling_min.append(np.nan)
#             rolling_max.append(np.nan)

#     steps = np.linspace(0, 100*plot_len, plot_len)*1e-5
#     mu = np.array(rolling_mean)
#     minv = np.array(rolling_min)
#     maxv = np.array(rolling_max)

#     plt.figure(figsize=(7, 5))
#     plt.plot(steps, mu, color=color, linewidth=1.5)
#     plt.fill_between(steps, minv, maxv, color=color, alpha=0.3)
#     plt.xlabel(r'Steps $(\times 10^5)$', fontsize=14)
#     plt.ylabel('Reward', fontsize=16)
#     plt.xticks(fontsize=16)
#     plt.yticks(fontsize=16)
#     plt.grid()
#     plt.tight_layout()
#     # plt.legend(fontsize=14)
#     plt.savefig(os.path.join(save_dir, 'reward_histories_min_max.png'))
#     plt.show()
# %%
def plot_pcav_freq_mean_std(pcav_files, freq_files, save_dir):
    """
    Plot mean and std of P_cav and Frequency from multiple npy files.

    Args:
        pcav_files (list): List of file paths to .npy files for P_cav.
        freq_files (list): List of file paths to .npy files for Frequency.
        save_dir (str): Directory to save the plots.
    """
    import matplotlib.pyplot as plt

    plt.style.use('physrev.mplstyle')

    # Use dark blue and red colors
    pcav_color = '#1a237e'  # dark blue
    freq_color = '#b71c1c'  # dark red

    # Load and pad arrays to same length
    pcav_arrs = [np.load(f) for f in pcav_files]
    freq_arrs = [np.load(f) for f in freq_files]
    max_len = max(max(len(a) for a in pcav_arrs), max(len(a) for a in freq_arrs))
    pcav_pad = np.full((len(pcav_arrs), max_len), np.nan)
    freq_pad = np.full((len(freq_arrs), max_len), np.nan)
    for i, arr in enumerate(pcav_arrs):
        pcav_pad[i, :len(arr)] = arr
    for i, arr in enumerate(freq_arrs):
        freq_pad[i, :len(arr)] = arr

    norm_freq_pad = 2*freq_pad*env.tR.item() / env.alpha  # Convert rad/s to GHz
    mu_pcav = np.nanmean(pcav_pad, axis=0)
    sd_pcav = np.nanstd(pcav_pad, axis=0)
    mu_freq = np.nanmean(freq_pad, axis=0)
    sd_freq = np.nanstd(freq_pad, axis=0)
    mu_norm_freq = np.nanmean(norm_freq_pad, axis=0)
    sd_norm_freq = np.nanstd(norm_freq_pad, axis=0)

    x = np.arange(max_len)

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ln1, = ax1.plot(x, mu_pcav, color=pcav_color, linewidth=1.8, label='P_cav mean')
    ax1.fill_between(x, mu_pcav - sd_pcav, mu_pcav + sd_pcav, color=pcav_color, alpha=0.25, label='P_cav ±1σ')
    ax1.set_xlabel('Tuning Steps', fontsize=16)
    ax1.set_ylabel(r'$P_{cav}$ (W)', fontsize=16, color=pcav_color)
    ax1.tick_params(axis='y', labelcolor=pcav_color)
    ax1.grid(True, alpha=0.4)

    ax2 = ax1.twinx()
    ln2, = ax2.plot(x, mu_freq, color=freq_color, linewidth=1.8, label='Freq mean')
    ax2.fill_between(x, mu_freq - sd_freq, mu_freq + sd_freq, color=freq_color, alpha=0.25, label='Freq ±1σ')
    ax2.set_ylabel('Frequency (GHz)', fontsize=16, color=freq_color)
    ax2.tick_params(axis='y', labelcolor=freq_color)

    # ax1.legend([ln1, ln2], ['P_cav mean', 'Freq mean'], loc='upper right', fontsize=12)

    fig.tight_layout()
    ax1.tick_params(axis='x', labelsize=14)
    ax1.tick_params(axis='y', labelsize=14)
    ax2.tick_params(axis='y', labelsize=14)
    fig.savefig(os.path.join(save_dir, 'pcav_freq_mean_std.png'), dpi=200)
    fig.savefig(os.path.join(save_dir, 'pcav_freq_mean_std.svg'))
    plt.show()
    plt.close(fig)

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ln1, = ax1.plot(x, mu_pcav, color=pcav_color, linewidth=1.8, label='P_cav mean')
    ax1.fill_between(x, mu_pcav - sd_pcav, mu_pcav + sd_pcav, color=pcav_color, alpha=0.25, label='P_cav ±1σ')
    ax1.set_xlabel('Tuning Steps', fontsize=16)
    ax1.set_ylabel(r'$P_{cav}$ (W)', fontsize=16, color=pcav_color)
    ax1.tick_params(axis='y', labelcolor=pcav_color)
    ax1.grid(True, alpha=0.4)

    ax2 = ax1.twinx()
    ln2, = ax2.plot(x, mu_norm_freq, color=freq_color, linewidth=1.8, label='Freq mean')
    ax2.fill_between(x, mu_norm_freq - sd_norm_freq, mu_norm_freq + sd_norm_freq, color=freq_color, alpha=0.25, label='Freq ±1σ')
    ax2.set_ylabel(r'$\frac{2(\delta _{\Theta} + \delta _0)}{(\alpha + \kappa)}$', fontsize=16, color=freq_color)
    ax2.tick_params(axis='y', labelcolor=freq_color)

    # ax1.legend([ln1, ln2], ['P_cav mean', 'Freq mean'], loc='upper right', fontsize=12)

    fig.tight_layout()
    ax1.tick_params(axis='x', labelsize=14)
    ax1.tick_params(axis='y', labelsize=14)
    ax2.tick_params(axis='y', labelsize=14)
    fig.savefig(os.path.join(save_dir, 'pcav_norm_freq_mean_std.png'), dpi=200)
    fig.savefig(os.path.join(save_dir, 'pcav_norm_freq_mean_std.svg'))
    plt.show()
    plt.close(fig)
    print('Saved pcav_freq_mean_std.*')
# %%
import torch.multiprocessing as mp
import os
import glob
import numpy as np

if __name__ == '__main__':
    # Create save directory if not exists
    save_dir = os.path.join('./results', agent.run_name, 'new3')
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
    # npy_files = glob.glob(os.path.join(save_dir, '*.npy'))
    # # # Example usage: plot all reward histories with a rolling window of 100
    # plot_reward_histories_sigma(npy_files, N=5, S=0, label='Reward', color='C0')
    # plot_reward_histories_min_max(npy_files, N=5, S=0, label='Reward', color='C0')
    # pcav_files = sorted(glob.glob(os.path.join(save_dir, '*_p_cav.npy')))
    # freq_files = sorted(glob.glob(os.path.join(save_dir, '*_detuning_theta_sum.npy')))
    # plot_pcav_freq_mean_std(pcav_files, freq_files, save_dir)
# '''
# %%
# if __name__ == '__main__':
#     import torch.multiprocessing as mp
#     import os
#     import glob
#     import numpy as np
#     from queue import Empty

#     save_dir = os.path.join('./results', agent.run_name)
#     os.makedirs(save_dir, exist_ok=True)
#     print('Save dir:', save_dir)
#     mp.set_start_method('spawn', force=True)  # safer for PyTorch

#     num_runs = 10
#     result_queue = mp.Queue()
#     processes = []
#     for run_id in range(num_runs):
#         p = mp.Process(target=run_test_processes, args=(run_id, save_dir, result_queue))
#         p.start()
#         processes.append(p)
#     for p in processes:
#         p.join()
#     # Ensure all processes have completed
#     print('All processes completed.')

#     # Collect results
#     pcav_all = []
#     freq_all = []
#     while True:
#         try:
#             rid, pcav, freq = result_queue.get_nowait()
#         except Empty:  # <-- fix exception type
#             break
#         if pcav is not None and freq is not None:
#             pcav_all.append(pcav)
#             freq_all.append(freq)

#     # Clean up the queue
#     result_queue.close()
#     result_queue.join_thread()
#     # Save aggregated results (ragged lists are stored as object arrays)
#     np.savez_compressed(
#         os.path.join(save_dir, 'aggregated_results.npz'),
#         pcav_hist=np.array(pcav_all),
#         freq=np.array(freq_all),
#     )
#     print(f'Collected {len(pcav_all)}/{num_runs} successful runs. Saved aggregated_results.npz')

#     # Existing reward plots (if you still want to run them)
#     npy_files = glob.glob(os.path.join(save_dir, '*.npy'))
# %%
# import glob
# save_dir = os.path.join('./results', agent.run_name, 'new')
# pcav_files = sorted(glob.glob(os.path.join(save_dir, '*_p_cav.npy')))
# freq_files = sorted(glob.glob(os.path.join(save_dir, '*_detuning_theta_sum.npy')))
# plot_pcav_freq_mean_std(pcav_files, freq_files, save_dir)

# %%
