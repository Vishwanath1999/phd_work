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
                 delta_omega_min=-1e6, delta_omega_max=1e6, delta_omega_step=1e4, soft_clamp=False, softness=0.5):
        

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

        self.soft_clamp_ = soft_clamp
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
        del_omega_init = 14*self.un_norm_kappa
        self.del_omega_init = del_omega_init
        self.current_del_omega = del_omega_init
        # del_omega_end = self.sim_tensor['domega_end']
        del_omega_end = -14*self.un_norm_kappa
        self.del_omega_end = del_omega_end

        # del_omega_stop = self.sim_tensor['domega_stop']
        del_omega_stop = -14*self.un_norm_kappa
        self.del_omega_stop = self.del_omega_end-2*self.un_norm_kappa
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
        self.primary_sidebands = torch.tensor(self.primary_sidebands, device=DEVICE, dtype=torch.float64)
        # self.pcav_ref = loadmat('Pcomb_rl_allv2.mat')['Pcomb'].T
        self.seq_len = seq_len
        self.p_max = p_max
        self.p_min = p_min
        self.ctrl_freq = ctrl_freq
        self.delta_omega_min = delta_omega_min  
        self.delta_omega_max = delta_omega_max
        self.delta_omega_step = delta_omega_step
        self.softness = softness

    
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
            curr_pcav = np.sum(np.abs(Acav_np)**2)
            
            if idx % self.ctrl_freq == 0:
                self.ecav_state.append(Ecav_dBm.cpu().numpy())
                self.pcav_hist.append(curr_pcav)
                if len(self.ecav_state) > self.seq_len:
                    self.ecav_state.pop(0)
                    self.pcav_hist.pop(0)

        self.primary_sidebands_flag = False
        self.ecav_state = np.array(self.ecav_state)

        self.env_p_hist = []

        self.det_out_cntr = 0
        
        print('Reset...')
        return self.state, Acav_np, self.ecav_state, np.array(self.pcav_hist)

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
        

        if self.step_cntr-self.init_steps_ >= int(0.5*self.Nt) and corr < 0.25: #and self.step_cntr-self.init_steps_ <= self.Nt:
            terminal = True
            reward_penalty = -5
            print('Did not form soliton ...')
            print('Spectral Corr:', corr)
        
        return terminal, reward_penalty
    
    @staticmethod
    def soft_clamp(x: torch.Tensor,
                low: torch.Tensor,
                high: torch.Tensor,
                softness: float = 0.5) -> torch.Tensor:
        """
        Smooth alternative to torch.clamp.
        softness -> 0  →  hard clamp
        softness → 1   →  almost linear
        """
        # centre & scale
        mid   = (high + low) / 2
        span  = (high - low) / 2          # >0
        z     = (x - mid) / (span * softness)

        # ⟨−∞,∞⟩ → ⟨−1,1⟩  (tanh)  then back to physical units
        return mid + span * torch.tanh(z)
    
    @staticmethod
    def spectral_width_dbm_torch(y_dbm: torch.Tensor, threshold: float = -60, center_idx: int = None, norm: int = 300) -> torch.Tensor:
        """
        Counts how many points in y_dbm (PyTorch tensor) are above threshold (dBm), excluding the central mode,
        and divides by norm (default 300).
        """
        if center_idx is None:
            center_idx = y_dbm.shape[0] // 2
        mask = (y_dbm > threshold)
        mask[center_idx] = False  # exclude central mode
        width = torch.sum(mask.float()) / norm
        return width.item()
    
    def step(self, state, action, desired_spectrum):
        '''
        The step function takes the current state and an action as input, and returns the next state, reward, 
        done flag, terminal flag, achieved flag, Acav_np, ecav_state, and Ewg.
        Parameters:
            state (torch.Tensor) : Current state
            action (np.ndarray) : Action to be taken
            desired_spectrum (torch.Tensor) : Desired spectrum
        Returns:
            next_state (torch.Tensor) : Next state
            reward (float) : Reward obtained
            done (bool) : Whether the episode is done
            terminal (bool) : Whether the episode is terminated
            achieved (bool) : Whether the desired spectrum is achieved
            Acav_np (np.ndarray) : Cavity field in numpy array
            ecav_state (np.ndarray) : Electric field in cavity state
            Ewg (np.ndarray) : Electric field in waveguide
        '''
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

            if self.soft_clamp_ == False:
                self.current_del_omega = torch.clamp(del_omega, min=min(self.del_omega_end, self.del_omega_init), max=max(self.del_omega_end, self.del_omega_init))
            else:
                clamped_ = self.soft_clamp(del_omega*1e-9, low=min(self.del_omega_end, self.del_omega_init)*1e-9, high=max(self.del_omega_end, self.del_omega_init)*1e-9, softness=self.softness)
                self.current_del_omega = clamped_*1e9

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
        if len(self.pcav_hist) > env.seq_len:
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

        if torch.corrcoef(torch.stack([self.primary_sidebands, Ecav_dBm]))[0,1].item() > 0.7:
            self.primary_sidebands_flag = True
        
        # if self.primary_sidebands_flag == False:
        #     reward = curr_pcav
        # else:
        reward = 4*torch.corrcoef(torch.stack([desired_spectrum_dBm, Ecav_dBm]))[0,1].item() + self.spectral_width_dbm_torch(Ecav_dBm, norm=150) # - 0.3*curr_pcav#+ 
        # self.primary_sidebands_flag = torch.corrcoef(torch.stack([desired_spectrum_dBm, Ecav_dBm]))[0,1]
        # penalize for high variance in power
        # if len(self.env_p_hist) > 1:
        #     power_var = np.std(self.env_p_hist)
        #     if power_var > 0.01:
        #         reward -= 2*len(self.env_p_hist) * (power_var - 0.01)
        
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
            
        
        return self.next_state, reward, done, terminal, achieved, Acav_np, self.ecav_state, Ewg.cpu().numpy()

# %%
# torch seed
# torch.manual_seed(0)
env = RL_MRR_Env(seq_len=100, p_max=0.2, p_min=0.05, ctrl_freq=100, thermal_effect='high',\
                  delta_omega_min=-5e6, delta_omega_max=5e6, delta_omega_step=5e4, soft_clamp=False, softness=0.5)
fpmp = env.sim_tensor['f_pmp'].item()
freq = (fpmp + np.arange(-220,221)*env.FSR.item())*1e-12
# %%
desired_spectrum = loadmat('desired_spec2.mat')['Ewg'][0]
desired_spectrum_dBm = 10*np.log10(np.abs(desired_spectrum)**2)+30 
desired_spectrum_dBm = np.clip(desired_spectrum_dBm, -60, None)
desired_spectrum_tensor = torch.tensor(desired_spectrum, device=DEVICE, dtype=torch.complex128)
# %%
config = {
    'input_dim': [env.seq_len, 300+3],
    'n_actions': 2,
    'alpha': 3e-4,
    'beta': 3e-4,
    'mem_size': int(1e6),
    'run_name': 'mrr_sac_cluster_delayed_toptica_pow_ton_un_norm_high',
    'batch_size': 256,
    'dist': 'beta', # 'beta' or 'normal'
    'train':True,
    'p_max': env.p_max,
    'p_min': env.p_min,
    'fc_dim':256,
    'use_per':True,
    'delta_omega_min': env.delta_omega_min,  # Minimum detuning in Hz
    'delta_omega_max': env.delta_omega_max,   # Maximum detuning in Hz
    'delta_omega_step': env.delta_omega_step,   # Step size for detuning in
    'bidirectional': False,  # Whether to use bidirectional detuning
    'env_soft_clamp': env.soft_clamp_,  # Whether to use soft clamping in the environment
    'softness': env.softness,  # Softness parameter for soft clamping
    }
# %%

from rl_codes.sac_v3 import SACAgent
agent = SACAgent(input_dim=config['input_dim'], n_actions=config['n_actions'], alpha=config['alpha'], beta=config['beta'],
                mem_size=config['mem_size'], batch_size=config['batch_size'], dist=config['dist'], run_name=config['run_name'],
                eval_mode=not(torch.cuda.is_available()), fc_dim=config['fc_dim'], use_per=config['use_per'], bidir=config['bidirectional'])
print(agent.actor)
print(agent.critic_1)

# %%
# # init wandb run
if config['train']:
    wandb.init(project='maddpg_mrr', entity='viswacolab-technical-university-of-denmark', config=config)
    # wandb.watch(agent.actor, log='gradients', log_freq=10)
    # set the wandb run name
    wandb.run.name = agent.run_name
# %% MADDPG train loop
# '''
if config['train']:
    logs={}
    n_games = 1000
    # r_hist = []
    # scaled_r_hist = []
    global_n_steps = 0
    scores = []
    best_score = -np.inf
    den = env.p_max - env.p_min
    acav_hist = []
    ecav_hist = []
    for i in range(n_games):
        score = 0
        done = False
        n_steps = 0
        state, acav, ecav, pcav = env.reset(10000)
        logs['pump power'] = env.power
        log_pcav = 10*np.log10(pcav + 1e-12) + 30
        obs = np.concatenate((ecav[:,len(env.mu)//2-150:len(env.mu)//2+150]/10,env.power*np.ones((env.seq_len,1))/den,np.zeros((env.seq_len,1)), log_pcav[:,np.newaxis]),axis=1)
        pbar = tqdm(total=env.max_steps-env.init_steps_, ncols=120, position=i, desc='Episode %d' % i)
        acav_hist = []
        ecav_hist = []
        det_hist = []
        to_chaos_hist = []
        while not done:
            action = agent.choose_action(obs)
            
            next_state, reward, done, terminal, achieved, acav_, ecav_, ewg = env.step(state, action, desired_spectrum_tensor)
            # log perf action
            logs['power (W)'] = env.power
            logs['detuning (MHz)'] = env.rescale_and_quantize(action[1], env.delta_omega_min, env.delta_omega_max, env.delta_omega_step)*1e-6
            logs['reward'] = reward  
            curr_pcav = np.sum(np.abs(acav_)**2,keepdims=True)
            
            ecav_obs = np.concatenate((ecav_[-1,len(env.mu)//2-150:len(env.mu)//2+150]/10, env.power/den, 5*env.current_del_omega/(env.del_omega_init - env.del_omega_end), 10*np.log10(curr_pcav)+30), axis=0)
            obs_ = np.concatenate((obs[1:], ecav_obs[np.newaxis,:]), axis=0)
            obs = obs_   
            agent.remember(obs, action, reward, obs_, terminal)
            state = next_state
            ecav = ecav_
            score += reward
            n_steps += 1
            pbar.update(env.ctrl_freq)

            ewg_dBm = 10*np.log10(np.abs(ewg)**2)+30
            ewg_dBm = np.clip(ewg_dBm, -60, 20)

            acav_hist.append(np.abs(acav_))
            ecav_hist.append(ewg_dBm)
            det_hist.append(env.current_del_omega.item()/(2*np.pi*1e9))  # Convert rad/s to GHz
            to_chaos_hist.append((env.delta_theta.item()*env.FSR.item())/(2*np.pi*1e9))  # Convert rad/s to GHz
            
            if agent.memory.mem_cntr > 4*agent.batch_size:
                cl, al, ent_loss, ent_coeff = agent.learn(global_n_steps)
                logs['critic_loss'] = cl
                logs['actor_loss'] = al
                logs['entropy_loss'] = ent_loss
                logs['entropy_coeff'] = ent_coeff
                # print('Critic loss:', cl, 'Actor loss:', al, 'Entropy loss:', ent_loss, 'Entropy coeff:', ent_coeff)

            if config['use_per']:
                logs['beta_per'] = agent.beta
            # if global_n_steps%100 == 0:
            wandb.log(logs)

            global_n_steps += 1
        scores.append(score)
        avg_score = np.mean(scores[-20:])
        pbar.close()

        if terminal==False :
                
            fig=plt.figure(figsize=(14,4))
            plt.vlines(np.arange(-220,221, 1), -60*np.ones(len(ecav[-1])), ewg_dBm, \
                    label='Obtained Spectrum', linewidth=1.5)
            plt.vlines(np.arange(-220,221, 1), -60*np.ones(len(desired_spectrum)),\
                        desired_spectrum_dBm, color='red', label='Desired Spectrum',alpha=0.5,linewidth=1.5)
            
            plt.xlabel('Rel. Mode no.', fontsize=16)
            plt.ylabel('Power(dBm)', fontsize=16)
            plt.grid()
            plt.ylim(-90,25)
            plt.xlim(-180,180)
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            plt.legend(fontsize=16, loc='lower center')
            plt.title('Correlation '+str(np.round(np.corrcoef(ewg_dBm, np.clip(desired_spectrum_dBm,-60,20))[0,1],2)), fontsize=14)
            plt.tight_layout()
            wandb.log({"ecav_modes": wandb.Image(fig)})
            plt.close(fig)

            fig=plt.figure(figsize=(14,4))
            plt.vlines(freq, -60*np.ones(len(ecav[-1])), ewg_dBm, \
                    label='Obtained Spectrum', linewidth=1.5)
            plt.vlines(freq, -60*np.ones(len(desired_spectrum)),\
                        desired_spectrum_dBm, color='red', label='Desired Spectrum',alpha=0.5,linewidth=1.5)
            
            plt.xlabel('Freq (THz).', fontsize=16)
            plt.ylabel('Power(dBm)', fontsize=16)
            plt.grid()
            plt.ylim(-90,25)
            plt.xlim(freq[220-180],freq[220+180])
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            plt.legend(fontsize=16, loc='lower center')
            plt.title('Correlation '+str(np.round(np.corrcoef(ewg_dBm, np.clip(desired_spectrum_dBm,-60,20))[0,1],2)), fontsize=14)
            plt.tight_layout()
            wandb.log({"ecav_freq": wandb.Image(fig)})
            plt.close(fig)
        
        if avg_score > best_score or terminal==False:
            fig = plt.figure(figsize=(14,4))
            plt.imshow(np.array(acav_hist).T, aspect='auto', cmap='jet', origin='lower', extent=[0, len(acav_hist), -env.tR.item()/2*1e12,env.tR.item()/2*1e12])
            plt.colorbar(label='Power (a.u.)')
            plt.xlabel('Time step', fontsize=16)
            plt.ylabel(r'$t_R$ (ps)', fontsize=16)
            plt.title('Cavity Mode Power Evolution', fontsize=16)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            plt.tight_layout()
            wandb.log({"acav_modes": wandb.Image(fig)})
            plt.close(fig)

            fig = plt.figure(figsize=(14,4))
            plt.imshow(np.array(ecav_hist).T, aspect='auto', cmap='jet', origin='lower', extent=[0, len(ecav_hist), -220,220])
            plt.colorbar(label='Power (dBm)')
            plt.xlabel('Time step', fontsize=16)
            plt.ylabel(r'$\mu$', fontsize=16)
            plt.title('Cavity Spectrum Evolution', fontsize=16)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            plt.tight_layout()
            wandb.log({"ecav_modes": wandb.Image(fig)})
            plt.close(fig)

            fig = plt.figure(figsize=(10,6))
            plt.plot(det_hist, label=r'$f_{det}$', color='blue', linewidth=1.5)
            plt.plot(to_chaos_hist, label=r'$f_{\theta}$', color='orange', linewidth=1.5)
            plt.xlabel('Time step', fontsize=16)
            plt.ylabel('Freq (GHz)', fontsize=16)
            plt.title('Detuning and Chaos Frequency Evolution', fontsize=16)
            plt.grid()
            plt.legend(fontsize=16)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            plt.tight_layout()
            wandb.log({"detuning_chaos_freq": wandb.Image(fig)})
            plt.close(fig)

        if terminal == False and avg_score > best_score:
            agent.save_models()

        if avg_score > best_score:
            best_score = avg_score        

        print('episode: ', i, 'score: %.2f' % score, 'average score: %.2f' % avg_score,'best score: %.2f' % best_score, 'n_steps:', n_steps, 'terminal:', terminal)
# '''
# %%
import matplotlib.pyplot as plt
plt.style.use('phyrev.mplstyle')
# Plot episode scores
if config['train']:
    plt.figure(figsize=(10, 6))
    plt.plot(np.array(scores)/(env.max_steps), color='blue', linewidth=1.5)
    plt.xlabel('Episode no.', fontsize=16)
    plt.ylabel('Score', fontsize=16)
    plt.title('Episode Scores Normalized by length', fontsize=16)
    plt.grid(which='both', linestyle='--', linewidth=0.5)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.savefig(agent.run_name+'_scores.png')
    # save svg
    plt.savefig(agent.run_name+'_scores.svg', format='svg')
    plt.show()
    # wandb.log({"episode_scores": wandb.Histogram(scores)})
# %%
# state, acav, ecav = env.reset(10000)
# den = env.p_max - env.p_min
# obs = np.concatenate((ecav/10,env.power*np.ones((env.seq_len,1))/den),axis=1)
# print('Chosen power:', env.power)
# r_hist = []
# action_hist = []
# acav_hist = []
# score = 0
# done = False
# pcav_hist = []
# pbar = tqdm(total=env.max_steps-env.init_steps_, ncols=120)
# idx = 0
# done = False
# ecav_hist = []
# achieved = False
# while not done:
# # for idx in tqdm(range(env.init_steps_, int(env.max_steps)), ncols=120):
#     # perform random actions
#     # try:
#         # action = agent.choose_action(obs, True)
#         # action = np.random.uniform(-1,1,size=(1,))
#         # action = np.random.normal(0,1,size=(1,))
#         if achieved==True:
#             action = np.array([0])
#         else:
#             action = np.array([1])#np.random.choice([0, 1, 2], p=[1/3, 1/3, 1/3])

#         next_state, reward, done, terminal, achieved, acav_, ecav_ = env.step(state, action[0], desired_spectrum_tensor)
#         state = next_state
#         ecav = ecav_
#         obs_ = np.concatenate((ecav_/10,env.power*np.ones((env.seq_len,1))/den),axis=1)
#         obs = obs_
#         score += reward
#         curr_pcav = np.sum(np.abs(acav_))
#         pcav_hist.append(curr_pcav)
#         r_hist.append(reward)
#         action_hist.append(action)
       
#         acav_hist.append(acav_)
#         idx += env.ctrl_freq
#         pbar.update(env.ctrl_freq)
# pbar.close()

# print('Test score %.2f' % score)
# %%
import numpy as np
import matplotlib.pyplot as plt


def soft_clamp(x, low, high, softness=0.05):
    """
    Smooth alternative to np.clip.
    softness → 0  →  hard clamp
    softness → 1  →  almost linear
    
    Parameters
    ----------
    x         : array-like – input values
    low/high  : float      – clamp interval
    softness  : float      – fraction of the span that is “soft”
    """
    mid  = (high + low) / 2.0
    span = (high - low) / 2.0               # positive
    z    = (x - mid) / (span * softness)    # scale to ±1/softness
    return mid + span * np.tanh(z)


# -------------------------------------------------
# Quick visual sanity-check
# -------------------------------------------------
x_max = 7*env.un_norm_kappa.item()/(2*np.pi*1e9)
x_min = -1*env.un_norm_kappa.item()/(2*np.pi*1e9)
print('Detuning range for soft clamp test:', x_min, ' to ', x_max, ' GHz')
x = np.linspace(x_min, x_max, 10000)         # test inputs
x = x[::-1]
low, high = x_min, x_max

hard   = np.clip(x, low, high)
soft05 = soft_clamp(x, low, high, softness=0.05)
soft10 = soft_clamp(x, low, high, softness=0.10)
soft20 = soft_clamp(x, low, high, softness=0.5)

plt.figure(figsize=(10, 6))
plt.plot(x, hard,   label='Hard clamp',   color='black', linewidth=2)
plt.plot(x, soft05, label='Soft clamp (0.05)')
plt.plot(x, soft10, label='Soft clamp (0.10)')
plt.plot(x, soft20, label='Soft clamp (0.50)')
plt.axhline(low,  color='gray', ls='--')
plt.axhline(high, color='gray', ls='--')
plt.title('Hard clamp vs. Soft clamp')
plt.xlabel('Input value')
plt.ylabel('Clamped output')
plt.legend()
plt.grid(True)
plt.show()

# %%
a = np.random.normal(0,1,size=(3900,))
b = env.rescale_and_quantize(a, env.delta_omega_min, env.delta_omega_max, env.delta_omega_step)
# insert env.del_omega_start as the first element of b
b = np.insert(b, 0, env.del_omega_init.item()/(2*np.pi))
b = np.cumsum(b)*1e-9
# clamped_b = soft_clamp(b, low=min(env.del_omega_end.item()/(2*np.pi), env.del_omega_init.item()/(2*np.pi))*1e-9, high=max(env.del_omega_end.item()/(2*np.pi), env.del_omega_init.item()/(2*np.pi))*1e-9, softness=0.5)
# plt.figure(figsize=(10, 6))
# plt.plot(clamped_b, label='Soft clamp (0.5)', color='blue')
# plt.plot(np.clip(b, min(env.del_omega_end.item()/(2*np.pi), env.del_omega_init.item()/(2*np.pi))*1e-9, max(env.del_omega_end.item()/(2*np.pi), env.del_omega_init.item()/(2*np.pi))*1e-9), label='Hard clamp', color='black', linestyle='--')
# plt.axhline(env.del_omega_init.item()*1e-9/(2*np.pi),  color='gray', ls='--')
# plt.axhline(env.del_omega_end.item()*1e-9/(2*np.pi), color='gray', ls='--')
# plt.title('Detuning Range: '+str(np.round(env.del_omega_init.item()/(2*np.pi*1e9),2))+' to '+str(np.round(env.del_omega_end.item()/(2*np.pi*1e9),2))+' GHz')
# plt.xlabel('Cumulative detuning (GHz)')
# plt.ylabel('Clamped detuning (GHz)')
# plt.legend()
# plt.grid(True)
# plt.show()
# %%
