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
        del_omega_init = 7*self.un_norm_kappa
        self.del_omega_init = del_omega_init
        self.del_omega_ul = 7*self.un_norm_kappa
        self.current_del_omega = del_omega_init
        del_omega_end = -7*self.un_norm_kappa
        self.del_omega_end = del_omega_end

        # del_omega_stop = self.sim_tensor['domega_stop']
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
        self.max_steps = int(7e5)
        t_end  = self.max_steps*tR.cpu().numpy()
        t_ramp = t_end
        tr = tR.cpu().numpy()
        self.Nt = self.max_steps#np.round(t_ramp/tr/dt).astype(int)

        self.del_omega_0 = del_omega_init + (1/self.Nt)*(del_omega_end - del_omega_init)


        self.t_sim_step = 0

        self.primary_sidebands = loadmat('primary_sidebands.mat')['spec'][0]
        self.primary_sidebands = torch.tensor(self.primary_sidebands, device=DEVICE, dtype=torch.float64)
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
        self.t_sim_step = 0

        # self.power = np.random.uniform(self.p_min, self.p_max, size=(1,))
        self.power = np.array([0.16])
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

            Fdrive_val = self.Fdrive(del_omega + (self.delta_theta/self.tR), self.t_sim_step, self.Ain)
            u0 = self.ssfm_step(self.state, self.step_cntr, self.alpha, self.Dint_shift, del_omega + (self.delta_theta/self.tR), self.tR, self.gamma, \
                                self.L, 10, 1e-3, 1, self.kext, Fdrive_val)
            self.step_cntr += 1
            self.state = u0
            P_avg = torch.mean(torch.abs(u0)**2)  # Compute average power
            d_delta_theta_dt = -self.delta_theta / self.tau0 + self.xi * P_avg
            self.delta_theta += (1 * self.tR) * d_delta_theta_dt  # Euler step
            self.t_sim_step += self.tR
             

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

        # self.env_p_hist = []
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
        if torch.isnan(width):
            print('Width is NaN')
            return 0.0
        return width.item()

    # @staticmethod
    # def spectral_symm_torch(y_dbm: torch.Tensor, threshold: float = -58) -> float:
    #     """
    #     Computes the Pearson correlation coefficient between the left and right thresholded spectrum parts,
    #     excluding the central mode, using torch.corrcoef.
    #     """
    #     center_idx = y_dbm.shape[0] // 2
    #     mask = (y_dbm > threshold)
    #     if torch.sum(mask).item() <= 1:
    #         return 0.0
    #     left_modes = mask[:center_idx].float()*y_dbm[:center_idx]
    #     right_modes = mask[center_idx+1:].float()*y_dbm[center_idx+1:]
    #     right_modes_flipped = torch.flip(right_modes, dims=[0])
    #     min_len = min(left_modes.shape[0], right_modes_flipped.shape[0])
    #     if min_len < 2:
    #         return 0.0
    #     left_modes = left_modes[-min_len:]
    #     right_modes_flipped = right_modes_flipped[:min_len]
    #     # Stack into 2xN to compute correlation matrix
    #     stacked = torch.stack([left_modes, right_modes_flipped], dim=0)
    #     corr_mat = torch.corrcoef(stacked)
    #     corr = corr_mat[0, 1]
    #     # check if corr is nan
    #     if torch.isnan(corr):
    #         print('Symm is NaN')
    #         return 0.0
    #     return corr.item()
    @staticmethod
    def spectral_symm_torch(spectrum: torch.Tensor, floor: float=-55) -> float:
        n = spectrum.size(0)
        center_idx = n // 2

        left = spectrum[:center_idx]
        right = spectrum[center_idx+1:]

        left_filtered = left[left > floor]
        right_filtered = right[right > floor]

        min_len = min(len(left_filtered), len(right_filtered))
        if min_len <2:
            return 0.0  # Return zero if no correlation

        left_trim = left_filtered[:min_len]
        right_trim = right_filtered[:min_len]
        right_trim = torch.flip(right_trim, dims=[0])

        stacked = torch.stack((left_trim, right_trim), dim=0)
        corr_matrix = torch.corrcoef(stacked)

        return corr_matrix[0, 1].item()

    # @staticmethod
    # def spectral_smoothness_torch(y_dbm: torch.Tensor, floor: float = -60) -> float:
    #     """
    #     Computes the mean squared difference between adjacent spectral bins,
    #     excluding the central mode and bins below the floor threshold.
    #     """
    #     mask = torch.ones_like(y_dbm, dtype=torch.float)
    #     mask[y_dbm.shape[0] // 2] = 0  # exclude central mode
    #     y_dbm_masked = y_dbm * mask
    #     y_diff = torch.diff(y_dbm_masked)
    #     smoothness = torch.mean(y_diff ** 2) / 10.0
    #     return smoothness.item()
    
    @staticmethod
    def spectral_corr_torch(y_dbm: torch.Tensor, y_target_dbm: torch.Tensor) -> float:
        """
        Computes the Pearson correlation coefficient between y_dbm and y_target_dbm using torch.corrcoef.
        """
        stacked = torch.stack([y_dbm, y_target_dbm])
        corr_mat = torch.corrcoef(stacked)
        corr = corr_mat[0, 1]
        if torch.isnan(corr):
            print('Corr is NaN')
            return 0.0
        return corr.item()
    
    @staticmethod
    def any_above_threshold_excluding_center(tensor, threshold):
        n = tensor.size(0)
        center_idx = n // 2
        # Exclude central value
        mask = torch.ones(n, dtype=bool)
        mask[center_idx] = False
        tensor_exc_center = tensor[mask]
        return torch.any(tensor_exc_center > threshold).item()

    @staticmethod
    def calculate_proximity_bonus(ecav_hist: np.ndarray, desired_spectrum_dbm: np.ndarray) -> float:
        """
        Calculates a time-weighted bonus for maintaining a high correlation with the desired spectrum.
        This version uses np.corrcoef in a loop.

        Args:
            ecav_hist (np.ndarray): History of spectra, shape (100, 300).
            desired_spectrum_dbm (np.ndarray): The target spectrum, shape (300,).

        Returns:
            float: The calculated reward bonus.
        """
        seq_len = ecav_hist.shape[0]
        if seq_len < 10:  # Ensure there's enough history to be meaningful
            return 0.0

        # Create weights that increase linearly, giving more importance to recent spectra.
        weights = np.linspace(0.4, 1.0, seq_len)
        
        correlations = np.zeros(seq_len)
        for i in range(seq_len):
            # Calculate correlation for each historical spectrum against the target
            corr_matrix = np.corrcoef(ecav_hist[i], desired_spectrum_dbm)
            # The correlation value is at [0, 1] (or [1, 0])
            correlations[i] = corr_matrix[0, 1]

        # Handle any potential NaN values that might arise from zero-variance slices
        correlations = np.nan_to_num(correlations, nan=0.0)

        # Calculate the weighted average of the correlations.
        weighted_avg_corr = np.average(correlations, weights=weights) # Square weights to emphasize recent history more strongly.

        # Shape the bonus: reward high correlation, penalize low correlation.
        # This creates a strong signal to stay above a certain performance threshold.
        # The bonus is positive if the weighted average is > 0.7, and negative otherwise.
        maintenance_bonus = (weighted_avg_corr - 0.5)*1.2
        
        return maintenance_bonus

    
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
        # env.power = self.rescale_power(action, lower_limit=self.p_min, upper_limit=self.p_max, step_size=0.001)
        # self.env_p_hist.append(env.power[0])
        # if len(self.env_p_hist) > env.seq_len:
        #     self.env_p_hist.pop(0)
        
        Ppmp = torch.tensor(self.power, dtype=torch.float64)
        
        for ii in range(1):
            self.Ein[ii,int(self.mu0+self.ind_pmp[ii])] = torch.sqrt(Ppmp[ii])*len(self.mu)
            self.Ain[ii] = torch.fft.ifft(torch.fft.fftshift(self.Ein[ii],dim=0),dim=0)*torch.exp(-1j*self.phi_pmp[ii])
        
        delta_omega = (2*torch.pi)*self.rescale_and_quantize(action, self.delta_omega_min, self.delta_omega_max, self.delta_omega_step)  # Convert GHz to rad/s
        
        for _ in range(self.ctrl_freq):
            del_omega = self.current_del_omega + delta_omega 

            if self.soft_clamp_ == False:
                self.current_del_omega = torch.clamp(del_omega, min=min(self.del_omega_end, self.del_omega_ul), max=max(self.del_omega_end, self.del_omega_ul))
            else:
                clamped_ = self.soft_clamp(del_omega*1e-9, low=min(self.del_omega_end, self.del_omega_init)*1e-9, high=max(self.del_omega_end, self.del_omega_init)*1e-9, softness=self.softness)
                self.current_del_omega = clamped_*1e9

            Fdrive_val = self.Fdrive(self.current_del_omega + self.delta_theta/(self.tR), self.t_sim_step, self.Ain)
            u0 = self.ssfm_step(state, self.step_cntr, self.alpha, self.Dint_shift, self.current_del_omega + self.delta_theta/(self.tR), self.tR, self.gamma, \
                                self.L, 10, 1e-3, 1, self.kext, Fdrive_val)
            state = u0
            P_avg = torch.mean(torch.abs(u0)**2)  # Compute average power
            d_delta_theta_dt = -self.delta_theta / self.tau0 + self.xi * P_avg
            self.delta_theta += (1 * self.tR) * d_delta_theta_dt  # Euler step
            self.step_cntr += 1
            self.t_sim_step += self.tR
            
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
        if self.ecav_state.shape[0] >= self.seq_len:
            self.ecav_state = np.delete(self.ecav_state, 0, axis=0)
        self.ecav_state = np.concatenate((self.ecav_state, Ecav_dBm.cpu().numpy()[np.newaxis,:]), axis=0)

        if torch.corrcoef(torch.stack([self.primary_sidebands, Ecav_dBm]))[0,1].item() > 0.7:
            self.primary_sidebands_flag = True
        
        # if self.any_above_threshold_excluding_center(Ecav_dBm, threshold=-56) == False:
        #     reward = 1*np.mean(self.pcav_hist)
        # else:
        reward = 2*env.spectral_width_dbm_torch(Ecav_dBm) + 2*(env.spectral_corr_torch(Ecav_dBm, desired_spectrum_dBm)-0.5) + 2*(env.spectral_symm_torch(Ecav_dBm))

        
        # if self.current_del_omega + self.delta_theta/(self.tR)
        # reward = 4*torch.corrcoef(torch.stack([desired_spectrum_dBm, Ecav_dBm]))[0,1].item() + 2*self.spectral_width_dbm_torch(Ecav_dBm)
        # 4*torch.corrcoef(torch.stack([desired_spectrum_dBm, Ecav_dBm]))[0,1].item() + 
        
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
# a fuction that takes in env as argument and calculates the manitude of difference between curr_del_omega and del_omega_init and del_omega_end
def calc_detuning_distance(env, scale=1):
    # distance to initial detuning
    span = env.del_omega_ul - env.del_omega_end
    d_low = (env.current_del_omega - env.del_omega_end)/span
    d_high = (env.del_omega_ul - env.current_del_omega)/span
    return np.array([scale*d_low.item(), scale*d_high.item()])
# %%
# torch seed
# torch.manual_seed(0)
env = RL_MRR_Env(seq_len=100, p_max=0.2, p_min=0.05, ctrl_freq=100, thermal_effect='high',\
                  delta_omega_min=-2e6, delta_omega_max=2e6, delta_omega_step=1e4, soft_clamp=False, softness=0.35)
fpmp = env.sim_tensor['f_pmp'].item()
freq = (fpmp + np.arange(-220,221)*env.FSR.item())*1e-12
# %%
desired_spectrum = loadmat('desired_spec2.mat')['Ewg'][0]
desired_spectrum_dBm = 10*np.log10(np.abs(desired_spectrum)**2)+30 
desired_spectrum_dBm = np.clip(desired_spectrum_dBm, -60, None)
desired_spectrum_tensor = torch.tensor(desired_spectrum, device=DEVICE, dtype=torch.complex128)
# %%
config = {
    'input_dim': [env.seq_len, 300+2+2+1],
    'n_actions': 1,
    'alpha': 3e-4,
    'beta': 3e-4,
    'mem_size': int(1e6),
    'run_name': 'mrr_sac_cluster_delayed_toptica_pow_ton_un_norm_high_only_detuningv4',
    'batch_size': 256,
    'dist': 'normal', # 'beta' or 'normal'
    'train':False,
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
    'alpha_per': 0.4,  # Initial value of alpha for PER
    'beta_per': int(2e5),   # Number of steps to reach beta=1 for PER
    }
# %%

from rl_codes.sac_v3 import SACAgent
agent = SACAgent(input_dim=config['input_dim'], n_actions=config['n_actions'], alpha=config['alpha'], beta=config['beta'],
                mem_size=config['mem_size'], batch_size=config['batch_size'], dist=config['dist'], run_name=config['run_name'],
                eval_mode=not(torch.cuda.is_available()), fc_dim=config['fc_dim'], use_per=config['use_per'], bidir=config['bidirectional'],
                beta_steps=config['beta_per'], alpha_per=config['alpha_per'])
print(agent.actor)
print(agent.critic_1)
agent.load_models()
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
    plt.xlabel(r'Time ($\mu$s)', fontsize=20)
    plt.ylabel(r'Power (mW)', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.title('Intracavity Power', fontsize=22, fontweight='bold')
    plt.tight_layout()    
    plt.savefig(os.path.join(save_dir, mod_pow + f'_{thermal_effect}_run_{idx}_pcav_spec_all_ctrl.png'))
    plt.savefig(os.path.join(save_dir, mod_pow + f'_{thermal_effect}_run_{idx}_pcav_spec_all_ctrl.svg'), format='svg')
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(time_axis, 1e3*np.array(pcav_hist), linewidth=1.5)
    plt.grid()
    plt.xlabel(r'Time ($\mu$s)', fontsize=20)
    plt.ylabel(r'Power (mW)', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlim(0, 0.025)  # Zoom in to first 25 microseconds
    plt.tight_layout()    
    plt.savefig(os.path.join(save_dir, mod_pow + f'_{thermal_effect}_run_{idx}_pcav_zoom.png'))
    plt.savefig(os.path.join(save_dir, mod_pow + f'_{thermal_effect}_run_{idx}_pcav_zoom.svg'), format='svg')
    plt.close()

    # 2. Acav time evolution
    plt.figure(figsize=(14,4))
    plt.imshow(1e3*np.abs(np.array(acav_hist).T)**2, aspect='auto', cmap='jet',
               extent=[time_axis[0], time_axis[-1], -1e12*env.tR.item()/2, 1e12*env.tR.item()/2])
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=20)
    cbar.set_label(r'Power $(mW)$', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-1, 1))
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.xlabel(r'Time ($\mu$s)', fontsize=20)
    plt.ylabel(r'Fast time $(t_R)$', fontsize=20)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, mod_pow + f'_{thermal_effect}_run_{idx}_ecav_hist_spec_all_ctrl.png'))
    plt.savefig(os.path.join(save_dir, mod_pow + f'_{thermal_effect}_run_{idx}_ecav_hist_spec_all_ctrl.svg'), format='svg')
    plt.close()

    # 3. Spectrum evolution (FFT of acav_hist)
    plt.figure(figsize=(14,4))
    spectrum = np.fft.fftshift(np.fft.fft(np.array(acav_hist).T, axis=0), axes=0)
    spectrum_dBm = 10*np.log10(np.abs(spectrum)**2)+30
    spectrum_dBm = np.clip(spectrum_dBm, -60, 10)
    plt.imshow(spectrum_dBm, aspect='auto', cmap='jet',
               extent=[time_axis[0], time_axis[-1], env.mu.min().item(), env.mu.max().item()])
    plt.xlabel(r'Time ($\mu$s)', fontsize=20)
    # plt.ylabel(r'$\mu$' +'(rel)', fontsize=20)
    plt.ylabel('Mode number', fontsize=20)
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=20)
    cbar.set_label(r'Power $(dBm)$', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, mod_pow + f'_{thermal_effect}_run_{idx}_ecav_hist_ifft_spec_all_ctrl.png'))
    plt.savefig(os.path.join(save_dir, mod_pow + f'_{thermal_effect}_run_{idx}_ecav_hist_ifft_spec_all_ctrl.svg'), format='svg')
    plt.close()

    # 4. Ewg spectrum evolution
    plt.figure(figsize=(14,4))
    e_wg_hist = np.array(e_wg_hist).T
    ewg_dBm = 10*np.log10(np.abs(e_wg_hist)**2)+30
    ewg_dBm = np.clip(ewg_dBm, -60, 50)
    plt.imshow(ewg_dBm, aspect='auto', cmap='jet',
               extent=[time_axis[0], time_axis[-1], env.mu.min().item(), env.mu.max().item()])
    plt.xlabel(r'Time ($\mu$s)', fontsize=20)
    # plt.ylabel(r'$\mu$' +'(rel)', fontsize=20)
    plt.ylabel('Mode number', fontsize=20)
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=20)
    cbar.set_label(r'Power $(dBm)$', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, mod_pow + f'_{thermal_effect}_run_{idx}_ewg_hist_spec_all_ctrl.png'))
    plt.savefig(os.path.join(save_dir, mod_pow + f'_{thermal_effect}_run_{idx}_ewg_hist_spec_all_ctrl.svg'), format='svg')
    plt.close()

    # 5. Reward plot
    plt.figure(figsize=(10, 6))
    plt.plot(time_axis, r_hist)
    plt.xlabel(r'Time ($\mu$s)', fontsize=20)
    plt.ylabel('Reward ', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, mod_pow + f'_{thermal_effect}_run_{idx}_rewards_spec_all_ctrl.png'))
    plt.savefig(os.path.join(save_dir, mod_pow + f'_{thermal_effect}_run_{idx}_rewards_spec_all_ctrl.svg'), format='svg')
    plt.close()

    # 8. Detuning array (GHz)
    detuning_array = np.array(det_hist)
    plt.figure(figsize=(10, 6))
    plt.plot(time_axis, detuning_array*1e-9, linewidth=1.5, label='Pump detuning')
    plt.plot(time_axis, np.array(delta_theta)*1e-9, linewidth=1.5 , label=r'$f_{\Theta}$')
    plt.xlabel(r'Time ($\mu$s)', fontsize=20)
    plt.ylabel('Freq (GHz)', fontsize=20)
    plt.grid()
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(fontsize=20)
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-1, 1))
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, mod_pow + f'_{thermal_effect}_run_{idx}_actions_spec_all_ctrl.png'))
    plt.savefig(os.path.join(save_dir, mod_pow + f'_{thermal_effect}_run_{idx}_actions_spec_all_ctrl.svg'), format='svg')
    plt.close()

    # 9. Detuning in kappa units
    kappa = env.un_norm_kappa.item()/(2*np.pi)
    detuning_kappa = detuning_array / kappa
    plt.figure(figsize=(10, 6))
    plt.plot(time_axis, detuning_kappa, linewidth=1.5, label=r'$\Delta f_{pmp}$')
    if env.thermal_effect != 'none':
        plt.plot(time_axis, np.array(delta_theta)/kappa, linewidth=1.5 , label=r'$f_{\Theta}$')
        plt.plot(time_axis, detuning_kappa+np.array(delta_theta)/kappa, linewidth=1.5 , label=r'$\Delta f_{eff}$')
    plt.xlabel(r'Time ($\mu$s)', fontsize=20)
    plt.ylabel(r'Frequency', fontsize=20)
    plt.grid()
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(fontsize=20)

    ymin, ymax = plt.ylim()
    yticks = np.linspace(ymin, ymax, 5)
    def frac_label(val):
        if val == 0:
            return r"$0$"
        elif val == int(val):
            return rf"${int(val)}\,\kappa$"
        else:
            return rf"${val:.1f}\,\kappa$"
    ytick_labels = [frac_label(y) for y in yticks]
    plt.yticks(yticks, ytick_labels, fontsize=20)
    plt.title('Pump Detuning', fontsize=22, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, mod_pow + f'_{thermal_effect}_run_{idx}_actions_spec_all_ctrl_kappa.png'))
    plt.savefig(os.path.join(save_dir, mod_pow + f'_{thermal_effect}_run_{idx}_actions_spec_all_ctrl_kappa.svg'), format='svg')
    plt.close()

    # Detuning in kappa units (zoomed)
    plt.figure(figsize=(10, 6))
    plt.plot(time_axis, detuning_kappa, linewidth=1.5, label=r'$\Delta f_{pmp}$')
    if env.thermal_effect != 'none':
        plt.plot(time_axis, np.array(delta_theta)/kappa, linewidth=1.5 , label=r'$f_{\Theta}$')
        plt.plot(time_axis, detuning_kappa+np.array(delta_theta)/kappa, linewidth=1.5 , label=r'$\Delta f_{eff}$')
    plt.xlabel(r'Time ($\mu$s)', fontsize=20)
    plt.ylabel(r'Frequency', fontsize=20)
    plt.grid()
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    # plt.legend(fontsize=20)
    plt.xlim(0, 0.025)  # Zoom in to first 25 microseconds 
    ymin, ymax = plt.ylim()
    yticks = np.linspace(np.floor(ymin*2)/2, np.ceil(ymax*2)/2 + 0.01, num=8)
    ytick_labels = [frac_label(y) for y in yticks]
    plt.yticks(yticks, ytick_labels, fontsize=20)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, mod_pow + f'_{thermal_effect}_run_{idx}_actions_spec_all_ctrl_kappa_zoom.png'))
    plt.savefig(os.path.join(save_dir, mod_pow + f'_{thermal_effect}_run_{idx}_actions_spec_all_ctrl_kappa_zoom.svg'), format='svg')
    plt.close()

    # # 10. Pump power
    # plt.figure(figsize=(10, 6))
    # plt.plot(time_axis, 1000*env.rescale_power(action_hist[:,0], env.p_max, env.p_min), linewidth=1.5)
    # plt.xlabel(r'Time ($\mu$s)', fontsize=20)
    # plt.ylabel('Pump Power (mW)', fontsize=20)
    # plt.grid()
    # plt.xticks(fontsize=20)
    # plt.yticks(fontsize=20)
    # plt.ylim(25, 1000*env.p_max)
    # plt.tight_layout()
    # plt.savefig(os.path.join(save_dir, mod_pow + f'_{thermal_effect}_run_{idx}_actions_power_spec_all_ctrl.png'))
    # plt.savefig(os.path.join(save_dir, mod_pow + f'_{thermal_effect}_run_{idx}_actions_power_spec_all_ctrl.svg'), format='svg')
    # plt.close()

    # 12. Detuning vs delta theta
    plt.figure(figsize=(10, 6))
    plt.plot(detuning_array*1e-9, np.array(delta_theta)*1e-9, linewidth=1.5)
    plt.xlabel(r'$\Delta f_{pmp}$ (GHz)', fontsize=20)
    plt.ylabel(r'$f _{\Theta}$ (GHz)', fontsize=20)
    plt.grid()
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, mod_pow + f'_{thermal_effect}_run_{idx}_detuning_delta_theta_spec_all_ctrl.png'))
    plt.savefig(os.path.join(save_dir, mod_pow + f'_{thermal_effect}_run_{idx}_detuning_delta_theta_spec_all_ctrl.svg'), format='svg')
    plt.close()

    # 6. Obtained vs desired spectrum (mode index)
    plt.figure(figsize=(14,4))
    plt.vlines(np.arange(-220,221, 1), -60*np.ones(len(ecav[-1])), obtained_spectrum,
               label='Obtained Spectrum',alpha=1, linewidth=1.5)
    plt.vlines(np.arange(-220,221, 1), -60*np.ones(len(desired_spectrum_dBm)),
               desired_spectrum_dBm, color='red', label='Desired Spectrum',alpha=0.5, linewidth=1.5)
    plt.xlabel('Rel. Mode no.', fontsize=20)
    plt.ylabel('Power(dBm)', fontsize=20)
    plt.grid()
    plt.ylim(-90,30)
    plt.xlim(-150, 150)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(fontsize=20,loc='lower center')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, mod_pow + f'_{thermal_effect}_run_{idx}_ecav_spec_all_ctrl_modes.png'))
    plt.savefig(os.path.join(save_dir, mod_pow + f'_{thermal_effect}_run_{idx}_ecav_spec_all_ctrl_modes.svg'), format='svg')
    plt.close()

    # 7. Obtained vs desired spectrum (frequency)
    plt.figure(figsize=(14,4))
    plt.vlines(freq, -60*np.ones(len(ecav[-1])), obtained_spectrum,
               label='Obtained Spectrum',alpha=1, linewidth=1.5)
    plt.vlines(freq, -60*np.ones(len(desired_spectrum_dBm)),
               desired_spectrum_dBm, color='red', label='Desired Spectrum',alpha=0.5, linewidth=1.5)
    plt.xlabel('Freq. (THz)', fontsize=20)
    plt.ylabel('Power(dBm)', fontsize=20)
    plt.grid()
    plt.ylim(-90,30)
    plt.xlim(freq[220-150], freq[220+150])
    plt.xticks(fontsize=20, fontweight='bold')
    plt.yticks(fontsize=20, fontweight='bold')
    plt.legend(fontsize=20, loc='lower center')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, mod_pow + f'_{thermal_effect}_run_{idx}_ecav_spec_all_ctrl_freq.png'))
    plt.savefig(os.path.join(save_dir, mod_pow + f'_{thermal_effect}_run_{idx}_ecav_spec_all_ctrl_freq.svg'), format='svg')
    plt.close()


    f_det = np.array(detuning_array) * 1e-9  # rad/s to GHz
    f_theta = np.array(delta_theta) * 1e-9   # rad to GHz
    p_cav = 1e3 * np.array(pcav_hist)  # W to mW

    fig = plt.figure(figsize=(14, 14))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(f_det, f_theta, p_cav, c=p_cav, cmap='jet', s=15)
    ax.plot(f_det, f_theta, p_cav, color='blue', alpha=0.7)
    ax.set_xlabel(r'$\Delta f_{pmp}$ (GHz)', fontsize=20)
    ax.set_ylabel(r'$f_{\Theta}$ (GHz)', fontsize=20)
    ax.set_zlabel(r'$P_{cav}$ (mW)', fontsize=20, labelpad=-0.7, rotation=90)
    ax.tick_params(axis='both', labelsize=20)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, mod_pow + f'_{thermal_effect}_run_{idx}_3d_detuning_delta_theta_pcav.png'))
    plt.savefig(os.path.join(save_dir, mod_pow + f'_{thermal_effect}_run_{idx}_3d_detuning_delta_theta_pcav.svg'), format='svg')
    plt.close()

    print('All plots saved successfully in', save_dir)
# %%
from scipy.signal import spectrogram, find_peaks, welch
from scipy.stats import entropy

# Advanced Soliton Analysis Functions for Micro Ring Resonator RL
# Integration code for your plot_all_results function

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram, find_peaks, welch
from scipy.stats import entropy
from mpl_toolkits.mplot3d import Axes3D
import os

def phase_portrait_analysis(acav_hist, det_hist, delta_theta, save_dir, idx, thermal_effect, env):
    """
    Phase Portrait Analysis for Soliton Dynamics
    
    Creates 4 different phase portraits to reveal:
    - Periodic vs chaotic attractors in power evolution
    - 3D phase space reconstruction showing complex dynamics
    - Detuning-thermal frequency relationships
    - Power derivative phase portraits
    """
    acav_array = np.array(acav_hist)
    power_envelope = np.sum(np.abs(acav_array)**2, axis=1)
    
    tau = 50  # delay in steps
    if len(power_envelope) > tau:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Power envelope phase portrait
        x = power_envelope[:-tau]
        y = power_envelope[tau:]
        scatter = axes[0,0].scatter(x*1e3, y*1e3, c=np.arange(len(x)), cmap='viridis', s=1, alpha=0.7)
        axes[0,0].set_xlabel('P(t) (mW)', fontsize=12)
        axes[0,0].set_ylabel(f'P(t+{tau}Δt) (mW)', fontsize=12)
        axes[0,0].set_title('Power Envelope Phase Portrait', fontsize=12)
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. Detuning vs thermal frequency phase portrait
        det_array = np.array(det_hist)*1e-9  # Convert to GHz
        theta_array = np.array(delta_theta)*1e-9/(2*np.pi*env.tR.item())  # Convert to GHz
        
        if len(det_array) > tau:
            scatter2 = axes[0,1].scatter(det_array[:-tau], theta_array[:-tau], 
                            c=np.arange(len(det_array[:-tau])), cmap='plasma', s=2, alpha=0.8)
            axes[0,1].set_xlabel('Pump Detuning (GHz)', fontsize=12)
            axes[0,1].set_ylabel(r'$f_\Theta$ (GHz)', fontsize=12)
            axes[0,1].set_title('Detuning-Thermal Phase Portrait', fontsize=12)
            axes[0,1].grid(True, alpha=0.3)
        
        # 3. Power derivative phase portrait
        power_deriv = np.diff(power_envelope)
        if len(power_deriv) > tau:
            x_deriv = power_deriv[:-tau]
            y_deriv = power_deriv[tau:]
            scatter3 = axes[1,0].scatter(x_deriv*1e6, y_deriv*1e6, c=np.arange(len(x_deriv)), 
                            cmap='coolwarm', s=1, alpha=0.7)
            axes[1,0].set_xlabel('dP/dt (μW/step)', fontsize=12)
            axes[1,0].set_ylabel(f'd²P/dt² (μW/step)', fontsize=12)
            axes[1,0].set_title('Power Velocity Phase Portrait', fontsize=12)
            axes[1,0].grid(True, alpha=0.3)
        
        # 4. 3D phase portrait
        if len(power_envelope) > 2*tau:
            axes[1,1].remove()
            ax_3d = fig.add_subplot(2, 2, 4, projection='3d')
            x_3d = power_envelope[:-2*tau]
            y_3d = power_envelope[tau:-tau]
            z_3d = power_envelope[2*tau:]
            
            scatter4 = ax_3d.scatter(x_3d*1e3, y_3d*1e3, z_3d*1e3, 
                                  c=np.arange(len(x_3d)), cmap='viridis', s=1, alpha=0.7)
            ax_3d.set_xlabel('P(t) (mW)', fontsize=10)
            ax_3d.set_ylabel(f'P(t+{tau}Δt) (mW)', fontsize=10)
            ax_3d.set_zlabel(f'P(t+{2*tau}Δt) (mW)', fontsize=10)
            ax_3d.set_title('3D Phase Portrait', fontsize=12)
        
        mod_pow = str(env.power[0]).replace('.','_')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{mod_pow}_{thermal_effect}_run_{idx}_phase_portraits.png'), dpi=200, bbox_inches='tight')
        plt.savefig(os.path.join(save_dir, f'{mod_pow}_{thermal_effect}_run_{idx}_phase_portraits.svg'), bbox_inches='tight')
        plt.show()


def time_frequency_spectrogram(acav_hist, save_dir, idx, thermal_effect, env):
    """
    Time-Frequency Spectrogram Analysis
    
    Analyzes:
    - Central modes spectral evolution during soliton formation
    - Mode participation and effective mode number
    - Spectral entropy as complexity measure
    - Rolling power statistics
    """
    acav_array = np.array(acav_hist)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Central modes power evolution
    center_idx = acav_array.shape[1] // 2
    central_modes = acav_array[:, center_idx-20:center_idx+21]  # 41 central modes
    power_central = np.abs(central_modes)**2
    
    im1 = axes[0,0].imshow(power_central.T, aspect='auto', cmap='jet', origin='lower',
                          extent=[0, len(power_central), -20, 20])
    axes[0,0].set_ylabel('Mode Index (rel to center)', fontsize=12)
    axes[0,0].set_xlabel('Time Steps', fontsize=12)
    axes[0,0].set_title('Central Modes Power Evolution', fontsize=12)
    plt.colorbar(im1, ax=axes[0,0], label='Power (a.u.)')
    
    # 2. Total power with rolling statistics
    total_power = np.sum(np.abs(acav_array)**2, axis=1)
    window_size = 100
    
    if len(total_power) > window_size:
        rolling_mean = np.convolve(total_power, np.ones(window_size)/window_size, mode='valid')
        rolling_std = np.array([np.std(total_power[i:i+window_size]) 
                               for i in range(len(total_power)-window_size+1)])
        
        time_axis = np.arange(len(rolling_mean))
        axes[0,1].plot(time_axis, rolling_mean*1e3, 'b-', linewidth=1.5, label='Mean Power')
        axes[0,1].fill_between(time_axis, (rolling_mean-rolling_std)*1e3, 
                              (rolling_mean+rolling_std)*1e3, alpha=0.3, color='blue', label='±1σ')
        axes[0,1].set_ylabel('Power (mW)', fontsize=12)
        axes[0,1].set_xlabel('Time Steps', fontsize=12)
        axes[0,1].set_title('Rolling Power Statistics', fontsize=12)
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
    
    # 3. Mode participation (effective number of modes)
    mode_power = np.abs(acav_array)**2
    mode_occupation = mode_power / (np.sum(mode_power, axis=1, keepdims=True) + 1e-12)
    eff_modes = 1.0 / np.sum(mode_occupation**2, axis=1)
    
    axes[1,0].plot(eff_modes, 'r-', linewidth=1.5)
    axes[1,0].set_ylabel('Effective Mode Number', fontsize=12)
    axes[1,0].set_xlabel('Time Steps', fontsize=12)
    axes[1,0].set_title('Mode Participation Evolution', fontsize=12)
    axes[1,0].grid(True, alpha=0.3)
    
    # 4. Spectral entropy evolution
    spectral_entropy = np.array([entropy(mode_occupation[i] + 1e-12) for i in range(len(mode_occupation))])
    axes[1,1].plot(spectral_entropy, 'g-', linewidth=1.5)
    axes[1,1].set_ylabel('Spectral Entropy', fontsize=12)
    axes[1,1].set_xlabel('Time Steps', fontsize=12)
    axes[1,1].set_title('Spectral Entropy Evolution', fontsize=12)
    axes[1,1].grid(True, alpha=0.3)
    
    mod_pow = str(env.power[0]).replace('.','_')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{mod_pow}_{thermal_effect}_run_{idx}_time_frequency.png'), dpi=200, bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, f'{mod_pow}_{thermal_effect}_run_{idx}_time_frequency.svg'), bbox_inches='tight')
    plt.show()


def chaos_metrics_analysis(acav_hist, det_hist, delta_theta, r_hist, save_dir, idx, thermal_effect, env):
    """
    Chaos Metrics Analysis
    
    Provides:
    - Chaos indicators based on power fluctuations
    - Power spectral density analysis
    - Reward landscape mapping
    - Autocorrelation functions
    - Phase space reconstruction
    """
    acav_array = np.array(acav_hist)
    power_envelope = np.sum(np.abs(acav_array)**2, axis=1)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # 1. Chaos indicator (power fluctuations)
    window_size = 200
    chaos_indicators = []
    window_centers = []
    
    for start in range(0, len(power_envelope) - window_size, 50):
        end = start + window_size
        window_data = power_envelope[start:end]
        
        # Use coefficient of variation as chaos indicator
        chaos_ind = np.std(window_data) / (np.mean(window_data) + 1e-12)
        chaos_indicators.append(chaos_ind)
        window_centers.append(start + window_size // 2)
    
    axes[0,0].plot(window_centers, chaos_indicators, 'b-', linewidth=1.5)
    axes[0,0].axhline(y=0.1, color='r', linestyle='--', alpha=0.7, label='Chaos threshold')
    axes[0,0].set_ylabel('Chaos Indicator', fontsize=12)
    axes[0,0].set_xlabel('Time Steps', fontsize=12)
    axes[0,0].set_title('Power Fluctuation Analysis', fontsize=12)
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # 2. Power spectral density
    if len(power_envelope) > 64:
        freqs, psd = welch(power_envelope, fs=1.0, nperseg=min(256, len(power_envelope)//4))
        axes[0,1].loglog(freqs[1:], psd[1:])  # Skip DC component
        axes[0,1].set_ylabel('PSD', fontsize=12)
        axes[0,1].set_xlabel('Frequency (1/steps)', fontsize=12)
        axes[0,1].set_title('Power Spectral Density', fontsize=12)
        axes[0,1].grid(True, alpha=0.3)
    
    # 3. Reward landscape analysis
    rewards = np.array(r_hist)
    det_array = np.array(det_hist)
    theta_array = np.array(delta_theta)
    
    if len(rewards) > 100:
        # Create 2D histogram of reward vs parameters
        H, xedges, yedges = np.histogram2d(det_array*1e-9, theta_array*1e-9, 
                                          bins=20, weights=rewards)
        H_counts, _, _ = np.histogram2d(det_array*1e-9, theta_array*1e-9, bins=20)
        H_avg = np.divide(H, H_counts, out=np.zeros_like(H), where=H_counts!=0)
        
        im = axes[0,2].imshow(H_avg.T, origin='lower', aspect='auto', cmap='RdYlBu_r',
                              extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
        axes[0,2].set_xlabel('Pump Detuning (GHz)', fontsize=12)
        axes[0,2].set_ylabel(r'Thermal Freq (GHz)', fontsize=12)
        axes[0,2].set_title('Average Reward Landscape', fontsize=12)
        plt.colorbar(im, ax=axes[0,2], label='Average Reward')
    
    # 4. Temporal correlation function
    def autocorrelation(x, max_lag=100):
        x = x - np.mean(x)
        autocorr = np.correlate(x, x, mode='full')
        autocorr = autocorr[autocorr.size // 2:]
        autocorr = autocorr / autocorr[0]  # Normalize
        return autocorr[:max_lag]
    
    lags = np.arange(100)
    autocorr = autocorrelation(power_envelope)
    axes[1,0].plot(lags, autocorr, 'purple', linewidth=1.5)
    axes[1,0].set_ylabel('Autocorrelation', fontsize=12)
    axes[1,0].set_xlabel('Time Lag (steps)', fontsize=12)
    axes[1,0].set_title('Power Autocorrelation', fontsize=12)
    axes[1,0].grid(True, alpha=0.3)
    
    # 5. Phase space reconstruction (2D)
    tau_embed = 20
    if len(power_envelope) > 2*tau_embed:
        x_embed = power_envelope[:-tau_embed]
        y_embed = power_envelope[tau_embed:]
        
        # Color by time
        scatter = axes[1,1].scatter(x_embed*1e3, y_embed*1e3, 
                                   c=np.arange(len(x_embed)), cmap='plasma', s=1, alpha=0.6)
        axes[1,1].set_xlabel('P(t) (mW)', fontsize=12)
        axes[1,1].set_ylabel(f'P(t+{tau_embed}) (mW)', fontsize=12)
        axes[1,1].set_title('Phase Space Reconstruction', fontsize=12)
        axes[1,1].grid(True, alpha=0.3)
    
    # 6. Power vs detuning evolution
    det_sub = det_array[::10] * 1e-9  # Subsample and convert to GHz
    power_sub = power_envelope[::10] * 1e3  # Subsample and convert to mW
    
    scatter = axes[1,2].scatter(det_sub, power_sub, c=np.arange(len(det_sub)), 
                               cmap='viridis', s=20, alpha=0.6)
    axes[1,2].set_xlabel('Pump Detuning (GHz)', fontsize=12)
    axes[1,2].set_ylabel('Power (mW)', fontsize=12)
    axes[1,2].set_title('Power vs Detuning Evolution', fontsize=12)
    axes[1,2].grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=axes[1,2], label='Time Step')
    
    mod_pow = str(env.power[0]).replace('.','_')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{mod_pow}_{thermal_effect}_run_{idx}_chaos_metrics.png'), dpi=200, bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, f'{mod_pow}_{thermal_effect}_run_{idx}_chaos_metrics.svg'), bbox_inches='tight')
    plt.show()


def soliton_birth_death_analysis(acav_hist, save_dir, idx, thermal_effect, env):
    """
    Soliton Birth/Death Analysis
    
    Tracks:
    - Soliton population dynamics over time
    - Individual soliton trajectories
    - Collision/interaction events
    - Birth/death rate statistics
    - Power distribution analysis
    """
    acav_array = np.array(acav_hist)
    
    # Detect solitons by finding localized peaks in the field profile
    def detect_solitons(field_snapshot, threshold_factor=0.3):
        power = np.abs(field_snapshot)**2
        threshold = threshold_factor * np.max(power)
        
        peaks, properties = find_peaks(power, height=threshold, width=3)
        
        return peaks, power[peaks] if len(peaks) > 0 else np.array([])
    
    # Track soliton count over time
    soliton_counts = []
    soliton_positions = []
    soliton_powers = []
    
    for i, field in enumerate(acav_array):
        peaks, heights = detect_solitons(field)
        soliton_counts.append(len(peaks))
        soliton_positions.append(peaks)
        soliton_powers.append(heights)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Soliton count evolution
    axes[0,0].plot(soliton_counts, 'b-', linewidth=1.5)
    axes[0,0].set_ylabel('Number of Solitons', fontsize=12)
    axes[0,0].set_xlabel('Time Steps', fontsize=12)
    axes[0,0].set_title('Soliton Population Dynamics', fontsize=12)
    axes[0,0].grid(True, alpha=0.3)
    
    # 2. Soliton trajectory map
    max_solitons = max(soliton_counts) if soliton_counts else 0
    if max_solitons > 0:
        trajectory_map = np.full((len(acav_array), len(acav_array[0])), np.nan)
        
        for t, (positions, powers) in enumerate(zip(soliton_positions, soliton_powers)):
            for pos, power in zip(positions, powers):
                if pos < trajectory_map.shape[1]:
                    trajectory_map[t, pos] = power*1e3
        
        im = axes[0,1].imshow(trajectory_map.T, aspect='auto', cmap='hot', origin='lower',
                             extent=[0, len(acav_array), 0, len(acav_array[0])])
        axes[0,1].set_xlabel('Time Steps', fontsize=12)
        axes[0,1].set_ylabel('Position Index', fontsize=12)
        axes[0,1].set_title('Soliton Trajectory Map', fontsize=12)
        plt.colorbar(im, ax=axes[0,1], label='Soliton Power (mW)')
    
    # 3. Collision/interaction events detection
    collision_events = []
    for t in range(1, len(soliton_positions)):
        prev_pos = set(soliton_positions[t-1])
        curr_pos = set(soliton_positions[t])
        
        # Simple collision detection: significant change in soliton number
        if abs(len(prev_pos) - len(curr_pos)) > 0:
            collision_events.append(t)
        elif len(prev_pos) > 1 and len(curr_pos) > 1:
            # Check if solitons are closer than before
            if len(prev_pos) > 1:
                min_dist_prev = min([abs(p1-p2) for p1 in prev_pos for p2 in prev_pos if p1 != p2])
            else:
                min_dist_prev = float('inf')
                
            if len(curr_pos) > 1:
                min_dist_curr = min([abs(p1-p2) for p1 in curr_pos for p2 in curr_pos if p1 != p2])
                if min_dist_curr < 0.7 * min_dist_prev and min_dist_curr < 20:
                    collision_events.append(t)
    
    # Mark collision events on soliton count plot
    for event in collision_events[:20]:  # Limit to first 20 events for clarity
        axes[0,0].axvline(x=event, color='red', alpha=0.6, linewidth=0.5)
    
    if collision_events:
        event_counts = [soliton_counts[e] for e in collision_events[:20]]
        axes[0,0].scatter(collision_events[:20], event_counts, 
                         color='red', s=20, label='Interaction Events', zorder=5)
        axes[0,0].legend()
    
    # 4. Soliton power distribution
    all_powers = [p for powers in soliton_powers for p in powers]
    if all_powers:
        axes[1,0].hist(np.array(all_powers)*1e3, bins=30, alpha=0.7, color='green', edgecolor='black')
        axes[1,0].set_xlabel('Soliton Power (mW)', fontsize=12)
        axes[1,0].set_ylabel('Frequency', fontsize=12)
        axes[1,0].set_title('Soliton Power Distribution', fontsize=12)
        axes[1,0].grid(True, alpha=0.3)
    
    # 5. Birth/death rate analysis
    birth_death_rate = np.diff(soliton_counts)
    if len(birth_death_rate) > 0:
        axes[1,1].plot(birth_death_rate, 'purple', linewidth=1.5)
        axes[1,1].axhline(y=0, color='black', linestyle='--', alpha=0.7)
        axes[1,1].set_xlabel('Time Steps', fontsize=12)
        axes[1,1].set_ylabel('ΔN (solitons/step)', fontsize=12)
        axes[1,1].set_title('Soliton Birth/Death Rate', fontsize=12)
        axes[1,1].grid(True, alpha=0.3)
        
        # Add annotations for significant events
        birth_events = np.where(birth_death_rate > 0)[0]
        death_events = np.where(birth_death_rate < 0)[0]
        
        for i, event in enumerate(birth_events[:3]):  # Show first 3
            axes[1,1].annotate('Birth', xy=(event, birth_death_rate[event]), 
                              xytext=(event, birth_death_rate[event]+0.5),
                              arrowprops=dict(arrowstyle='->', color='green', alpha=0.7),
                              fontsize=8, color='green')
        
        for i, event in enumerate(death_events[:3]):  # Show first 3
            axes[1,1].annotate('Death', xy=(event, birth_death_rate[event]), 
                              xytext=(event, birth_death_rate[event]-0.5),
                              arrowprops=dict(arrowstyle='->', color='red', alpha=0.7),
                              fontsize=8, color='red')
    
    mod_pow = str(env.power[0]).replace('.','_')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{mod_pow}_{thermal_effect}_run_{idx}_soliton_dynamics.png'), dpi=200, bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, f'{mod_pow}_{thermal_effect}_run_{idx}_soliton_dynamics.svg'), bbox_inches='tight')
    plt.show()


def multistability_landscape_analysis(acav_hist, det_hist, delta_theta, r_hist, save_dir, idx, thermal_effect, env):
    """
    Multistability Landscape Analysis
    
    Maps:
    - Success probability across parameter space
    - Hysteresis effects in system response
    - State classification and transitions
    - Control efficiency optimization
    - System stability regions
    """
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # 1. Success probability map
    det_array = np.array(det_hist) * 1e-9  # Convert to GHz
    theta_array = np.array(delta_theta) * 1e-9 / (2*np.pi*env.tR.item())  # Convert to GHz
    rewards = np.array(r_hist)
    
    # Define success threshold
    success_threshold = 2.0
    success = rewards > success_threshold
    
    # Create 2D grid for success probability
    det_bins = np.linspace(det_array.min(), det_array.max(), 25)
    theta_bins = np.linspace(theta_array.min(), theta_array.max(), 25)
    
    success_map = np.zeros((len(theta_bins)-1, len(det_bins)-1))
    count_map = np.zeros((len(theta_bins)-1, len(det_bins)-1))
    
    for i in range(len(det_bins)-1):
        for j in range(len(theta_bins)-1):
            mask = ((det_array >= det_bins[i]) & (det_array < det_bins[i+1]) &
                   (theta_array >= theta_bins[j]) & (theta_array < theta_bins[j+1]))
            
            if np.sum(mask) > 0:
                success_map[j,i] = np.mean(success[mask])
                count_map[j,i] = np.sum(mask)
    
    # Mask regions with too few samples
    success_map[count_map < 5] = np.nan
    
    im1 = axes[0,0].imshow(success_map, extent=[det_bins[0], det_bins[-1], theta_bins[0], theta_bins[-1]],
                          origin='lower', aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
    axes[0,0].set_xlabel('Pump Detuning (GHz)', fontsize=12)
    axes[0,0].set_ylabel('Thermal Frequency (GHz)', fontsize=12)
    axes[0,0].set_title('Success Probability Map', fontsize=12)
    plt.colorbar(im1, ax=axes[0,0], label='Success Probability')
    
    # Continue with other analyses...
    # [Additional code for hysteresis, state classification, etc.]
    
    mod_pow = str(env.power[0]).replace('.','_')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{mod_pow}_{thermal_effect}_run_{idx}_multistability.png'), dpi=200, bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, f'{mod_pow}_{thermal_effect}_run_{idx}_multistability.svg'), bbox_inches='tight')
    plt.show()


# INTEGRATION FUNCTION FOR YOUR PLOT_ALL_RESULTS
def add_advanced_analysis_to_plot_all_results(env, save_dir, idx, pcav_hist, acav_hist, e_wg_hist, r_hist, det_hist, delta_theta, action_hist, freq, ecav, obtained_spectrum, desired_spectrum_dBm, thermal_effect):
    """
    ADD THIS TO THE END OF YOUR plot_all_results FUNCTION:
    
    # Advanced chaos and soliton analysis
    print("Generating advanced visualizations...")
    add_advanced_analysis_to_plot_all_results(env, save_dir, idx, pcav_hist, acav_hist, e_wg_hist, r_hist, det_hist, delta_theta, action_hist, freq, ecav, obtained_spectrum, desired_spectrum_dBm, thermal_effect)
    """
    
    try:
        print("  → Phase portrait analysis...")
        phase_portrait_analysis(acav_hist, det_hist, delta_theta, save_dir, idx, thermal_effect, env)
        
        print("  → Time-frequency spectrogram...")
        time_frequency_spectrogram(acav_hist, save_dir, idx, thermal_effect, env)
        
        print("  → Chaos metrics analysis...")
        chaos_metrics_analysis(acav_hist, det_hist, delta_theta, r_hist, save_dir, idx, thermal_effect, env)
        
        print("  → Soliton birth/death analysis...")
        soliton_birth_death_analysis(acav_hist, save_dir, idx, thermal_effect, env)
        
        print("  → Multistability landscape...")
        multistability_landscape_analysis(acav_hist, det_hist, delta_theta, r_hist, save_dir, idx, thermal_effect, env)
        
        print("  ✓ All advanced analyses completed!")
        
    except Exception as e:
        print(f"  ✗ Advanced analysis failed: {e}")

# %%
def run_test_processes(run_id, save_dir):
    # Re-create environment and agent inside the process
    env = RL_MRR_Env(seq_len=100, p_max=0.2, p_min=0.05, ctrl_freq=100, thermal_effect='high',\
                  delta_omega_min=-2e6, delta_omega_max=2e6, delta_omega_step=1e4, soft_clamp=False, softness=0.35)
    desired_spectrum = loadmat('desired_spec2.mat')['Ewg'][0]
    desired_spectrum_tensor = torch.tensor(desired_spectrum, device=DEVICE, dtype=torch.complex128)
    from rl_codes.sac_v3 import SACAgent
    config = {
    'input_dim': [env.seq_len, 300+2+2+1],
    'n_actions': 1,
    'alpha': 3e-4,
    'beta': 3e-4,
    'mem_size': int(1e6),
    'run_name': 'mrr_sac_cluster_delayed_toptica_pow_ton_un_norm_high_only_detuningv4',
    'batch_size': 256,
    'dist': 'normal', # 'beta' or 'normal'
    'train':False,
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
    'alpha_per': 0.4,  # Initial value of alpha for PER
    'beta_per': int(2e5),   # Number of steps to reach beta=1 for PER
    }
    agent = SACAgent(input_dim=config['input_dim'], n_actions=config['n_actions'], alpha=config['alpha'], beta=config['beta'],
                    mem_size=config['mem_size'], batch_size=config['batch_size'], dist=config['dist'], run_name=config['run_name'],
                    eval_mode=not(torch.cuda.is_available()), fc_dim=config['fc_dim'], use_per=config['use_per'], bidir=config['bidirectional'])

    agent.load_models()

    state, acav, ecav, pcav = env.reset(10000)
    log_pcav = 10*np.log10(pcav + 1e-12) + 30
    bounds = calc_detuning_distance(env, scale=3)
    den = env.p_max - env.p_min
    obs = np.concatenate((ecav[:,len(env.mu)//2-150:len(env.mu)//2+150]/10,np.zeros((env.seq_len,1)), log_pcav[:,np.newaxis], bounds*np.ones((env.seq_len,1)), 100*env.delta_theta.item()*np.ones((env.seq_len,1))),axis=1)
    print('Chosen power:', env.power)
    r_hist = []
    action_hist = []
    score = 0
    done = False
    pcav_hist = []
    pbar = tqdm(total=env.max_steps-env.init_steps_, ncols=120, position=run_id, desc=f'Run {run_id}')
    idx = 0
    delta_theta = []
    det_hist = []
    e_wg_hist = []
    acav_hist = []
    while not done:
        action = agent.choose_action(obs, True)
        next_state, reward, done, terminal, _, acav_, ecav_, e_wg = env.step(state, action, desired_spectrum_tensor)
        state = next_state
        ecav = ecav_
        curr_pcav = np.sum(np.abs(acav_)**2,keepdims=True)            
        bounds = calc_detuning_distance(env, scale=3)
        ecav_obs = np.concatenate((ecav_[-1,len(env.mu)//2-150:len(env.mu)//2+150]/10, 3*env.current_del_omega/(env.del_omega_ul - env.del_omega_end), 10*np.log10(curr_pcav)+30, bounds, 100*env.delta_theta.item()*np.ones((1,))), axis=0)
        obs_ = np.concatenate((obs[1:], ecav_obs[np.newaxis,:]), axis=0)
        score += reward
        curr_pcav = np.sum(np.abs(acav_)**2)
        pcav_hist.append(curr_pcav)
        r_hist.append(reward)
        action_hist.append(action)
        delta_theta.append(env.delta_theta.item()/(env.tR.item()*2*np.pi))
        det_hist.append(env.current_del_omega.item()/(2*np.pi))
        e_wg_hist.append(e_wg)
        acav_hist.append(acav_)
        idx += env.ctrl_freq
        pbar.update(env.ctrl_freq)
    pbar.close()

    if terminal == False:
        print(f'Run {run_id} completed with score {score} at step {idx}')
        np.save(os.path.join(save_dir, str(run_id) + '_p_cav.npy'), np.array(pcav_hist))
        np.save(os.path.join(save_dir, str(run_id) + '_detuning_theta_sum.npy'), np.array(det_hist)*1e-9 + np.array(delta_theta)*1e-9)
        np.save(os.path.join(save_dir, str(run_id) + '_reward_history.npy'), np.array(r_hist))
        # Prepare spectra for plotting
        freq = (env.sim_tensor['f_pmp'].item() + np.arange(-220,221)*env.FSR.item())*1e-12
        obtained_spectrum = 10*np.log10(np.abs(e_wg_hist[-1])**2) + 30 if len(e_wg_hist) > 0 else np.zeros(441)
        desired_spectrum = loadmat('desired_spec2.mat')['Ewg'][0]
        desired_spectrum_dBm = 10*np.log10(np.abs(desired_spectrum)**2)+30
        desired_spectrum_dBm = np.clip(desired_spectrum_dBm, -60, None)
        plot_all_results(
            env, save_dir, run_id, pcav_hist, acav_hist, e_wg_hist, r_hist, det_hist, delta_theta,
            np.array(action_hist), freq, ecav, obtained_spectrum, desired_spectrum_dBm, env.thermal_effect
        )
        # add_advanced_analysis_to_plot_all_results(env, save_dir, idx, pcav_hist, acav_hist, e_wg_hist, r_hist, det_hist, delta_theta, action_hist, freq, ecav, obtained_spectrum, desired_spectrum_dBm, env.thermal_effect)
# %%
def plot_pcav_freq_mean_std(pcav_files, freq_files, reward_files, save_dir):
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

    norm_freq_pad = 2*freq_pad*env.tR.item()*1e9 / env.alpha  # Convert rad/s to GHz
    mu_pcav = 1e3*np.nanmean(pcav_pad, axis=0)
    sd_pcav = 1e3*np.nanstd(pcav_pad, axis=0)
    mu_freq = np.nanmean(freq_pad, axis=0)
    sd_freq = np.nanstd(freq_pad, axis=0)
    mu_norm_freq = np.nanmean(norm_freq_pad, axis=0)
    sd_norm_freq = np.nanstd(norm_freq_pad, axis=0)

    x = 100*env.tR.item()*np.arange(max_len)*1e6

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ln1, = ax1.plot(x, mu_pcav, color=pcav_color, linewidth=1.8, label='P_cav mean')
    ax1.fill_between(x, mu_pcav - sd_pcav, mu_pcav + sd_pcav, color=pcav_color, alpha=0.25, label='P_cav ±1σ')
    ax1.set_xlabel(r'Time ($\mu s$)', fontsize=18)
    ax1.set_ylabel(r'$P_{cav}$ (mW)', fontsize=16, color=pcav_color)
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
    ax1.set_xlabel(r'Time ($\mu s$)', fontsize=18)
    ax1.set_ylabel(r'$P_{cav}$ (mW)', fontsize=16, color=pcav_color)
    ax1.tick_params(axis='y', labelcolor=pcav_color)
    ax1.grid(True, alpha=0.4)

    ax2 = ax1.twinx()
    ln2, = ax2.plot(x, mu_norm_freq, color=freq_color, linewidth=1.8, label='Freq mean')
    ax2.fill_between(x, mu_norm_freq - sd_norm_freq, mu_norm_freq + sd_norm_freq, color=freq_color, alpha=0.25, label='Freq ±1σ')
    ax2.set_ylabel(r'$\frac{2(\delta _{\Theta} + \delta _0)}{(\alpha + \kappa)}$', fontsize=16, color=freq_color, rotation=90)
    # ax2.yaxis.set_label_rotation(90)
    ax2.tick_params(axis='y', labelcolor=freq_color)

    # ax1.legend([ln1, ln2], ['P_cav mean', 'Freq mean'], loc='upper right', fontsize=12)

    fig.tight_layout()
    ax1.tick_params(axis='x', labelsize=16)
    ax1.tick_params(axis='y', labelsize=16)
    ax2.tick_params(axis='y', labelsize=16)
    fig.savefig(os.path.join(save_dir, 'pcav_norm_freq_mean_std.png'), dpi=200)
    fig.savefig(os.path.join(save_dir, 'pcav_norm_freq_mean_std.svg'))
    plt.show()
    plt.close(fig)

    norm_freq_pad = freq_pad*1e9/(env.un_norm_kappa.item()/(2*np.pi))  # Convert rad/s to GHz
    mu_pcav = 1e3*np.nanmean(pcav_pad, axis=0)
    sd_pcav = 1e3*np.nanstd(pcav_pad, axis=0)
    mu_freq = np.nanmean(freq_pad, axis=0)
    sd_freq = np.nanstd(freq_pad, axis=0)
    mu_norm_freq = np.nanmean(norm_freq_pad, axis=0)
    sd_norm_freq = np.nanstd(norm_freq_pad, axis=0)

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ln1, = ax1.plot(x, mu_pcav, color=pcav_color, linewidth=1.8, label='P_cav mean')
    ax1.fill_between(x, mu_pcav - sd_pcav, mu_pcav + sd_pcav, color=pcav_color, alpha=0.25, label='P_cav ±1σ')
    ax1.set_xlabel(r'Time ($\mu s$)', fontsize=18)
    ax1.set_ylabel(r'$P_{cav}$ (mW)', fontsize=16, color=pcav_color)
    ax1.tick_params(axis='y', labelcolor=pcav_color)
    ax1.grid(True, alpha=0.4)

    ax2 = ax1.twinx()
    ln2, = ax2.plot(x, mu_norm_freq, color=freq_color, linewidth=1.8, label='Freq mean')
    ax2.fill_between(x, mu_norm_freq - sd_norm_freq, mu_norm_freq + sd_norm_freq, color=freq_color, alpha=0.25, label='Freq ±1σ')
    ax2.set_ylabel(r'$\frac{\Delta f_{eff}}{\kappa}$', fontsize=22, color=freq_color, rotation=90, labelpad=10)
    ax2.tick_params(axis='y', labelcolor=freq_color)

    # ax1.legend([ln1, ln2], ['P_cav mean', 'Freq mean'], loc='upper right', fontsize=12)

    fig.tight_layout()
    ax1.tick_params(axis='x', labelsize=16)
    ax1.tick_params(axis='y', labelsize=16)
    ax2.tick_params(axis='y', labelsize=16)
    fig.savefig(os.path.join(save_dir, 'pcav_kappa_freq_mean_std.png'), dpi=200)
    fig.savefig(os.path.join(save_dir, 'pcav_kappa_freq_mean_std.svg'))
    plt.show()
    plt.close(fig)

    # plot the mean and avg of reward histories w.r.t time
    reward_arrs = [np.load(f) for f in reward_files]
    reward_pad = np.full((len(reward_arrs), max_len), np.nan)
    for i, arr in enumerate(reward_arrs):
        reward_pad[i, :len(arr)] = arr
    mu_reward = np.nanmean(reward_pad, axis=0)
    sd_reward = np.nanstd(reward_pad, axis=0)
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ln1, = ax1.plot(x, mu_reward, color='green', linewidth=1.8, label='Reward mean')
    ax1.fill_between(x, mu_reward - sd_reward, mu_reward + sd_reward, color='green', alpha=0.25, label='Reward ±1σ')
    ax1.set_xlabel(r'Time ($\mu s$)', fontsize=20)
    ax1.set_ylabel(r'Reward', fontsize=20)
    ax1.tick_params(axis='y', labelcolor='green')
    ax1.grid(True, alpha=0.4)
    fig.tight_layout()
    ax1.tick_params(axis='x', labelsize=20)
    ax1.tick_params(axis='y', labelsize=20)
    fig.savefig(os.path.join(save_dir, 'reward_mean_std.png'), dpi=200)
    fig.savefig(os.path.join(save_dir, 'reward_mean_std.svg'))
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
    save_dir = os.path.join('./results', agent.run_name, env.thermal_effect,'new')
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
    # plot_reward_histories_sigma(npy_files, N=5, S=0, label='Reward', color='C0')
    # plot_reward_histories_min_max(npy_files, N=5, S=0, label='Reward', color='C0')
    pcav_files = sorted(glob.glob(os.path.join(save_dir, '*_p_cav.npy')))
    freq_files = sorted(glob.glob(os.path.join(save_dir, '*_detuning_theta_sum.npy')))
    reward_files = sorted(glob.glob(os.path.join(save_dir, '*_reward_history.npy')))
    plot_pcav_freq_mean_std(pcav_files, freq_files, reward_files, save_dir)