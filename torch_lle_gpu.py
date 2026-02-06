import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
from scipy import constants as cts
from tqdm import tqdm

# --- 1. CONFIGURATION & SETUP ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Running on: {DEVICE}")

# Constants
C0 = 299792458
H_BAR = cts.hbar

# Precision settings
DTYPE_C = torch.complex128 
DTYPE_R = torch.float64

# --- 2. LLE SOLVER CLASS ---
class LLESolver:
    def __init__(self, pt_file_path):
        # 1. Load Pre-processed Data
        print(f"Loading data from {pt_file_path}...")
        try:
            # Load with weights_only=False to support complex/numpy types if needed, 
            # though standard tensors work with True.
            data = torch.load(pt_file_path, map_location=DEVICE, weights_only=False)
        except FileNotFoundError:
            raise FileNotFoundError(f"Could not find {pt_file_path}. Run pre-processing first.")

        # Extract Dictionaries
        self.disp_t = data['disp_tensor']
        self.res_t = data['res_tensor']
        self.sim_t = data['sim_tensor']
        
        # 2. Extract Physics Parameters
        # Note: ensuring everything is on the correct device/dtype
        self.gamma = self.res_t['gamma'].to(DTYPE_R)
        self.L = (2 * torch.pi * self.res_t['R']).to(DTYPE_R)
        
        self.fpmp = self.sim_t['f_pmp'].to(DTYPE_R)
        
        # Handle ind_pmp (can be scalar or list)
        ind_pmp_raw = self.sim_t['ind_pmp']
        if ind_pmp_raw.ndim == 0:
            self.ind_pmp = [ind_pmp_raw.long().item()]
        else:
            self.ind_pmp = ind_pmp_raw.long().cpu().numpy().tolist()

        # Mode Setup
        mu_sim = self.sim_t['mucenter'].long()
        self.mu = torch.arange(mu_sim[0], mu_sim[1]+1, device=DEVICE)
        self.mu0 = (torch.where(self.mu == 0)[0][0] + 1).item()
        
        # FSR & Time
        D1 = self.disp_t['D1'].to(DTYPE_R)
        self.FSR = D1 / (2 * torch.pi)
        self.tR = 1 / self.FSR
        
        # Rates
        omega0 = 2 * torch.pi * self.fpmp
        Q0 = self.res_t['Qi'].to(DTYPE_R)
        Qc = self.res_t['Qc'].to(DTYPE_R)
        self.kext = (omega0[0] / Qc) * self.tR
        k0 = (omega0[0] / Q0) * self.tR
        self.alpha = k0 + self.kext

        # Dispersion
        self.Dint = self.disp_t['Dint_new'].to(DTYPE_R)
        self.Dint = self.Dint - self.Dint[self.mu0-1]
        self.Dint_shift = torch.fft.ifftshift(self.Dint)

        # Input Field (Pump) - Precompute
        self.Ain = torch.zeros(len(self.fpmp), len(self.mu), dtype=DTYPE_C, device=DEVICE)
        Ppmp = self.sim_t['Pin'].to(DTYPE_R)
        phi_pmp = self.sim_t['phi_pmp'].to(DTYPE_R)
        
        for ii in range(len(self.fpmp)):
            E_temp = torch.zeros(len(self.mu), dtype=DTYPE_C, device=DEVICE)
            # Create frequency domain pump
            E_temp[int(self.mu0 + self.ind_pmp[ii])] = torch.sqrt(Ppmp[ii]) * len(self.mu)
            # Convert to time domain (Ain)
            self.Ain[ii] = torch.fft.ifft(torch.fft.fftshift(E_temp)) * torch.exp(-1j * phi_pmp[ii])

        # State Initialization
        self.state = self.sim_t['DKS_init'].to(DTYPE_C)

        # Raman Setup
        self._init_raman()

    def _init_raman(self):
        # Parameters (SiN default)
        self.f_R = 0.15
        tau1 = 15e-15
        tau2 = 120e-15
        N = len(self.mu)
        
        dt_fast = (self.tR / float(N)).to(DTYPE_R)
        t_fast = torch.arange(N, device=DEVICE, dtype=DTYPE_R) * dt_fast
        
        factor = (tau1**2 + tau2**2) / (tau1 * tau2**2)
        h_R = factor * torch.exp(-t_fast / tau2) * torch.sin(t_fast / tau1)
        
        integral = torch.sum(h_R) * dt_fast
        h_R = torch.where(integral > 0, h_R / integral, h_R)
        
        h_R_fft = (h_R * dt_fast).to(DTYPE_C)
        self.H_R_FFT = torch.fft.fft(h_R_fft)

    # --- KERNELS ---
    @staticmethod
    def raman_response(A, H_R_FFT):
        power = torch.abs(A) ** 2
        return torch.fft.ifft(torch.fft.fft(power) * H_R_FFT).real

    @staticmethod
    def NL_with_Raman(uu, raman_resp, gamma, L, f_R):
        kerr = torch.abs(uu) ** 2
        if f_R > 0.0:
            total = (1.0 - f_R) * kerr + f_R * raman_resp
        else:
            total = kerr
        return -1j * (gamma * L * total)

    @staticmethod
    def FFT_Lin(alpha, Dint_shift, detuning, tR):
        # Linear operator L = -alpha/2 - i*delta + i*Dint
        return (-alpha / 2) + 1j * (Dint_shift - detuning) * tR

    @staticmethod
    def Fdrive(it, Ain, del_omega_all, Dint, t_sim, ind_pmp, mu0):
        # Optimized Drive calculation for potentially multiple pumps
        Force = torch.zeros(Ain.shape[1], device=DEVICE, dtype=DTYPE_C)
        
        for ii in range(len(ind_pmp)):
             if ii > 0:
                 # Phase rotation for secondary pumps relative to the first
                 sigma = (2*del_omega_all[ii, it] + Dint[(mu0)+ind_pmp[ii]] - 0.5*del_omega_all[0, it])*t_sim[it]
             else:
                 sigma = torch.zeros(1, device=DEVICE, dtype=DTYPE_R)
             
             Force = Force - 1j * Ain[ii] * torch.exp(1j * sigma)
        return Force

    @staticmethod
    def ssfm_step(A0, alpha, Dint_shift, detuning, tR, gamma, L, 
                  dt, kext, Fdrive_val, H_R_FFT, f_R):
        
        # 1. Add Drive
        A0 = A0 + Fdrive_val * torch.sqrt(kext) * dt
        
        # 2. Linear Operator
        lin_op = LLESolver.FFT_Lin(alpha, Dint_shift, detuning, tR) * dt / 2
        L_h_prop = torch.exp(lin_op)
        
        # 3. Half-step Linear
        A_L_h_prop = torch.fft.ifft(torch.fft.fft(A0) * L_h_prop)
        
        # 4. Nonlinear Predictor
        raman = LLESolver.raman_response(A0, H_R_FFT)
        NL_0 = LLESolver.NL_with_Raman(A0, raman, gamma, L, f_R)
        
        # 5. Fixed Iterations (Avoiding CPU synchronization)
        A_curr = A0
        fixed_iters = 5 
        
        for _ in range(fixed_iters):
            raman_i = LLESolver.raman_response(A_curr, H_R_FFT)
            NL_i = LLESolver.NL_with_Raman(A_curr, raman_i, gamma, L, f_R)
            
            NL_prop = (NL_0 + NL_i) * dt / 2
            
            # Full Step: Linear(1/2) -> Nonlinear -> Linear(1/2)
            A_curr = torch.fft.ifft(torch.fft.fft(A_L_h_prop * torch.exp(NL_prop)) * L_h_prop)
            
        return A_curr

    # --- MAIN SOLVER ---
    def solve(self, Nt, num_probe):
        # Prepare Sweep Arrays
        del_omega_init = self.sim_t['domega_init'].to(DTYPE_R)
        del_omega_end = self.sim_t['domega_end'].to(DTYPE_R)
        ind_sweep = self.sim_t['ind_pump_sweep']
        
        t_end = self.sim_t['Tscan'].to(DTYPE_R) * self.tR
        t_sim = torch.linspace(0, t_end[0], Nt, device=DEVICE, dtype=DTYPE_R)
        
        # Detuning vector setup
        del_omega_all = torch.zeros(len(self.fpmp), Nt, device=DEVICE, dtype=DTYPE_R)
        
        # Check if start == end (constant detuning)
        if torch.allclose(del_omega_init, del_omega_end):
             del_omega_all[:] = del_omega_init[0]
        else:
            xx = torch.arange(1, Nt+1, device=DEVICE)
            # Assuming sweeping the pumps in ind_sweep
            sweep_indices = ind_sweep.long().cpu().numpy()
            if sweep_indices.ndim == 0: sweep_indices = [sweep_indices.item()]
            
            for ii in sweep_indices:
                 del_omega_all[ii, :] = del_omega_init + xx/Nt * (del_omega_end - del_omega_init)

        # Pre-allocate Results on GPU
        probe_indices = np.linspace(0, Nt-1, num_probe, dtype=int)
        saved_u = torch.zeros(num_probe, len(self.mu), dtype=DTYPE_C, device=DEVICE)
        saved_det = torch.zeros(num_probe, device=DEVICE)
        
        u_curr = self.state
        dt = 1
        
        print("Starting Simulation...")
        p_idx = 0
        
        # Main Loop
        for it in tqdm(range(Nt), ncols=100):
            
            curr_det = del_omega_all[0, it]
            
            # Calculate Drive
            f_val = self.Fdrive(it, self.Ain, del_omega_all, self.Dint, t_sim, self.ind_pmp, self.mu0)
            
            # Take Step
            u_curr = self.ssfm_step(u_curr, self.alpha, self.Dint_shift, curr_det, self.tR, 
                                    self.gamma, self.L, dt, self.kext, f_val, 
                                    self.H_R_FFT, self.f_R)
            
            # Save Probe Data
            if p_idx < num_probe and it >= probe_indices[p_idx]:
                saved_u[p_idx] = u_curr
                saved_det[p_idx] = curr_det
                p_idx += 1

        # Return results dictionary (moved to CPU for plotting/saving)
        return {
            'u_probe': saved_u.cpu().numpy(),
            'detuning': saved_det.cpu().numpy(),
            't_sim': t_sim.cpu().numpy(),
            'FSR': self.FSR.cpu().numpy(),
            'fpmp': self.fpmp.cpu().numpy(),
            'alpha': self.alpha.cpu().numpy(),
            'kext': self.kext.cpu().numpy(),
            'tR': self.tR.cpu().numpy()
        }

# --- 3. COMPILATION (PyTorch 2.0+) ---
if int(torch.__version__.split('.')[0]) >= 2:
    print("Compiling Kernels...")
    try:
        # 1. Compile the underlying functions
        # We compile the function directly from the class dictionary or standard access
        # mode='default' is safest for complex numbers
        
        compiled_ssfm = torch.compile(LLESolver.ssfm_step, mode='default')
        compiled_fdrive = torch.compile(LLESolver.Fdrive, mode='default')
        compiled_raman = torch.compile(LLESolver.raman_response, mode='default')
        compiled_nl = torch.compile(LLESolver.NL_with_Raman, mode='default')
        
        # 2. Re-assign them as staticmethods
        # This prevents Python from passing 'self' automatically
        LLESolver.ssfm_step = staticmethod(compiled_ssfm)
        LLESolver.Fdrive = staticmethod(compiled_fdrive)
        LLESolver.raman_response = staticmethod(compiled_raman)
        LLESolver.NL_with_Raman = staticmethod(compiled_nl)
        
    except Exception as e:
        print(f"Compilation warning: {e}. Running uncompiled.")

# --- 4. PLOTTING HELPERS ---
def save_plots(data, save_dir='results'):
    os.makedirs(save_dir, exist_ok=True)
    print(f"Saving plots to '{save_dir}/'...")
    
    # Extract Data
    u_probe = data['u_probe'] # (Time, N)
    detuning = data['detuning']
    FSR = data['FSR']
    fpmp = data['fpmp']
    tR = data['tR']
    alpha = data['alpha']
    
    # Calculate Derived Fields
    Acav = np.sqrt(alpha/2) * u_probe * np.exp(1j*np.pi) / np.sqrt(u_probe.shape[1])
    Ecav = np.fft.fftshift(np.fft.fft(Acav, axis=1), axes=1) / np.sqrt(u_probe.shape[1])
    Ewg = Ecav # Approximation for plotting
    
    # Calculate Power Evolution
    Pcomb = np.sum(np.abs(Ecav)**2, axis=1)

    # 1. INTENSITY WATERFALL
    plt.figure(figsize=(10, 6))
    plt.imshow(np.abs(Acav).T, aspect='auto', cmap='jet', origin='lower')
    plt.xlabel('Step Index', fontsize=12)
    plt.ylabel(r"Time ($t_R$)", fontsize=12)
    plt.colorbar(label='Intracavity Power')
    plt.title("Soliton Evolution (Time Domain)")
    plt.savefig(os.path.join(save_dir, 'intensity_waterfall.png'), dpi=300)
    plt.close()

    # 2. SPECTRAL WATERFALL
    plt.figure(figsize=(10, 6))
    E_wg_dbm = 10 * np.log10(np.abs(Ewg)**2 + 1e-20) + 30
    plt.imshow(E_wg_dbm.T, aspect='auto', cmap='jet', origin='lower', vmin=-80, vmax=0)
    plt.colorbar(label='Power (dBm)')
    plt.xlabel('Step Index', fontsize=12)
    plt.ylabel('Mode Number', fontsize=12)
    plt.title('Spectral Evolution')
    plt.savefig(os.path.join(save_dir, 'spectral_waterfall.png'), dpi=300)
    plt.close()

    # 3. DETUNING CURVE
    plt.figure(figsize=(10, 4))
    det_ghz = detuning / (2*np.pi*1e9)
    plt.plot(det_ghz, Pcomb*1e3)
    # plt.gca().invert_xaxis() # Often detuning scans go high->low
    plt.xlabel('Detuning (GHz)')
    plt.ylabel('Intracavity Power (mW)')
    plt.title('Power vs Detuning')
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'power_vs_detuning.png'), dpi=300)
    plt.close()

    # 4. FINAL COMB SPECTRUM (At last step)
    idx = -1
    freq_x = (fpmp + np.arange(-(len(Ewg[idx])-1)//2, len(Ewg[idx])//2 + 1, 1) * FSR) * 1e-12
    
    # Logic for colored vlines
    x = freq_x
    y = 10 * np.log10(np.abs(Ewg[idx])**2 + 1e-20) + 30
    
    # Expand arrays for vlines
    x_ = np.repeat(x, 3)
    y_ = np.zeros_like(x_)
    floor = -100
    y_[0::3] = floor; y_[1::3] = y; y_[2::3] = floor
    
    plt.figure(figsize=(12, 4))
    plt.style.use('seaborn-v0_8-white')
    colors = cm.gist_rainbow(np.linspace(0, 1, len(Ewg[idx])))
    
    for i in range(len(Ewg[idx])):
        plt.vlines(x[i], ymin=floor, ymax=y[i], colors=colors[i], linestyles='-', alpha=1)
        
    plt.ylim(-100, 10)
    plt.xlabel("Frequency (THz)")
    plt.ylabel("Power (dBm)")
    plt.title("Final Output Spectrum")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'final_spectrum_comb.svg'), format='svg')
    plt.close()
    plt.style.use('default')

    # 5. AUTOCORRELATION (At last step)
    s = np.abs(Acav[idx])
    autocorr = np.correlate(s**2, s**2, mode='same')
    t = np.linspace(-tR/2, tR/2, len(autocorr)) * 1e12
    
    plt.figure(figsize=(10, 4))
    plt.plot(t, autocorr)
    plt.xlabel('Time (ps)')
    plt.ylabel('Autocorrelation')
    plt.title("Autocorrelation (Final Step)")
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'final_autocorr.png'), dpi=300)
    plt.close()

# --- 5. MAIN EXECUTION ---
if __name__ == "__main__":
    
    PT_FILE = 'mrr_data.pt'
    SAVE_DIR = 'results'
    
    # Simulation Parameters
    Nt = 200000 
    num_probe = 5000 
    
    try:
        # Initialize
        solver = LLESolver(PT_FILE)
        
        # Run
        results = solver.solve(Nt, num_probe)
        
        # Plot & Save
        save_plots(results, SAVE_DIR)
        
        print("Done! Check the 'results/' directory.")
        
    except Exception as e:
        print(f"An error occurred: {e}")