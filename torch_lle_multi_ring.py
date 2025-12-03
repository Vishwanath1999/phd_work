import torch
import numpy as np
from scipy.io import loadmat, savemat
from scipy import constants as cts
from tqdm import tqdm

device = torch.device('cpu')

# Load your original parameters
disp = loadmat('disp.mat')
res = loadmat('res.mat')
sim = loadmat('sim.mat')

c0 = 299792458
h_bar = cts.hbar

# %%
def dict_to_tensor(dic):
    """Convert dictionary to tensor, handling nested structures"""
    dic.pop('__header__', None)
    dic.pop('__version__', None)
    dic.pop('__globals__', None)
    dic.pop('dispfile', None)
    
    tensor_dic = dict()
    for key in dic:
        if isinstance(dic[key], np.ndarray):
            dic[key] = np.where(dic[key] == 'None', None, dic[key])
            dic[key] = dic[key].astype(np.float32)
            tensor_dic[key] = torch.tensor(dic[key], device=device)
        else:
            tensor_dic[key] = dic[key]
    return tensor_dic

def tensor_to_dict(tensor):
    """Convert tensor dictionary back to numpy"""
    dic = dict()
    for key in tensor:
        if isinstance(tensor[key], torch.Tensor):
            dic[key] = tensor[key].cpu().numpy()
        else:
            dic[key] = tensor[key]
    return dic

# Convert loaded data
disp_tensor = dict_to_tensor(disp)
res_tensor = dict_to_tensor(res)
sim_tensor = dict_to_tensor(sim)

# Extract parameters (Ring 1)
disp_tensor['D1'] = disp_tensor['D1'][0]
disp_tensor['FSR'] = disp_tensor['FSR'][0]
disp_tensor['FSR_center'] = disp_tensor['FSR_center'][0]

for key in res_tensor:
    res_tensor[key] = res_tensor[key][0]

for key in sim_tensor:
    if sim_tensor[key].ndim > 1:
        sim_tensor[key] = sim_tensor[key][0]

# %%
# ============================================
# COUPLED RESONATOR PARAMETERS
# ============================================
# Ring 1 parameters
ng1 = disp_tensor['ng']
R1 = res_tensor['R']
gamma1 = res_tensor['gamma']
L1 = 2*torch.pi*R1
Q01 = res_tensor['Qi']
Qc1 = res_tensor['Qc']
Dint1 = disp_tensor['Dint_sim'][0]

# Ring 2 parameters (can be different or same as Ring 1)
# For demonstration, assume identical rings
R2 = R1.clone()
gamma2 = gamma1.clone()
L2 = L1.clone()
Q02 = Q01.clone()
Qc2 = Qc1.clone()
Dint2 = Dint1.clone()  # Can be modified for detuning between rings

# Coupling coefficient between rings (THIS IS THE KEY NEW PARAMETER)
# Typical values: 0.01 to 1.0 * FSR (in rad/s)
D1 = disp_tensor['D1']
FSR = D1/(2*torch.pi)
mu_coupling = 0.25 * 2*torch.pi*FSR  # Coupling strength (rad/round-trip)

# %%
# Pump parameters
fpmp = sim_tensor['f_pmp']
Ppmp = sim_tensor['Pin']
phi_pmp = sim_tensor['phi_pmp']
num_probe = sim_tensor['num_probe']
fcenter = sim_tensor['f_center']

# Initial conditions
DKSinit_real = torch.real(sim_tensor['DKS_init'])
if sim_tensor['DKS_init'].dtype == torch.complex128:
    DKSinit_imag = torch.imag(sim_tensor['DKS_init'])
else:
    DKSinit_imag = torch.zeros_like(DKSinit_real, device=device)

DKS_init1 = torch.complex(DKSinit_real, DKSinit_imag)
DKS_init2 = torch.zeros_like(DKS_init1)  # Ring 2 starts from vacuum

# %%
# Frequency grid
omega0 = 2*torch.pi*fpmp
omega_center = 2*torch.pi*fcenter
tR = 1/FSR
T = 1*tR

# Loss and coupling rates
kext1 = omega0/Qc1 * tR
k01 = omega0/Q01 * tR
alpha1 = k01 + kext1

kext2 = omega0/Qc2 * tR
k02 = omega0/Q02 * tR
alpha2 = k02 + kext2

# Detuning scan
del_omega_init = sim_tensor['domega_init']
del_omega_end = sim_tensor['domega_end']
ind_sweep = sim_tensor['ind_pump_sweep']
t_end = sim_tensor['Tscan']

# Mode indices
mu_sim = sim_tensor['mucenter']
mu = torch.arange(mu_sim[0], mu_sim[1]+1, device=device)
mu0 = torch.where(mu == 0)[0][0]

# Center dispersion
Dint1 = Dint1 - Dint1[mu0]
Dint2 = Dint2 - Dint2[mu0]

# %%
# Time grid
dt = 1
t_end = t_end*tR
t_ramp = t_end
Nt = torch.round(t_ramp/tR/dt)[0].int()
theta = torch.linspace(0, 2*torch.pi, len(mu), device=device)

del_omega_tot = torch.abs(del_omega_end) + torch.abs(del_omega_init)
del_omega_perc = -1*torch.sign(del_omega_end+del_omega_init)*(torch.abs(del_omega_end+del_omega_init)/2)/del_omega_tot
t_sim = torch.linspace(-t_ramp[0]/2 + del_omega_perc[0]*t_ramp[0], 
                        t_ramp[0]/2 + del_omega_perc[0]*t_ramp[0], Nt, device=device)

# Detuning sweep
del_omega_all = torch.ones(len(fpmp), Nt, device=device)
xx = torch.arange(1, Nt+1, device=device)
for ii in ind_sweep.cpu().numpy().astype(int):
    del_omega_all[ii,:] = del_omega_init + xx/Nt * (del_omega_end - del_omega_init)

# %%
# Driving force setup
ind_pmp = [ii for ii in sim_tensor['ind_pmp']]
Ein1 = torch.zeros(len(fpmp), len(mu), dtype=torch.complex128, device=device)
Ain1 = torch.zeros(len(fpmp), len(mu), dtype=torch.complex128, device=device)

# Only pump Ring 1 (Ring 2 can be pumped separately if desired)
for ii in range(len(fpmp)):
    Ein1[ii, int(mu0+ind_pmp[ii])] = torch.sqrt(Ppmp[ii])*len(mu)
    Ain1[ii] = torch.fft.ifft(torch.fft.fftshift(Ein1[ii]))*torch.exp(1j*phi_pmp[ii])

# Optional: Pump Ring 2 (set to zero for passive Ring 2)
Ein2 = torch.zeros_like(Ein1)
Ain2 = torch.zeros_like(Ain1)

# %%
# Initial fields
u1 = DKS_init1.clone()
u2 = DKS_init2.clone()

# Storage for probing
saved_data = dict()
num_probe_int = num_probe[0].cpu().numpy().astype(int)
saved_data['u1_probe'] = torch.zeros(num_probe_int, len(u1), dtype=torch.complex128, device=device)
saved_data['u2_probe'] = torch.zeros(num_probe_int, len(u2), dtype=torch.complex128, device=device)
saved_data['detuning'] = torch.zeros(num_probe_int, device=device)
saved_data['t_sim'] = torch.zeros(num_probe_int, device=device)
saved_data['mu_coupling'] = mu_coupling

Dint1_shift = torch.fft.ifftshift(Dint1)
Dint2_shift = torch.fft.ifftshift(Dint2)

# %%
# ============================================
# COUPLED LLE OPERATORS
# ============================================

def Fdrive1(it):
    """Driving force for Ring 1"""
    Force = torch.zeros_like(theta, dtype=torch.complex128)
    for ii in range(len(fpmp)):
        if ii > 0:
            sigma = (2*del_omega_all[ii,it] + Dint1[mu0+ind_pmp[ii]] - 0.5*del_omega_all[0][it])*t_sim[it]
        else:
            sigma = torch.zeros(1, device=device)
        Force = Force - 1j*Ain1[ii]*torch.exp(1j*sigma)
    return Force

def Fdrive2(it):
    """Driving force for Ring 2 (if pumped)"""
    Force = torch.zeros_like(theta, dtype=torch.complex128)
    for ii in range(len(fpmp)):
        if ii > 0:
            sigma = (2*del_omega_all[ii,it] + Dint2[mu0+ind_pmp[ii]] - 0.5*del_omega_all[0][it])*t_sim[it]
        else:
            sigma = torch.zeros(1, device=device)
        Force = Force - 1j*Ain2[ii]*torch.exp(1j*sigma)
    return Force

@torch.jit.script
def FFT_Lin1(it: int, alpha1: torch.Tensor, Dint1_shift: torch.Tensor, 
             del_omega_all: torch.Tensor, tR: torch.Tensor) -> torch.Tensor:
    """Linear operator for Ring 1"""
    return (-alpha1 / 2) + 1j * (Dint1_shift - del_omega_all[0, it]) * tR

@torch.jit.script
def FFT_Lin2(it: int, alpha2: torch.Tensor, Dint2_shift: torch.Tensor,
             del_omega_all: torch.Tensor, tR: torch.Tensor) -> torch.Tensor:
    """Linear operator for Ring 2"""
    return (-alpha2 / 2) + 1j * (Dint2_shift - del_omega_all[0, it]) * tR

@torch.jit.script
def NL(uu: torch.Tensor, gamma: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
    """Nonlinear operator"""
    return -1j * (gamma * L * torch.square(torch.abs(uu)))

@torch.jit.script
def Coupling_term(u_other: torch.Tensor, mu_coupling: torch.Tensor) -> torch.Tensor:
    """Coupling term between rings"""
    return -1j * mu_coupling * u_other

# %%
# ============================================
# COUPLED SSFM STEP
# ============================================

# ============================================
# COUPLED SSFM STEP
# ============================================

def coupled_ssfm_step(A1, A2, it, tol=1e-6, max_iter=20):
    """
    Split-step Fourier method for coupled rings
    More robust implementation with better numerical stability
    """
    
    # Add driving forces
    A1 = A1 + Fdrive1(int(it)) * torch.sqrt(kext1) * dt
    A2 = A2 + Fdrive2(int(it)) * torch.sqrt(kext2) * dt
    
    # Check for NaN/Inf in inputs
    if torch.any(torch.isnan(A1)) or torch.any(torch.isnan(A2)):
        raise Exception("NaN detected in input fields")
    
    # Linear half-step propagators
    L1_h_prop = torch.exp(FFT_Lin1(it, alpha1, Dint1_shift, del_omega_all, tR)*dt/2)
    L2_h_prop = torch.exp(FFT_Lin2(it, alpha2, Dint2_shift, del_omega_all, tR)*dt/2)
    
    # Apply linear half-step in frequency domain
    A1_fft = torch.fft.fft(A1)
    A2_fft = torch.fft.fft(A2)
    
    A1_L_h = torch.fft.ifft(A1_fft * L1_h_prop)
    A2_L_h = torch.fft.ifft(A2_fft * L2_h_prop)
    
    # Initial guess for iterative step (use input fields)
    A1_h = A1_L_h.clone()
    A2_h = A2_L_h.clone()
    
    # Predictor step - use simple forward Euler for initial guess
    NL1_0 = NL(A1_L_h, gamma1, L1)
    NL2_0 = NL(A2_L_h, gamma2, L2)
    C1_0 = Coupling_term(A2_L_h, mu_coupling)
    C2_0 = Coupling_term(A1_L_h, mu_coupling)
    
    # Predictor
    A1_pred = A1_L_h * torch.exp((NL1_0 + C1_0) * dt)
    A2_pred = A2_L_h * torch.exp((NL2_0 + C2_0) * dt)
    
    # Corrector iterations
    success = False
    err = float('inf')
    
    for iter_count in range(max_iter):
        # Store previous iteration
        A1_prev = A1_h.clone()
        A2_prev = A2_h.clone()
        
        # Evaluate at current guess
        NL1_h = NL(A1_h, gamma1, L1)
        NL2_h = NL(A2_h, gamma2, L2)
        C1_h = Coupling_term(A2_h, mu_coupling)
        C2_h = Coupling_term(A1_h, mu_coupling)
        
        # Trapezoidal rule for nonlinear + coupling terms
        NL1_avg = (NL1_0 + NL1_h) * dt / 2
        NL2_avg = (NL2_0 + NL2_h) * dt / 2
        C1_avg = (C1_0 + C1_h) * dt / 2
        C2_avg = (C2_0 + C2_h) * dt / 2
        
        # Limit the exponential arguments to prevent overflow
        exp1_arg = NL1_avg + C1_avg
        exp2_arg = NL2_avg + C2_avg
        
        # Clamp to prevent numerical overflow
        exp1_arg = torch.clamp(exp1_arg.real, -10, 10) + 1j * torch.clamp(exp1_arg.imag, -10, 10)
        exp2_arg = torch.clamp(exp2_arg.real, -10, 10) + 1j * torch.clamp(exp2_arg.imag, -10, 10)
        
        # Update fields
        A1_h = A1_L_h * torch.exp(exp1_arg)
        A2_h = A2_L_h * torch.exp(exp2_arg)
        
        # Apply final linear half-step
        A1_new = torch.fft.ifft(torch.fft.fft(A1_h) * L1_h_prop)
        A2_new = torch.fft.ifft(torch.fft.fft(A2_h) * L2_h_prop)
        
        # Check convergence
        if torch.norm(A1_prev, 2) > 0 and torch.norm(A2_prev, 2) > 0:
            err1 = torch.norm(A1_new - A1_prev, 2) / torch.norm(A1_prev, 2)
            err2 = torch.norm(A2_new - A2_prev, 2) / torch.norm(A2_prev, 2)
            err = max(err1.item(), err2.item())
        else:
            err = torch.norm(A1_new - A1_prev, 2).item() + torch.norm(A2_new - A2_prev, 2).item()
        
        # Check for NaN
        if torch.any(torch.isnan(A1_new)) or torch.any(torch.isnan(A2_new)) or torch.isnan(torch.tensor(err)):
            raise Exception(f"NaN detected at iteration {iter_count}, reducing coupling strength might help")
        
        if err < tol:
            success = True
            break
        
        # Update for next iteration
        A1_h = A1_new
        A2_h = A2_new
        
        # Adaptive tolerance - if converging slowly, relax tolerance
        if iter_count > 10 and err < 1e-3:
            success = True
            break
    
    if success:
        return A1_new, A2_new
    else:
        # If convergence fails, try with reduced coupling for this step
        print(f"Convergence warning at step {it}: err={err:.2e}, using reduced coupling")
        reduced_coupling = mu_coupling * 0.1
        
        # Simple step with reduced coupling
        NL1_simple = NL(A1_L_h, gamma1, L1) * dt
        NL2_simple = NL(A2_L_h, gamma2, L2) * dt
        C1_simple = Coupling_term(A2_L_h, reduced_coupling) * dt
        C2_simple = Coupling_term(A1_L_h, reduced_coupling) * dt
        
        A1_simple = torch.fft.ifft(torch.fft.fft(A1_L_h * torch.exp(NL1_simple + C1_simple)) * L1_h_prop)
        A2_simple = torch.fft.ifft(torch.fft.fft(A2_L_h * torch.exp(NL2_simple + C2_simple)) * L2_h_prop)
        
        return A1_simple, A2_simple

# %%
# ============================================
# MAIN SOLVER WITH PROBING
# ============================================

def SaveStatus_Callback(it, Nt, saved_data, u1, u2, param):
    """Save field snapshots during simulation"""
    if it*param['num_probe']/Nt > param['probe']:
        saved_data['u1_probe'][param['probe'],:] = u1
        saved_data['u2_probe'][param['probe'],:] = u2
        saved_data['detuning'][param['probe']] = del_omega_all[0][it]
        saved_data['t_sim'][param['probe']] = t_sim[it]
        param['probe'] += 1
    return param

def CoupledMainSolver(Nt, saved_data, u1, u2):
    """Main simulation loop for coupled resonators - optimized"""
    param = {'probe': 0, 'num_probe': num_probe_int}
    
    print("JIT compiling functions (first iteration may be slow)...")
    _ = coupled_ssfm_step(u1.clone(), u2.clone(), 0)
    print("JIT compilation complete. Starting main simulation...")
    
    for it in tqdm(range(Nt), ncols=120, desc="Coupled LLE Simulation"):
        u1, u2 = coupled_ssfm_step(u1, u2, it)
        param = SaveStatus_Callback(it, Nt, saved_data, u1, u2, param)
    
    return u1, u2

# %%
# Run coupled simulation
print("Starting coupled MRR simulation...")
print(f"Coupling strength: {mu_coupling/(2*np.pi*FSR):.3f} × FSR")
u1_final, u2_final = CoupledMainSolver(Nt, saved_data, u1, u2)

# Save results
saved_data_numpy = tensor_to_dict(saved_data)
savemat('coupled_SSFM_data.mat', saved_data_numpy)
print("Simulation complete! Data saved to coupled_SSFM_data.mat")

# %%
# ============================================
# PLOTTING FUNCTIONS FOR COUPLED RINGS
# ============================================

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def plot_coupled_fields_evolution():
    """Plot field evolution in both rings"""
    fig = plt.figure(figsize=(14, 8))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # Ring 1 amplitude
    ax1 = fig.add_subplot(gs[0, 0])
    u1_probe = saved_data['u1_probe'].cpu().numpy()
    im1 = ax1.imshow(np.abs(u1_probe).T, aspect='auto', cmap='jet', origin='lower')
    ax1.set_title('Ring 1: Field Amplitude |A₁|')
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Mode Index')
    plt.colorbar(im1, ax=ax1, label='|A₁|')
    
    # Ring 2 amplitude
    ax2 = fig.add_subplot(gs[0, 1])
    u2_probe = saved_data['u2_probe'].cpu().numpy()
    im2 = ax2.imshow(np.abs(u2_probe).T, aspect='auto', cmap='jet', origin='lower')
    ax2.set_title('Ring 2: Field Amplitude |A₂|')
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Mode Index')
    plt.colorbar(im2, ax=ax2, label='|A₂|')
    
    # Ring 1 phase
    ax3 = fig.add_subplot(gs[1, 0])
    im3 = ax3.imshow(np.angle(u1_probe).T, aspect='auto', cmap='twilight', origin='lower')
    ax3.set_title('Ring 1: Field Phase ∠A₁')
    ax3.set_xlabel('Time Step')
    ax3.set_ylabel('Mode Index')
    plt.colorbar(im3, ax=ax3, label='Phase (rad)')
    
    # Ring 2 phase
    ax4 = fig.add_subplot(gs[1, 1])
    im4 = ax4.imshow(np.angle(u2_probe).T, aspect='auto', cmap='twilight', origin='lower')
    ax4.set_title('Ring 2: Field Phase ∠A₂')
    ax4.set_xlabel('Time Step')
    ax4.set_ylabel('Mode Index')
    plt.colorbar(im4, ax=ax4, label='Phase (rad)')
    
    plt.suptitle('Coupled MRR Field Evolution', fontsize=14, fontweight='bold')
    plt.show()

def plot_spectra_comparison(idx=-1):
    """Compare spectra of both rings at specific time"""
    u1_probe = saved_data['u1_probe'].cpu().numpy()
    u2_probe = saved_data['u2_probe'].cpu().numpy()
    
    E1 = np.fft.ifftshift(np.fft.ifft(u1_probe[idx], norm='forward'))
    E2 = np.fft.ifftshift(np.fft.ifft(u2_probe[idx], norm='forward'))
    
    P1 = 10*np.log10(np.abs(E1)**2 + 1e-30) + 30
    P2 = 10*np.log10(np.abs(E2)**2 + 1e-30) + 30
    
    modes = np.arange(-(len(E1)-1)//2, len(E1)//2 + 1)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Ring 1 spectrum
    ax1.plot(modes, P1, linewidth=1.5, color='#21808d', label='Ring 1')
    ax1.set_ylabel('Power (dBm)', fontsize=11)
    ax1.set_title('Ring 1 Spectrum', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([P1.max()-80, P1.max()+5])
    ax1.legend()
    
    # Ring 2 spectrum
    ax2.plot(modes, P2, linewidth=1.5, color='#e68161', label='Ring 2')
    ax2.set_xlabel('Comb Line Number', fontsize=11)
    ax2.set_ylabel('Power (dBm)', fontsize=11)
    ax2.set_title('Ring 2 Spectrum', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([P2.max()-80, P2.max()+5])
    ax2.legend()
    
    detuning = saved_data['detuning'][idx].cpu().numpy()
    plt.suptitle(f'Spectral Comparison (Detuning: {detuning*1e-9/(2*np.pi):.2f} GHz)', 
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()

def plot_power_transfer():
    """Plot power transfer between rings"""
    u1_probe = saved_data['u1_probe'].cpu().numpy()
    u2_probe = saved_data['u2_probe'].cpu().numpy()
    detuning = saved_data['detuning'].cpu().numpy()
    
    # Total power in each ring
    P1_total = np.sum(np.abs(u1_probe)**2, axis=1)
    P2_total = np.sum(np.abs(u2_probe)**2, axis=1)
    P_total = P1_total + P2_total
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Power vs detuning
    det_GHz = detuning*1e-9/(2*np.pi)
    ax1.plot(det_GHz, P1_total*1e3, label='Ring 1', linewidth=2, color='#21808d')
    ax1.plot(det_GHz, P2_total*1e3, label='Ring 2', linewidth=2, color='#e68161')
    ax1.plot(det_GHz, P_total*1e3, label='Total', linewidth=2, 
             linestyle='--', color='#626c71')
    ax1.set_ylabel('Power (mW)', fontsize=11)
    ax1.set_title('Intracavity Power', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Power ratio
    ratio = P2_total / (P1_total + 1e-12)
    ax2.plot(det_GHz, ratio, linewidth=2, color='#21808d')
    ax2.set_xlabel('Detuning (GHz)', fontsize=11)
    ax2.set_ylabel('Power Ratio (P₂/P₁)', fontsize=11)
    ax2.set_title('Power Transfer Ratio', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_temporal_profiles(idx=-1):
    """Plot temporal profiles in both rings"""
    u1_probe = saved_data['u1_probe'].cpu().numpy()
    u2_probe = saved_data['u2_probe'].cpu().numpy()
    
    A1 = u1_probe[idx] / np.sqrt(len(u1_probe[idx]))
    A2 = u2_probe[idx] / np.sqrt(len(u2_probe[idx]))
    
    tr = tR.cpu().numpy()
    t = np.linspace(-tr/2, tr/2, len(A1))*1e12
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Ring 1 temporal
    ax1.plot(t, np.abs(A1)**2, linewidth=2, color='#21808d', label='|A₁|²')
    ax1.set_ylabel('Power (W)', fontsize=11)
    ax1.set_title('Ring 1: Temporal Profile', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Ring 2 temporal
    ax2.plot(t, np.abs(A2)**2, linewidth=2, color='#e68161', label='|A₂|²')
    ax2.set_xlabel('Time (ps)', fontsize=11)
    ax2.set_ylabel('Power (W)', fontsize=11)
    ax2.set_title('Ring 2: Temporal Profile', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

# %%
# Example usage of plotting functions
print("\nGenerating plots...")
plot_coupled_fields_evolution()
plot_spectra_comparison(idx=-1)
plot_power_transfer()
plot_temporal_profiles(idx=-1)