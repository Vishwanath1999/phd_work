# %%
import torch
import numpy as np
from scipy.io import loadmat, savemat
from scipy import constants as cts
from tqdm import tqdm
from ipywidgets import interact, widgets
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.ticker as ticker


device = 'cpu'#torch.device('cuda' if torch.cuda.is_available() else 'cpu')

disp = loadmat('disp_copy.mat')
res = loadmat('res_copy.mat')
sim = loadmat('sim_copy.mat')

desired_spectrum = loadmat('desired_spec2.mat')['Ewg'][0]
desired_spectrum_dBm = 10*np.log10(np.abs(desired_spectrum)**2)+30
desired_spectrum_dBm = np.clip(desired_spectrum_dBm, -60, None)

# Define desired_spectrum_tensor as a global variable for use in other functions
desired_spectrum_tensor = torch.tensor(desired_spectrum_dBm, dtype=torch.float64, device=device)

# df = pd.read_csv('./DKS_init.csv')
# df["DKS_init"] = df.DKS_init.apply(lambda val: complex(val.strip('()')))
# δω = df.det.values[0] * 2*np.pi
# DKS_init = df.DKS_init.values



sim['DKS_init'] = np.array([complex(x.strip()) for x in sim['DKS_init']])

# Convert the complex numbers to np.complex64
sim['DKS_init'] = sim['DKS_init'].astype(np.complex128)

tol = 1e-3
max_iter = 10
step_factor = 0.1

c0 = 299792458
h_bar = cts.hbar

# %%
# write a function that converts the dictorionary to a tensor. Drop the first 3 keys
def dict_to_tensor(dic):
    dic.pop('__header__', None)
    dic.pop('__version__', None)
    dic.pop('__globals__', None)
    dic.pop('dispfile', None)

    tensor_dic = dict()
    # convert the dictionary to a tensor
    for key in dic:
        if isinstance(dic[key], np.ndarray):
            if key == 'DKS_init':
                tensor_dic[key] = torch.tensor(dic[key], device=device, dtype=torch.complex64)
            else:
                dic[key] = np.where(dic[key] == 'None', None, dic[key])
                dic[key] = dic[key].astype(np.float64)
                tensor_dic[key] = torch.tensor(dic[key], device=device)
        else:
            tensor_dic[key] = dic[key]
    return tensor_dic

def tensor_to_dict(tensor):
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

disp_tensor = dict_to_tensor(disp)
res_tensor = dict_to_tensor(res)
sim.pop('domega', None)
sim['domega'] = np.array(['None',0])
sim_tensor = dict_to_tensor(sim)

# %%
disp_tensor['D1'] = disp_tensor['D1'][0]
disp_tensor['FSR'] = disp_tensor['FSR'][0]
disp_tensor['FSR_center'] = disp_tensor['FSR_center'][0]

for key in res_tensor:
    res_tensor[key] = res_tensor[key][0]

# if ndim>1 then squeeze the tensor
for key in sim_tensor:
    if sim_tensor[key].ndim > 1:
        sim_tensor[key] = sim_tensor[key][0]

# %%
ng = disp_tensor['ng']
R = res_tensor['R']
gamma = res_tensor['gamma']
L = 2*torch.pi*R
# print all shapes
# print('ng:', ng.shape)
# print('R:', R.shape)
# print('gamma:', gamma.shape)
# %%
Q0 = res_tensor['Qi']
Qc = res_tensor['Qc']
fpmp = sim_tensor['f_pmp']
Ppmp = sim_tensor['Pin']
# Ppmp = torch.tensor([0.16], dtype=torch.float64)
phi_pmp = sim_tensor['phi_pmp']
num_probe = sim_tensor['num_probe']
# num_probe = num_probe[0].cpu().numpy().astype(int)
num_probe = 5000
fcenter = sim_tensor['f_center']

DKSinit_real = torch.real(sim_tensor['DKS_init'])
if sim_tensor['DKS_init'].dtype == torch.complex128 or sim_tensor['DKS_init'].dtype == torch.complex64:
    DKSinit_imag = torch.imag(sim_tensor['DKS_init'])
else:
    DKSinit_imag = torch.zeros_like(DKSinit_real, device=device)

# %%
DKS_init = torch.complex(DKSinit_real, DKSinit_imag)

D1 = disp_tensor['D1']
FSR = D1/(2*torch.pi)
omega0 = 2*torch.pi*fpmp
omega_center = 2*torch.pi*fcenter

tR = 1/FSR
T = 1*tR
kext = (omega0[0]/Qc) * tR
k0 = (omega0[0]/Q0) * tR
alpha = k0+kext

un_norm_kappa = 2*torch.pi*(fpmp[0]/Q0 + fpmp[0]/Qc)

del_omega_init = 7*un_norm_kappa#sim_tensor['domega_init']
del_omega_end = -7*un_norm_kappa#sim_tensor['domega_end']
del_omega_stop = -7*un_norm_kappa#sim_tensor['domega_stop']
ind_sweep = sim_tensor['ind_pump_sweep'] 
t_end = sim_tensor['Tscan']
Dint = disp_tensor['Dint_new']#[0]

# %%
del_omega = sim['domega']
ind_pmp = [ii for ii in sim_tensor['ind_pmp'].int().cpu().numpy()]    
mu_sim = sim_tensor['mucenter']
mu = torch.arange(mu_sim[0], mu_sim[1]+1, device=device)
# find center of mu
mu0 = torch.where(mu == 0)[0][0].int().cpu().numpy()+1

d_omega = 2*torch.pi*FSR * torch.arange(mu_sim[0], mu_sim[-1]+1, device=device)
domega_pmp1 = 2*torch.pi*FSR * torch.arange(mu_sim[0]-ind_pmp[0]-1, mu_sim[-1]-ind_pmp[0], device=device)
omega1 = omega0[0] + domega_pmp1

Dint = Dint-Dint[mu0-1]

Ptot = 0#torch.zeros(1, device=device)
for ii in range(len(fpmp)):
    Ptot += Ppmp[ii]

dt = 1
t_end = t_end*tR
t_ramp = t_end

# %%
Nt = torch.tensor([1400000], dtype=torch.int)[0]#int(4e5)#torch.round(t_ramp/tR/dt)[0].int()
theta = torch.linspace(0, 2*torch.pi, len(mu), device=device)
del_omega_tot = torch.abs(del_omega_end)+torch.abs(del_omega_init)
del_omega_perc = -1*torch.sign(del_omega_end+del_omega_init)*(torch.abs(del_omega_end+del_omega_init)/2)/del_omega_tot
t_sim = torch.linspace(-t_ramp[0]/2 + del_omega_perc[0]*t_ramp[0], t_ramp[0]/2 + del_omega_perc[0]*t_ramp[0], Nt, device=device, dtype=torch.float64)

# # xx = torch.arange(0,Nt, device=device)
del_omega_all = torch.ones(len(fpmp), Nt, device=device, dtype=torch.float64)
# # ind_sweep = ind_sweep.cpu().numpy().int()
# if del_omega_init == del_omega_end:
#     Nt = 200000
#     del_omega_all = del_omega_init*torch.ones(len(fpmp), Nt, device=device, dtype=torch.float64)
# else:
#     xx = torch.arange(1,Nt+1, device=device)
#     for ii in ind_sweep.cpu().numpy().astype(int):
#         # del_omega_all[ii,:] = torch.linspace(del_omega_init[0], del_omega_end[0], Nt)
#         del_omega_all[ii,:] = del_omega_init + xx/Nt * (del_omega_end - del_omega_init)

# %%

@torch.jit.script
def Noise(h_bar: float, omega1: torch.Tensor, mu_len: int, device: torch.device) -> torch.Tensor:
    Ephoton = torch.tensor(h_bar, device=device) * omega1
    phase = 2 * torch.pi * torch.rand(mu_len,  device=device)
    array = torch.rand(mu_len,  device=device)
    Enoise = array * torch.sqrt(Ephoton / 2) * torch.exp(1j * phase) * mu_len
    return torch.fft.ifftshift(torch.fft.ifft(Enoise)).squeeze()

# fpmp = fpmp[0]
Ain = torch.zeros(len(fpmp), len(mu),dtype=torch.complex128, device=device)
Ein = torch.zeros(len(fpmp), len(mu),dtype=torch.complex128, device=device)

for ii in range(len(fpmp)):
    Ein[ii,int(mu0+ind_pmp[ii])] = torch.sqrt(Ppmp[ii])*len(mu)
    Ain[ii] = torch.fft.ifft(torch.fft.fftshift(Ein[ii],dim=0),dim=0)*torch.exp(-1j*phi_pmp[ii])

# u0 = torch.zeros(len(mu) , dtype=torch.complex128, device=device)
u0 = DKS_init

# %%
saved_data = dict()

saved_data['u_probe'] = torch.zeros(num_probe, len(u0), dtype=torch.complex128, device=device)
saved_data['driving_force'] = torch.zeros(num_probe, len(u0), dtype=torch.complex128, device=device)
saved_data['detuning'] = torch.zeros(num_probe, device=device)
saved_data['t_sim'] = torch.zeros(num_probe, device=device)
saved_data['kappa_ext'] = kext
saved_data['kappa_0'] = k0
saved_data['alpha'] = alpha

# dint2 = loadmat('dint.mat')['dint'][:,0]
# Dint_shift = torch.Tensor(dint2, device=device, dtype=torch.complex128)
Dint_shift = torch.fft.ifftshift(Dint)
# %%
def Fdrive(it, A_in, t_sim):
    Force = torch.zeros(len(mu), device=device, dtype=torch.complex128)
    for ii in range(len(fpmp)):
        if ii > 0:
            sigma = (2*del_omega_all[ii,it] + Dint[(mu0)+ind_pmp[ii]] - 0.5*del_omega_all[0,it])*t_sim[it]
        else:
            sigma=torch.zeros(1, device=device)
        Force = Force - 1j*A_in[ii]*torch.exp(1j*sigma)
    return Force + Noise(h_bar, omega1, int(len(mu)), device)

def SaveStatus_Callback(it, saved_data, u0, param, Fdrive_val, delta_theta, del_omega_all):
    # if it*num_probe/Nt > param['probe']:
    saved_data['u_probe'][param['probe'],:] = u0
    saved_data['detuning'][param['probe']] = del_omega_all[0,it]
    saved_data['t_sim'][param['probe']] = t_sim[it]
    saved_data['driving_force'][param['probe'],:] = Fdrive_val
    saved_data['delta_theta'][param['probe']] = delta_theta
    param['probe'] += 1
    return param

def SaveData(saved_data, name=None):
    saved_data_numpy = tensor_to_dict(saved_data)
    if name is None:
        savemat('SSFM_half_data.mat', saved_data_numpy)
    else:
        savemat(name, saved_data_numpy)

# %%
tol = 1e-3
max_iter = 10
success = False

L_h_prop = torch.zeros(len(mu), dtype=torch.complex128, device=device)
A_L_h_prop = torch.zeros(len(mu), dtype=torch.complex128, device=device)
NL_h_prop = torch.zeros(len(mu), dtype=torch.complex128, device=device)
A_h_prop = torch.zeros(len(mu), dtype=torch.complex128, device=device)
A_prop = torch.zeros(len(mu), dtype=torch.complex128, device=device)
NL_h_prop_1 = torch.zeros(len(mu), dtype=torch.complex128, device=device)
NL_prop = torch.zeros(len(mu), dtype=torch.complex128, device=device)
Force = torch.zeros(len(mu), dtype=torch.complex128, device=device)

# %%
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
    return (-alpha / 2) + 1j * (Dint_shift - del_omega_all[0, it]) * tR


# Function for the Nonlinear operator
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

# %%
@torch.jit.script
def ssfm_step(A0: torch.Tensor, it: int, alpha: torch.Tensor, Dint_shift: torch.Tensor,
              del_omega_all: torch.Tensor, tR: torch.Tensor, gamma: torch.Tensor, L: torch.Tensor, 
              max_iter: int, tol: float, dt: int, kext: torch.Tensor, Fdrive_val:torch.Tensor, A_prop:torch.Tensor) -> torch.Tensor:
    
    A0 = A0 + Fdrive_val * torch.sqrt(kext) * dt
    L_h_prop = torch.exp(FFT_Lin(it, alpha, Dint_shift, del_omega_all, tR) * dt / 2)
    A_L_h_prop = torch.fft.ifft(torch.fft.fft(A0) * L_h_prop)
    NL_h_prop_0 = NL(A0, gamma, L)
    A_h_prop = A0#.clone()
    A_prop = 0*A0.clone()

    for _ in range(max_iter):
        err=0
        NL_h_prop_1 = NL(A_h_prop, gamma, L)
        NL_prop = (NL_h_prop_0 + NL_h_prop_1) * dt / 2
        A_prop = torch.fft.ifft(torch.fft.fft(A_L_h_prop * torch.exp(NL_prop)) * L_h_prop)
        err = torch.linalg.vector_norm(A_prop - A_h_prop, ord=2, dim=0) / torch.linalg.vector_norm(A_h_prop, ord=2, dim=0)
        if err < tol:
            return A_prop
        A_h_prop = A_prop
    err = torch.linalg.vector_norm(A_prop - A_h_prop, ord=2, dim=0) / torch.linalg.vector_norm(A_h_prop, ord=2, dim=0)
    raise ValueError(f"Convergence Error: {err}")

def MainSolver(Nt, saved_data, u0, del_omega_all, A_in, t_sim, show_progress=False, ton='strong'):
    param = dict()
    param['tol'] = 1e-3
    param['max_iter'] = max_iter
    param['step_factor'] = 0.1
    param['probe'] = 0
    step_num = Nt.item()  # Initialize step_num to Nt
    flag = False

    iterator = tqdm(range(Nt), ncols=120) if show_progress else range(Nt)
    delta_theta = torch.tensor(0.0, dtype=torch.float64, device=device)  # Initialize δΘ
    # Thermal parameters (Bao et al. 2017)
    tau0 = 100e-9       # Thermal response time (100 ns)
    if ton == 'weak':
        xi = -1.2e4         # Thermo-optic coefficient (W⁻¹s⁻¹)
    elif ton == 'moderate':
        xi = -4.5e4         # Thermo-optic coefficient (W⁻¹s⁻¹)
    elif ton == 'strong':
        xi = -1.2e5         # Thermo-optic coefficient (W⁻¹s⁻¹)
    else:
        raise ValueError("Invalid value for 'ton'. Choose from 'weak', 'moderate', or 'strong'.")
    # print(f"Thermal response type: {ton}, xi: {xi}")
    del_omega_all_to = del_omega_all.clone()
    for it in iterator:
        del_omega_all_to[0, it] += delta_theta/(tR)
        Fdrive_val = Fdrive(it, A_in, t_sim)
        u0 = ssfm_step(u0, it, alpha, Dint_shift, del_omega_all_to, tR, gamma, L, 
                       max_iter, tol, dt, kext, Fdrive_val, A_prop)
        # Update thermal detuning
        P_avg = torch.mean(torch.abs(u0)**2)  # Compute average power
        d_delta_theta_dt = -delta_theta / tau0 + xi * P_avg
        delta_theta += (1 * tR) * d_delta_theta_dt  # Euler step
        # find u0 in dBm
        u0_dBm = 10 * torch.log10(torch.abs(u0)**2) + 30
        u0_dBm = torch.clip(u0_dBm, -60, None)  # Clip to avoid extreme values
        # calculate the correlation coefficient using torch.corrcoef
        if desired_spectrum is not None:
            corr_coef = torch.corrcoef(torch.stack((u0_dBm, desired_spectrum_tensor), dim=0))[0, 1]
            if corr_coef > 0.9 and flag == False:
                step_num = it
                flag=True
        if it*num_probe/Nt > param['probe']:
            param = SaveStatus_Callback(it, saved_data, u0, param, Fdrive_val, delta_theta, del_omega_all)
    return saved_data, step_num / (Nt.item()/2)
    # SaveData(saved_data,name)

# %%
def rescale_power(power, lower_limit=0.05, upper_limit=0.2, step_size=0.001):
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
# %%
def rescale_and_quantize(action, lower_limit=-2e6, upper_limit=2e6, step_size=1e4):
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
    return quantized_value*2*np.pi  # Convert to radians per second
# %%
def initialize_del_omega_all(fine, dwell_steps, Nt, del_omega_init, del_omega_end, rescale_and_quantize, device):
    """
    Initializes the del_omega_all array for the simulation, using rescale_and_quantize to quantize the detuning step.

    Args:
        fine (float): Fine adjustment in [-1, 1].
        dwell_steps (int): Number of steps to dwell before applying delta.
        Nt (int): Total number of time steps.
        del_omega_init (torch.Tensor): Initial detuning (shape: [num_pumps]).
        del_omega_end (torch.Tensor): End detuning (shape: [num_pumps]).
        rescale_and_quantize (callable): Function to compute delta_del_omega from fine.
        device (torch.device): Torch device.

    Returns:
        torch.Tensor: del_omega_all array of shape [num_pumps, Nt].
    """
    num_pumps = del_omega_init.shape[0]
    del_omega_all = torch.zeros(num_pumps, Nt, dtype=torch.float64, device=device)
    del_omega_all[:, 0] = del_omega_init

    # Use fine as the action for quantization
    delta_del_omega = rescale_and_quantize(fine, lower_limit=-5e6, upper_limit=5e6, step_size=1e4)
    delta_del_omega = torch.full((num_pumps,), delta_del_omega, dtype=torch.float64, device=device)

    min_del_omega = torch.minimum(del_omega_init, del_omega_end)
    max_del_omega = torch.maximum(del_omega_init, del_omega_end)

    for t in range(1, Nt):
        if dwell_steps > 0 and t % dwell_steps == 0:
            next_val = del_omega_all[:, t-1] + delta_del_omega
            next_val = torch.max(torch.min(next_val, max_del_omega), min_del_omega)
            del_omega_all[:, t] = next_val
        else:
            del_omega_all[:, t] = del_omega_all[:, t-1]

    # Optionally, ensure last value is exactly del_omega_end
    del_omega_all[:, -1] = del_omega_end
    return del_omega_all
# %%
# del_omega_all = initialize_del_omega_all(
#     fine=-0.2,    # Fine adjustment in [-1, 1]
#     dwell_steps=10,  # Number of steps to dwell before applying delta
#     Nt=Nt.item(),  # Total number of time steps
#     del_omega_init=del_omega_init,
#     del_omega_end=del_omega_end,
#     rescale_and_quantize=rescale_and_quantize,
#     device=device
# )
def reset_saved_data():
    saved_data = dict()

    saved_data['u_probe'] = torch.zeros(num_probe, len(u0), dtype=torch.complex128, device=device)
    saved_data['driving_force'] = torch.zeros(num_probe, len(u0), dtype=torch.complex128, device=device)
    saved_data['detuning'] = torch.zeros(num_probe, device=device)
    saved_data['t_sim'] = torch.zeros(num_probe, device=device)
    saved_data['kappa_ext'] = kext
    saved_data['kappa_0'] = k0
    saved_data['alpha'] = alpha
    saved_data['delta_theta'] = torch.zeros(num_probe, device=device)  # Initialize delta_theta
    return saved_data

# %%
# del_omega_all = initialize_del_omega_all(
#     fine=-0.002,    # Fine adjustment in [-1, 1]
#     dwell_steps=10,  # Number of steps to dwell before applying delta
#     Nt=Nt.item(),  # Total number of time steps
#     del_omega_init=del_omega_init,
#     del_omega_end=del_omega_end,
#     rescale_and_quantize=rescale_and_quantize,
#     device=device
#     )
# u0 = DKS_init
# # Reset saved_data for a new simulation run
# saved_data = reset_saved_data()
# # Run the main solver with the updated del_omega_all
# saved_data = MainSolver(Nt, saved_data, u0, del_omega_all)
# np_dict = tensor_to_dict(saved_data)
# Acav = np.sqrt(np_dict['alpha']/2)* np_dict['u_probe']*np.exp(1j*np.pi)/np.sqrt(np_dict['u_probe'].shape[1])
# Ecav = np.fft.fftshift(np.fft.fft(Acav, axis=1),axes=1)/np.sqrt(np_dict['u_probe'].shape[1])
# spec_dBm = 10 * np.log10(np.abs(Ecav)**2) + 30
# spec_dBm = np.clip(spec_dBm, -60, 10)  # Clip to avoid extreme values
# %%
def LLE(fine, dwell_steps, pump_power, start_factor, end_factor, progress_bar=False, ton='strong', save_dir=None, save_plots=False):
    """
    Main function to run the LLE simulation with given parameters.
    
    Args:
        fine (float): Fine adjustment in [-1, 1].
        dwell_steps (int): Number of steps to dwell before applying delta.
        pump_power (float): Pump power in W.
        progress_bar (bool): Whether to show a progress bar (default: False).
        ton (str): Thermal response type ('weak', 'moderate', 'strong').
        run_name (str): Name for saving the results (default: None).
        
    Returns:
        spec_dBm (np.ndarray): Spectrum in dBm.
        num_step (int): Number of steps taken to achieve soliton.
    """
    pump_power = rescale_power(pump_power, lower_limit=0.05, upper_limit=0.2, step_size=0.001)
    pump_power = torch.tensor(pump_power, dtype=torch.float64, device=device)
    del_omega_init = start_factor*un_norm_kappa
    del_omega_end = end_factor*un_norm_kappa
    del_omega_tot = torch.abs(del_omega_end)+torch.abs(del_omega_init)
    del_omega_perc = -1*torch.sign(del_omega_end+del_omega_init)*(torch.abs(del_omega_end+del_omega_init)/2)/del_omega_tot
    t_sim = torch.linspace(-t_ramp[0]/2 + del_omega_perc[0]*t_ramp[0], t_ramp[0]/2 + del_omega_perc[0]*t_ramp[0], Nt, device=device, dtype=torch.float64)
    del_omega_all = initialize_del_omega_all(
    fine=fine,    # Fine adjustment in [-1, 1]
    dwell_steps=dwell_steps,  # Number of steps to dwell before applying delta
    Nt=Nt.item(),  # Total number of time steps
    del_omega_init=del_omega_init,
    del_omega_end=del_omega_end,
    rescale_and_quantize=rescale_and_quantize,
    device=device
    )
    u0 = DKS_init
    # Reset saved_data for a new simulation run
    saved_data = reset_saved_data()
    Ain = torch.zeros(len(fpmp), len(mu),dtype=torch.complex128, device=device)
    Ein = torch.zeros(len(fpmp), len(mu),dtype=torch.complex128, device=device)

    for ii in range(len(fpmp)):
        Ein[ii,int(mu0+ind_pmp[ii])] = torch.sqrt(pump_power)*len(mu)
        Ain[ii] = torch.fft.ifft(torch.fft.fftshift(Ein[ii],dim=0),dim=0)*torch.exp(-1j*phi_pmp[ii])
    # Run the main solver with the updated del_omega_all
    saved_data, num_step = MainSolver(Nt, saved_data, u0, del_omega_all, Ain, t_sim, show_progress=progress_bar, ton=ton)
    np_dict = tensor_to_dict(saved_data)
    Acav = np.sqrt(np_dict['alpha']/2)* np_dict['u_probe']*np.exp(1j*np.pi)/np.sqrt(np_dict['u_probe'].shape[1])
    Ecav = np.fft.fftshift(np.fft.fft(Acav, axis=1),axes=1)/np.sqrt(np_dict['u_probe'].shape[1])

    wg = np_dict['driving_force'] * np.sqrt(1-np_dict['kappa_ext'])
    cav = np.sqrt(np_dict['kappa_ext'])*np_dict['u_probe']*np.exp(1j*np.pi)

    Awg = (wg+cav)/np.sqrt(np_dict['u_probe'].shape[1])
    Ewg = np.fft.fftshift(np.fft.fft(Awg, axis=1),axes=1)/np.sqrt(np_dict['u_probe'].shape[1])
    Acav_abs = 1e3*np.abs(Acav.T)**2
    spec_dBm = 10 * np.log10(np.abs(Ewg)**2) + 30
    spec_dBm = np.clip(spec_dBm, -60, None)  # Clip to avoid extreme values
    if progress_bar:

        plt.figure(figsize=(14, 4))
        plt.imshow(Acav_abs, aspect='auto', cmap='jet', extent=[0,Nt*tR.item()*1e6, -tR.item()*1e12/2, tR.item()*1e12/2], origin='lower')
        cbar = plt.colorbar()
        cbar.ax.tick_params(labelsize=20)
        cbar.set_label(r'Power $(mW)$', fontsize=20)
        formatter = ticker.ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((-1, 1))
        # plt.gca().xaxis.set_major_formatter(formatter)
        # plt.title(ton + ' themal effect', fontsize=18)
        plt.xlabel(r'Time $(\mu s)$', fontsize=20)
        plt.ylabel(r'Fast time (ps)', fontsize=20)
        plt.tight_layout()
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        # plt.savefig('./GA_results_corr_coeff/field_amplitude_ton_' + ton + '.png')
        # plt.savefig('./GA_results_corr_coeff/field_amplitude_ton_' + ton + '.svg', format='svg')
        # save images in save_dir
        if save_dir is not None:
            plt.savefig(os.path.join(save_dir, 'field_amplitude_ton_' + ton + '.png'))
            plt.savefig(os.path.join(save_dir, 'field_amplitude_ton_' + ton + '.svg'), format='svg')
        plt.show()
        plt.close()

        plt.figure(figsize=(10,6))
        x = np.linspace(0, Nt*tR.item()*1e6, len(np_dict['detuning']))
        plt.plot(x,np_dict['delta_theta']*FSR.item()/(2*torch.pi*1e9), linewidth=1.5, label=r'$f _{\Theta}$')
        plt.plot(x,np_dict['detuning']/(2*torch.pi*1e9), linewidth=1.5, label=r'$\Delta f_{pmp}$')
        plt.plot(x,(np_dict['delta_theta']*FSR.item() + np_dict['detuning'])/(2*torch.pi*1e9), linewidth=1.5, label=r'$\Delta f_{eff}$')
        plt.xlabel(r'Time $(\mu s)$', fontsize=20)
        plt.ylabel('Freq. (GHz)', fontsize=20)
        plt.xticks(fontsize=20)
        plt.legend(fontsize=18)
        # plt.title(ton + ' thermal effect', fontsize=18)
        plt.yticks(fontsize=20)
        # plt.gca().xaxis.set_major_formatter(formatter)
        plt.grid(visible=True, which='both', axis='both', linestyle='--', linewidth=0.5)
        # plt.savefig('./GA_results_corr_coeff/delta_theta_ton_' + ton + '.png')
        # plt.savefig('./GA_results_corr_coeff/delta_theta_ton_' + ton + '.svg', format='svg')
        if save_dir is not None:
            plt.savefig(os.path.join(save_dir, 'delta_theta_ton_' + ton + '.png'))
            plt.savefig(os.path.join(save_dir, 'delta_theta_ton_' + ton + '.svg'), format='svg')
        plt.show()
        plt.close()

        plt.figure(figsize=(10,6))
        x = np.linspace(0, Nt*tR.item()*1e6, len(np_dict['detuning']))
        plt.plot(x,np_dict['delta_theta']*FSR.item()/un_norm_kappa.item(), linewidth=1.5, label=r'$f _{\Theta}$')
        plt.plot(x,np_dict['detuning']/un_norm_kappa.item(), linewidth=1.5, label=r'$\Delta f_{pmp}$')
        plt.plot(x,(np_dict['delta_theta']*FSR.item() + np_dict['detuning'])/un_norm_kappa.item(), linewidth=1.5, label=r'$\Delta f_{eff}$')
        plt.xlabel(r'Time $(\mu s)$', fontsize=20)
        plt.ylabel(r'Freq. ($\times \kappa$)', fontsize=20)
        plt.xticks(fontsize=20)
        plt.legend(fontsize=18)
        # plt.title(ton + ' thermal effect', fontsize=18)
        plt.yticks(fontsize=20)
        # plt.gca().xaxis.set_major_formatter(formatter)
        plt.grid(visible=True, which='both', axis='both', linestyle='--', linewidth=0.5)
        # plt.savefig('./GA_results_corr_coeff/delta_theta_ton_' + ton + '.png')
        # plt.savefig('./GA_results_corr_coeff/delta_theta_ton_' + ton + '.svg', format='svg')
        if save_dir is not None:
            plt.savefig(os.path.join(save_dir, 'delta_theta_ton_kappa_' + ton + '.png'))
            plt.savefig(os.path.join(save_dir, 'delta_theta_ton_kappa_' + ton + '.svg'), format='svg')
        plt.show()
        plt.close()

        # plot detuning
        # plt.figure(figsize=(10,6))
        # x = np.linspace(0, Nt*tR.item()*1e6, len(np_dict['detuning']))
        # plt.plot(x,np_dict['detuning']/(2*torch.pi), linewidth=1.5)
        # plt.xlabel(r'Time $(\mu s)$', fontsize=20)
        # plt.ylabel(r'$\Delta f_{pmp}$ (GHz)', fontsize=20)
        # plt.xticks(fontsize=20)
        # plt.yticks(fontsize=20)
        # plt.gca().xaxis.set_major_formatter(formatter)
        # plt.grid(visible=True, which='both', axis='both', linestyle='--', linewidth=0.5)
        # # plt.savefig('./GA_results_corr_coeff/detuning_ton_' + ton + '.png')
        # # plt.savefig('./GA_results_corr_coeff/detuning_ton_' + ton + '.svg', format='svg')
        # if save_dir is not None:
        #     plt.savefig(os.path.join(save_dir, 'detuning_ton_' + ton + '.png'))
        #     plt.savefig(os.path.join(save_dir, 'detuning_ton_' + ton + '.svg'), format='svg')
        # plt.show()
        # plt.close()

        # plot pcav
        plt.figure(figsize=(10,6))
        plt.plot(x,np.sum(Acav_abs, axis=0), linewidth=1.5)
        plt.xlabel(r'Time $(\mu s)$', fontsize=20)
        plt.ylabel(r'$P_{cav}$ (mW)', fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        # plt.gca().xaxis.set_major_formatter(formatter)
        plt.grid(visible=True, which='both', axis='both', linestyle='--', linewidth=0.5)
        # plt.savefig('./GA_results_corr_coeff/pcav_ton_' + ton + '.png')
        # plt.savefig('./GA_results_corr_coeff/pcav_ton_' + ton + '.svg', format='svg')
        if save_dir is not None:
            plt.savefig(os.path.join(save_dir, 'pcav_ton_' + ton + '.png'))
            plt.savefig(os.path.join(save_dir, 'pcav_ton_' + ton + '.svg'), format='svg')
        plt.show()
        plt.close()
    return spec_dBm[-1], num_step
# %%
def fitness(spec):
    """
    Calculate the fitness of the spectrum.
    """
    # desired_spectrum = loadmat('desired_spec2.mat')['Ewg'][0]
    # desired_spectrum_dBm = 10*np.log10(np.abs(desired_spectrum)**2)+30
    # desired_spectrum_dBm = np.clip(desired_spectrum_dBm, -60, None)  # Clip to avoid extreme values
    mse = np.mean((spec - desired_spectrum_dBm)**2)
    return mse

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

# write a fitness function using correlation coefficient
def fitness_correlation(spec, num_step):
    """
    Calculate the fitness of the spectrum using correlation coefficient.
    """
    # desired_spectrum = loadmat('desired_spec2.mat')['Ewg'][0]
    # desired_spectrum_dBm = 10*np.log10(np.abs(desired_spectrum)**2)+30
    # desired_spectrum_dBm = np.clip(desired_spectrum_dBm, -60, None)  # Clip to avoid extreme values
    correlation = np.corrcoef(spec, desired_spectrum_dBm)[0, 1]
    # center_idx = len(spec) // 2
    # mask = y
    return 4*(1-correlation) #+ num_step

def objective(X):
    """
    Objective function for the genetic algorithm.
    X is expected to be a 1D array with two elements del_omega and dewll_steps.
    """
    fine = X[0]  # GA passes an array
    dwell_steps = 100#int(X[1])  # Convert to integer for dwell steps
    spec, num_step = LLE(fine=fine, dwell_steps=dwell_steps, pump_power=X[1],\
                          start_factor=X[2], end_factor=X[3], ton='strong')
    return fitness_correlation(spec, num_step)  # Return the fitness value
# %%
from GA_lib2 import geneticalgorithm
import os

plt.style.use('physrev.mplstyle')

# desired_spectrum = loadmat('desired_spec2.mat')['Ewg'][0]
# desired_spectrum_dBm = 10*np.log10(np.abs(desired_spectrum)**2)+30
# desired_spectrum_dBm = np.clip(desired_spectrum_dBm, -60, None)

# Define desired_spectrum_tensor as a global variable for use in other functions
desired_spectrum_tensor = torch.tensor(desired_spectrum_dBm, dtype=torch.float64, device=device)

run_name = 'GA_LLE_optim_corr_coeff_num_steps_strong_v2'

algorithm_param = {
    'max_num_iteration': 40,
    'population_size': 40,
    'mutation_probability': 0.25,
    'elit_ratio': 0.01,
    'crossover_probability': 0.5,
    'parents_portion': 0.3,
    'crossover_type': 'uniform',
    'max_iteration_without_improv': 20,
}

if __name__ == "__main__":
    varbound = np.array([ [-1, 1],  [-1, 1], [3, 14], [-14, -3] ])  # Adjust range as needed
    seed = 42  # Set a seed for reproducibility
    model = geneticalgorithm(
        function=objective,
        dimension=4,
        variable_boundaries=varbound,
        parallel=True,
        n_processes=15,
        progress_bar=True,
        algorithm_parameters=algorithm_param,
        random_seed=seed,
        fitness_threshold=None,  # Set a fitness threshold for early stopping
    )
    save_dir = os.path.join(os.getcwd(), run_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    best_var, best_fit = model.run(save_dir)
    
    print("Best Solution:", best_var)
    print("Best Fitness:", best_fit)
    save_path = os.path.join(save_dir, 'best_solution.txt')
    with open(save_path, 'w') as f:
        f.write(f"Best Solution: {best_var}\n")
        f.write(f"Best Fitness: {best_fit}\n")
    
    # run the simulation with best params
    # -0.77720861  0.82866072
    fine = best_var[0]
    dwell_steps = 100  # Convert to integer for dwell steps
    domega = initialize_del_omega_all(fine, dwell_steps, Nt.item(), del_omega_init, del_omega_end, rescale_and_quantize, device).cpu().numpy()
    domega = domega[0]/(2*np.pi)  # Convert to GHz for plotting
    x = np.linspace(0, Nt*tR.item()*1e6, len(domega))
    plt.figure(figsize=(10,6))
    plt.plot(x,domega*1e-9, linewidth=1.5)
    plt.xlabel(r'Time $(\mu s)$', fontsize=20)
    plt.ylabel('Detuning (GHz)', fontsize=20)
    plt.title('Pump Detuning', fontsize=20, fontweight='bold')
    plt.grid(visible=True, which='both', axis='both', linestyle='--', linewidth=0.5)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.tight_layout()
    # make xticks in scientific notation
    # plt.gca().xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    # plt.gca().xaxis.get_major_formatter().set_scientific(True)
    plt.savefig(os.path.join(save_dir, 'detuning_vs_tuning_steps.png'))
    plt.savefig(os.path.join(save_dir, 'detuning_vs_tuning_steps.svg'), format='svg')
    plt.show()
    pump_power = best_var[1]  # Assuming the third parameter is the pump power
    
    for ton_case in ['weak', 'moderate', 'strong']:
        print(f"Running simulation with ton='{ton_case}', fine={rescale_and_quantize(best_var[0])}, dwell_steps={dwell_steps}, pump_power={rescale_power(best_var[1])}")
        spec = LLE(fine, dwell_steps, pump_power, progress_bar=True, save_dir=save_dir, ton=ton_case)
        desired_spectrum = loadmat('desired_spec2.mat')['Ewg'][0]
        desired_spectrum_dBm = 10*np.log10(np.abs(desired_spectrum)**2)+30
        desired_spectrum_dBm = np.clip(desired_spectrum_dBm, -60, None)
        plt.figure(figsize=(14, 4))
        plt.vlines(np.arange(-220,221,1),-60*np.ones(441), spec, label='Optimized Spectrum', color='blue')
        plt.vlines(np.arange(-220,221,1),-60*np.ones(441), desired_spectrum_dBm, label='Desired Spectrum', color='red', alpha=0.5)
        # plt.title(ton_case + ' thermal effect', fontsize=18, fontweight='bold')
        plt.xlabel('Mode no.', fontsize=16)
        plt.ylabel('Power (dBm)', fontsize=16)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.ylim(-70, 30)
        plt.xlim(-180,180)
        plt.legend(fontsize=16)
        plt.grid()
        plt.tight_layout()
        # plt.savefig(f'./GA_results_corr_coeff/optimized_spectrum_{ton_case}.png', dpi=300)
        # plt.savefig(f'./GA_results_corr_coeff/optimized_spectrum_{ton_case}.svg', dpi=300, format='svg')
        plt.savefig(os.path.join(save_dir, f'optimized_spectrum_{ton_case}.png'))
        plt.savefig(os.path.join(save_dir, f'optimized_spectrum_{ton_case}.svg'), format='svg')
        plt.show()
# %%
plt.style.use('physrev.mplstyle')
run_name = 'GA_LLE_optim_corr_coeff_num_steps_strong_v2'
save_dir = os.path.join(os.getcwd(), run_name)
os.makedirs(save_dir, exist_ok=True)
# run the simulation with best params
# load the best solution from file
# with open(os.path.join(save_dir, 'best_solution.txt'), 'r') as f:
#     line = f.readline()
#     # Extract the part inside the brackets
#     import re
#     numbers = re.findall(r"[-+]?\d*\.\d+|\d+", line)
#     best_var = np.array([float(num) for num in numbers])
# [  0.04546566   0.7921826    3.27961039 -12.8131 ]
fine = 0.04546566#best_var[0]
dwell_steps = 100
pump_power = 0.7921826#best_var[1]  # Assuming the third parameter is the pump power
start_factor = 3.27961039#best_var[2]
end_factor = -12.8131#best_var[3]
del_omega_init = start_factor*un_norm_kappa
del_omega_end = end_factor*un_norm_kappa
domega = initialize_del_omega_all(fine, dwell_steps, Nt.item(), del_omega_init, del_omega_end, rescale_and_quantize, device).cpu().numpy()
domega = domega[0]/(2*np.pi)  # Convert to GHz for plotting
x = np.linspace(0, Nt*tR.item()*1e6, len(domega))
plt.figure(figsize=(10,6))
plt.plot(x,domega*1e-9, linewidth=1.5)
plt.xlabel(r'Time $(\mu s)$', fontsize=20)
plt.ylabel('Detuning (GHz)', fontsize=20)
plt.title('Pump Detuning', fontsize=20, fontweight='bold')
plt.grid(visible=True, which='both', axis='both', linestyle='--', linewidth=0.5)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.tight_layout()
# make xticks in scientific notation
# plt.gca().xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
# plt.gca().xaxis.get_major_formatter().set_scientific(True)
plt.savefig(os.path.join(save_dir, 'detuning_vs_tuning_steps.png'))
plt.savefig(os.path.join(save_dir, 'detuning_vs_tuning_steps.svg'), format='svg')
plt.show()

print(f"Running simulation with fine={rescale_and_quantize(fine)/(2*np.pi*1e9)} GHz, dwell_steps={dwell_steps}, pump_power={rescale_power(pump_power)} W")
# spec = LLE(fine, dwell_steps, pump_power, progress_bar=True)
# plot the spectrum against the desired spectrum
desired_spectrum = loadmat('desired_spec2.mat')['Ewg'][0]
desired_spectrum_dBm = 10*np.log10(np.abs(desired_spectrum)**2)+30
desired_spectrum_dBm = np.clip(desired_spectrum_dBm, -60, None)  # Clip to avoid extreme values

for ton_case in ['weak', 'moderate','strong']:#'weak', 'moderate', 
    print(f"Running simulation with ton='{ton_case}', fine={rescale_and_quantize(fine)/(2*np.pi*1e9)} GHz, dwell_steps={dwell_steps}, pump_power={rescale_power(pump_power)} W")
    spec, num_steps = LLE(fine, dwell_steps, pump_power, start_factor, end_factor, progress_bar=True, ton=ton_case, save_dir=save_dir)
    print(f"Number of steps taken to achieve soliton: {num_steps}")
    plt.figure(figsize=(14, 4))
    plt.vlines(np.arange(-220,221,1),-60*np.ones(441), spec, label='Optimized Spectrum', color='blue')
    plt.vlines(np.arange(-220,221,1),-60*np.ones(441), desired_spectrum_dBm, label='Desired Spectrum', color='red', alpha=0.5)
    # plt.title('Optimized Spectrum vs Desired Spectrum', fontsize=18, fontweight='bold')
    plt.xlabel('Mode no.', fontsize=20)
    plt.ylabel('Power (dBm)', fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.ylim(-70, 30)
    plt.xlim(-180,180)
    plt.legend(fontsize=14, loc='lower center')
    plt.grid()
    plt.tight_layout()
    # plt.savefig(f'./GA_results_corr_coeff/optimized_spectrum_{ton_case}.png')
    # plt.savefig(f'./GA_results_corr_coeff/optimized_spectrum_{ton_case}.svg', format='svg')
    plt.savefig(os.path.join(save_dir, f'optimized_spectrum_{ton_case}.png'))
    plt.savefig(os.path.join(save_dir, f'optimized_spectrum_{ton_case}.svg'), format='svg')
    plt.show()
    # print the mse 
    mse = fitness(spec)
    print(f"Mean Squared Error (MSE) for {ton_case} thermal effect: {mse:.4f}")

# %%
'''
# import the population fitness mat file
# population_fitness = loadmat('./GA_results/population_fitness.mat')['fitness']
# def plot_population_fitness_from_mat(population_fitness, save_path=None):
#     """
#     Plot the fitness of the entire population per generation from a loaded .mat file,
#     highlighting the best solution in each generation.
#     """
#     fitness_matrix = np.array(population_fitness)  # shape: (generations, population_size)
#     generations = np.arange(fitness_matrix.shape[0])
#     plt.figure(figsize=(8, 5))
#     # Plot all individuals
#     plt.plot(generations[:, None], fitness_matrix, '*', color='black', markersize=3, alpha=0.7)
#     # Highlight best
#     best_fitness = np.min(fitness_matrix, axis=1)
#     plt.plot(generations, best_fitness, 'ro-', linewidth=2, label='Best Fitness', markersize=3)
#     plt.xlabel('Generation', fontsize=20)
#     plt.ylabel('Fitness '+ r'$mse$', fontsize=20)
#     plt.xticks(fontsize=16)
#     plt.yticks(fontsize=16)
#     plt.yscale('log')
#     plt.title('Population Fitness per Generation', fontsize=18, fontweight='bold')
#     plt.legend()
#     plt.grid(visible=True, which='both', axis='both', linestyle='--', linewidth=0.5)
#     plt.tight_layout()
#     fig_name = 'population_fitness_curve.png'
#     if save_path is not None:
#         save_path = save_path if save_path.endswith('.png') else f"{save_path}/{fig_name}"
#     else:
#         save_path = fig_name
#     print(f"Saving population fitness plot to {save_path}")
#     plt.savefig(save_path, dpi=300)
#     plt.show()

# def plot_fitness_curve_from_mat(population_fitness, save_path=None):
#     """
#     Plot and optionally save the best fitness (minimum per generation) vs generations from a loaded .mat file.
#     """
#     fitness_matrix = np.array(population_fitness)
#     best_fitness = np.min(fitness_matrix, axis=1)
#     plt.figure(figsize=(8, 5))
#     plt.plot(best_fitness, label='Best Fitness', linewidth=2)
#     plt.xlabel('Generation', fontsize=20)
#     plt.ylabel('Best Fitness '+r'$mse$', fontsize=20)
#     plt.title('Fitness Curve', fontsize=18, fontweight='bold')
#     plt.xticks(fontsize=16)
#     plt.yticks(fontsize=16)
#     plt.yscale('log')
#     plt.grid(visible=True, which='both', axis='both', linestyle='--', linewidth=0.5)
#     plt.legend()
#     plt.tight_layout()
#     fig_name = 'fitness_curve.png'
#     if save_path is not None:
#         save_path = save_path if save_path.endswith('.png') else f"{save_path}/{fig_name}"
#     else:
#         save_path = fig_name
#     print(f"Saving fitness curve plot to {save_path}")
#     plt.savefig(save_path, dpi=300)
#     plt.show()
# # %%
# # Example usage:
# save_path = os.path.join(os.getcwd(), 'GA_results')
# plot_population_fitness_from_mat(population_fitness, save_path=save_path)
# plot_fitness_curve_from_mat(population_fitness, save_path=save_path)
# '''
# %%
