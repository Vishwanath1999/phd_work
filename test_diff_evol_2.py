import torch
import numpy as np
from scipy.io import loadmat, savemat
from scipy import constants as cts
from tqdm import tqdm
from ipywidgets import interact, widgets
import matplotlib.pyplot as plt
import torch
from torch import autograd as ag


device = 'cpu'#torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('using device:', device)

disp = loadmat('disp.mat')
res = loadmat('res.mat')
sim = loadmat('sim.mat')

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
phi_pmp = sim_tensor['phi_pmp']
num_probe = sim_tensor['num_probe']
num_probe = num_probe.cpu().numpy().astype(int)[0]
fcenter = sim_tensor['f_center']

DKSinit_real = torch.real(sim_tensor['DKS_init'])
if sim_tensor['DKS_init'].dtype == 'complex128':
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

del_omega_init = sim_tensor['domega_init']
del_omega_end = sim_tensor['domega_end']
del_omega_stop = sim_tensor['domega_stop']
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
Nt = torch.round(t_ramp/tR/dt)[0].int()
theta = torch.linspace(0, 2*torch.pi, len(mu), device=device)
del_omega_tot = torch.abs(del_omega_end)+torch.abs(del_omega_init)
del_omega_perc = -1*torch.sign(del_omega_end+del_omega_init)*(torch.abs(del_omega_end+del_omega_init)/2)/del_omega_tot
t_sim = torch.linspace(-t_ramp[0]/2 + del_omega_perc[0]*t_ramp[0], t_ramp[0]/2 + del_omega_perc[0]*t_ramp[0], Nt, device=device, dtype=torch.float64)

del_omega_all = torch.ones(len(fpmp), Nt, device=device, dtype=torch.float64)
xx = torch.arange(1,Nt+1, device=device)
for ii in ind_sweep.cpu().numpy().astype(int):
    del_omega_all[ii,:] = del_omega_init + xx/Nt * (del_omega_end - del_omega_init)

# %%
def Noise():
    Ephoton = torch.tensor(h_bar)*omega1
    phase = 2*torch.pi*torch.rand(len(mu),1, device=device)
    array = torch.rand(len(mu),1, device=device)
    Enoise = array*torch.sqrt(Ephoton/2)*torch.exp(1j*phase)*len(mu)
    return torch.fft.ifftshift(torch.fft.ifft(Enoise))
# fpmp = fpmp[0]
Ain = torch.zeros(len(fpmp), len(mu),dtype=torch.complex128, device=device)
Ein = torch.zeros(len(fpmp), len(mu),dtype=torch.complex128, device=device)

for ii in range(len(fpmp)):
    Ein[ii,int(mu0+ind_pmp[ii])] = torch.sqrt(Ppmp[ii])*len(mu)
    Ain[ii] = torch.fft.ifft(torch.fft.fftshift(Ein[ii]))*torch.exp(1j*phi_pmp[ii])

# %%
def get_Ain(Ppmp, phi_pmp):
    Ain = torch.zeros(len(fpmp), len(mu),dtype=torch.complex128, device=device)
    Ein = torch.zeros(len(fpmp), len(mu),dtype=torch.complex128, device=device)

    for ii in range(len(fpmp)):
        Ein[ii,int(mu0+ind_pmp[ii])] = torch.sqrt(Ppmp[ii])*len(mu)
        Ain[ii] = torch.fft.ifft(torch.fft.fftshift(Ein[ii]))*torch.exp(1j*phi_pmp[ii])
    return Ain, Ein

Dint_shift = torch.fft.ifftshift(Dint)
# %%
def Fdrive_opt(it, Ain):
    Force = 0*torch.exp(1j*theta)
    for ii in range(len(fpmp)):
        if ii > 0:
            sigma = (2*del_omega_all[ii,it] + Dint[mu0+ind_pmp[ii]] - 0.5*del_omega_all[0,it])*t_sim[it]
        else:
            sigma=torch.zeros(1, device=device)
        Force = Force - 1j*Ain[ii]*torch.exp(1j*sigma)
    return Force #+ Noise()[0]

def SaveStatus_Callback(saved_data, u0, param):
    # with torch.no_grad():
    saved_data['u_probe'][param['probe'],:] = u0
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
    raise RuntimeError(f"Convergence Error")


def MainSolver(Nt, saved_data, u0, name=None, Ain=None):
    param = dict()
    param['tol'] = 1e-3
    param['max_iter'] = max_iter
    param['step_factor'] = 0.1
    param['probe'] = 0


    for it in range(Nt):
    # for it in tqdm(range(Nt),ncols=120):
        Fdrive_val = Fdrive_opt(it, Ain)
        u0 = ssfm_step(u0, it, alpha, Dint_shift, del_omega_all, tR, gamma, L, 
                       max_iter, tol, dt, kext, Fdrive_val, A_prop)
        if it*num_probe/Nt > param['probe']:
            param = SaveStatus_Callback(saved_data, u0, param)
    return saved_data

# %%
name = 'SSFM_half_data.mat'

# %%
def process_data(saved_data, idx=5000):
    u_probe = saved_data['u_probe']
    alpha = saved_data['alpha']
    Acav = torch.sqrt(alpha/2)*u_probe*np.exp(1j*np.pi)/np.sqrt(u_probe.shape[1])
    Ecav = torch.fft.fftshift(torch.fft.fft(Acav, dim=1),dim=1)/np.sqrt(u_probe.shape[1])
    pred_spec = 10 * torch.log10(torch.abs(Ecav[idx])**2) + 30
    return pred_spec

# %%
def compute_loss(predicted_spectrum, ref_spectrum):
    # return torch.mean((ref_spectrum-predicted_spectrum)**2)
    return np.linalg.norm(predicted_spectrum - ref_spectrum, ord=2, axis=0)/ np.linalg.norm(ref_spectrum, ord=2, axis=0)

# %% get the reference spectrum
ref_spectrum = loadmat('ref_spec.mat')['ref_spectrum'][0]
# %%
Ppmp = np.array([90e-3])
phi_pmp = np.array([np.pi])

# def fitness_function(params):
#     print(params.shape)
#     Ppmp = params[0,:]
#     phi_pmp = params[1,:]
#     Ain, _ = get_Ain(torch.tensor([Ppmp]), torch.tensor([phi_pmp]))
#     u0 = DKS_init
#     saved_data = dict()
#     saved_data['u_probe'] = torch.zeros(num_probe, len(u0), dtype=torch.complex128, device=device)
#     saved_data['alpha'] = alpha
#     saved_data = MainSolver(Nt, saved_data, u0, name, Ain)
#     pred_spec = process_data(saved_data).cpu().numpy()
#     loss = compute_loss(pred_spec, ref_spectrum)
#     return loss
# %%
import multiprocessing

# Assuming required imports like MainSolver, process_data, get_Ain, etc. are already done.

def parallel_evaluation(Ppmp, phi_pmp):
    """
    This function calculates the fitness for a single solution vector.
    It runs the necessary functions like get_Ain, MainSolver, process_data, and compute_loss for each solution.
    """
    # Prepare the initial conditions for each solution
    Ain, _ = get_Ain(torch.tensor([Ppmp]), torch.tensor([phi_pmp]))
    u0 = DKS_init
    saved_data = dict()
    saved_data['u_probe'] = torch.zeros(num_probe, len(u0), dtype=torch.complex128, device=device)
    saved_data['alpha'] = alpha

    # Solve using MainSolver
    saved_data = MainSolver(Nt, saved_data, u0, name, Ain)
    pred_spec = process_data(saved_data).cpu().numpy()

    # Compute the loss (fitness)
    loss = compute_loss(pred_spec, ref_spectrum)
    return loss

def fitness_function(params):
    """
    This function handles the vectorized fitness evaluation using multiprocessing.
    It calls parallel_evaluation for each solution vector.
    """
    # Number of solutions to evaluate
    # num_solutions = params.shape[1]  # S: Population size (number of solutions)

    # Prepare the parameter values for parallel execution
    Ppmp_batch = params[0, :]
    phi_pmp_batch = params[1, :]

    # Use multiprocessing to evaluate all solution vectors in parallel
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        # Map the parallel_evaluation function to each solution vector (Ppmp, phi_pmp)
        results = pool.starmap(parallel_evaluation, zip(Ppmp_batch, phi_pmp_batch))

    # Return the loss values
    return np.array(results)
# %%
# bounds
bounds = [(0, 500e-3), (0, 2*np.pi)]
# %%
# Function to initialize tqdm progress bar
def create_progress_bar():
    return tqdm(total=100, desc='Optimization Progress', bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [elapsed: {elapsed} remaining: {remaining}]')

# Callback function for tqdm progress tracking
def callback(progress_bar):
    def inner_callback(xk, convergence):
        # Update progress bar based on convergence (number of iterations)
        progress_bar.update(1)  # Update by 1 step per iteration
        # Print best MSE (fitness) after each iteration
        print(f"Best MSE: {convergence}")
        print(f"Best Parameters: {xk}")
    return inner_callback

# def callback():
#     def inner_callback(xk, convergence):
#         # Update progress bar based on convergence (number of iterations)
#         # Print best MSE (fitness) after each iteration
#         print(f"Best MSE: {convergence}")
#         print(f"Best Parameters: {xk}")
#     return inner_callback

# Set up tqdm progress bar
progress_bar = create_progress_bar()

# # Define the callback function for tqdm
callback_fn = callback(progress_bar)
# %%
from scipy.optimize import differential_evolution
# Run the Differential Evolution algorithm
result = differential_evolution(
    fitness_function,
    bounds,
    strategy='best1bin',
    maxiter=2,
    popsize=15,
    tol=1e-6,
    mutation=(0.5, 1),
    recombination=0.5,
    seed=42,
    vectorized=True,
    callback=callback_fn
)

# Extract the results
optimal_params = result.x
optimal_fitness = result.fun
print(f"Optimal Parameters: {optimal_params}")
print(f"Optimal Fitness (MSE): {optimal_fitness}")
# %%
