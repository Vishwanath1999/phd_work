import torch
import numpy as np
from scipy.io import loadmat, savemat
from scipy import constants as cts
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.optimize import minimize

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


    # for it in range(Nt):
    for it in tqdm(range(Nt),ncols=120):
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
# Define the RBF Kernel for Gaussian Process
def rbf_kernel(x1, x2, length_scale=1.0, variance=1.0):
    dist = torch.cdist(x1, x2, p=2) ** 2
    return variance * torch.exp(-0.5 * dist / length_scale**2)

class GaussianProcess:
    def __init__(self):
        self.x_train = None
        self.y_train = None
        self.K_inv = None

    def kernel(self, x1, x2, length_scale=1.0, variance=1.0):
        dist = torch.cdist(x1, x2, p=2) ** 2
        return variance * torch.exp(-0.5 * dist / length_scale**2)

    def fit(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

        K = self.kernel(x_train, x_train)
        self.K_inv = torch.linalg.inv(K + 1e-6 * torch.eye(K.size(0)))  # Add small noise for numerical stability

    def predict(self, x_test, length_scale=1.0, variance=1.0):
        """
        Predict the mean and standard deviation at new test points.
        """
        # Ensure x_test is 2D (e.g., a single test point should be reshaped to (1, 2))
        if x_test.dim() == 1:
            x_test = x_test.unsqueeze(0)

        K_s = self.kernel(self.x_train, x_test, length_scale, variance)  # Cross-covariance between training and test points
        K_ss = self.kernel(x_test, x_test, length_scale, variance)  # Covariance of the test points with themselves

        # Compute the mean and standard deviation
        mean = K_s.T @ self.K_inv @ self.y_train
        std_dev = torch.sqrt(K_ss - K_s.T @ self.K_inv @ K_s)

        return mean, std_dev

# Reward function: Assuming the reward is negative loss computed on CPU
def compute_reward(Ppmp, phi_pmp=0):
    """
    Compute the reward (negative loss) for given parameters (Ppmp, phi_pmp).
    Assume the calculation happens on the CPU.
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
    return loss  # Reward is negative of the loss.

# Acquisition function - for simplicity, we use the **Upper Confidence Bound (UCB)**
def acquisition_function(x, gp, best_y, acquisition_type="ucb", kappa=2.0, xi=0.01, length_scale=1.0, variance=1.0):
    """
    Acquisition function with multiple options.
    
    Args:
        x: The current test point (1D tensor).
        gp: The Gaussian Process model.
        best_y: The best reward observed so far.
        acquisition_type: The type of acquisition function ('ucb', 'ei', or 'pi').
        kappa: Exploration parameter for UCB.
        xi: Exploration parameter for EI and PI.
        length_scale: Kernel length scale.
        variance: Kernel variance.
    
    Returns:
        Acquisition value at the test point.
    """
    # Reshape x to be 2D (1 test point with 2 features)
    x = x.unsqueeze(0)  # Now x is (1, 2)
    mean, std_dev = gp.predict(x, length_scale, variance)
    
    if acquisition_type == "ucb":
        # Upper Confidence Bound (UCB): mean + kappa * std_dev
        return mean + kappa * std_dev
    elif acquisition_type == "ei":
        # Expected Improvement (EI)
        z = (mean - best_y - xi) / (std_dev + 1e-9)
        ei = (mean - best_y - xi) * torch.distributions.Normal(0, 1).cdf(z) + std_dev * torch.distributions.Normal(0, 1).log_prob(z).exp()
        return ei
    elif acquisition_type == "pi":
        # Probability of Improvement (PI)
        z = (mean - best_y - xi) / (std_dev + 1e-9)
        pi = torch.distributions.Normal(0, 1).cdf(z)
        return pi
    else:
        raise ValueError(f"Invalid acquisition type: {acquisition_type}. Choose 'ucb', 'ei', or 'pi'.")

# Main loop for GPR-based optimization
# def gpr_bandit_optimization(num_iterations=10):
# Bounds for Ppmp and phi_pmp
bounds = [(0.001, 0.5), (0, 6.28)]  # Ppmp in range [0, 10], phi_pmp in range [0, 2*pi]

# Initialize training data
x_train = torch.tensor([[0.01, 0], [0.25, np.pi]], dtype=torch.float32)  # Initial samples
y_train = torch.tensor([compute_reward(x[0].item(), x[1].item()) for x in x_train], dtype=torch.float32)

# Initialize GP model
gp = GaussianProcess()

# Fit GP with initial data
gp.fit(x_train, y_train)
num_iterations = 50
# Optimization loop
for i in tqdm(range(num_iterations),ncols=120, desc='Optimization Loop'):
    # Find the next point by maximizing the acquisition function
    # Randomly initialize and optimize over the parameter space (Ppmp, phi_pmp)
    x_next = torch.tensor(np.random.uniform(*bounds, size=(1,2)), dtype=torch.float32)

    # Use the acquisition function to find the next best point
    # We will use the gradient-free optimization (minimize the negative acquisition function)
    best_y = y_train.max().item()  # Best reward so far
    acquisition = lambda x: acquisition_function(torch.tensor(x, dtype=torch.float32), gp, best_y)
    
    # Use scipy or any optimizer (e.g., L-BFGS-B) to optimize the acquisition function
    result = minimize(acquisition, x0=x_next.flatten().cpu().numpy(), bounds=bounds)
    x_next = torch.tensor(result.x, dtype=torch.float32)

    # Evaluate the reward at the new point
    y_next = torch.tensor([compute_reward(x_next[0].item(), x_next[1].item())], dtype=torch.float32)

    # Update the training set
    x_train = torch.cat([x_train, x_next.unsqueeze(0)])
    y_train = torch.cat([y_train, y_next])

    # Refit the Gaussian Process model
    gp.fit(x_train, y_train)

    # Print progress
    print(f"Iteration {i + 1}: x_next = {x_next.cpu().numpy()}, loss = {y_next.item()}")

# Final results
print(f"Optimal parameters: x = {x_train[y_train.argmin()].cpu().numpy()}, min loss: {y_train.min().item()}")

# Run the GPR bandit optimization
# gpr_bandit_optimization(num_iterations=50)
# %%