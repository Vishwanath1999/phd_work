# Highly optimized JAX version with extensive JIT and vmap
# %%
import jax
import jax.numpy as jnp
from jax import jit, vmap, lax
import numpy as np
from scipy.io import loadmat, savemat
from scipy import constants as cts
from tqdm import tqdm
from functools import partial

# Set up GPU device
jax.config.update('jax_platform_name', 'gpu')

disp = loadmat('disp.mat')
res = loadmat('res.mat')
sim = loadmat('sim.mat')

tol = 1e-3
max_iter = 10
step_factor = 0.1

c0 = 299792458
h_bar = cts.hbar

# %%
# Convert dictionary to JAX arrays
def dict_to_jax(dic):
    dic.pop('__header__', None)
    dic.pop('__version__', None)
    dic.pop('__globals__', None)
    dic.pop('dispfile', None)

    jax_dic = dict()
    for key in dic:
        if isinstance(dic[key], np.ndarray):
            dic[key] = np.where(dic[key] == 'None', 0, dic[key])
            dic[key] = dic[key].astype(np.float64)  # Ensure float64 for JAX compatibility
            jax_dic[key] = jnp.array(dic[key])
        else:
            jax_dic[key] = dic[key]
    return jax_dic


def jax_to_dict(jax_dict):
    dic = dict()
    for key in jax_dict:
        if isinstance(jax_dict[key], jnp.ndarray):
            dic[key] = np.array(jax_dict[key])
        else:
            dic[key] = jax_dict[key]
    return dic


disp_jax = dict_to_jax(disp)
res_jax = dict_to_jax(res)
sim_jax = dict_to_jax(sim)

# %%
# Squeeze arrays
disp_jax['D1'] = disp_jax['D1'][0]
disp_jax['FSR'] = disp_jax['FSR'][0]
disp_jax['FSR_center'] = disp_jax['FSR_center'][0]

for key in res_jax:
    res_jax[key] = res_jax[key][0]

for key in sim_jax:
    if sim_jax[key].ndim > 1:
        sim_jax[key] = sim_jax[key][0]

# %%
# Load parameters
ng = disp_jax['ng']
R = res_jax['R']
gamma = res_jax['gamma']
L = 2 * jnp.pi * R

Q0 = res_jax['Qi']
Qc = res_jax['Qc']
fpmp = sim_jax['f_pmp']
Ppmp = sim_jax['Pin']
phi_pmp = sim_jax['phi_pmp']
num_probe = 5000#int(sim_jax['num_probe'])
fcenter = sim_jax['f_center']

DKSinit_real = jnp.real(sim_jax['DKS_init'])
DKSinit_imag = jnp.where(
    sim_jax['DKS_init'].dtype == jnp.complex128,
    jnp.imag(sim_jax['DKS_init']),
    jnp.zeros_like(DKSinit_real)
)
DKS_init = DKSinit_real + 1j * DKSinit_imag

D1 = disp_jax['D1']
FSR = D1 / (2 * jnp.pi)
omega0 = 2 * jnp.pi * fpmp
omega_center = 2 * jnp.pi * fcenter

tR = 1 / FSR
T = 1 * tR
kext = omega0 / Qc * tR
k0 = omega0 / Q0 * tR
alpha = k0 + kext

del_omega_init = sim_jax['domega_init']
del_omega_end = sim_jax['domega_end']
del_omega_stop = sim_jax['domega_stop']
ind_sweep = sim_jax['ind_pump_sweep']
t_end = sim_jax['Tscan']
Dint = disp_jax['Dint_sim'][0]

# %%
del_omega = sim['domega']
ind_pmp = jnp.array([int(ii) for ii in sim_jax['ind_pmp']])
mu_sim = sim_jax['mucenter']
mu = jnp.arange(int(mu_sim[0]), int(mu_sim[1]) + 1)
mu0 = jnp.where(mu == 0)[0][0]

d_omega = 2 * jnp.pi * FSR * jnp.arange(int(mu_sim[0]), int(mu_sim[-1]) + 1)
domega_pmp1 = 2 * jnp.pi * FSR * jnp.arange(int(mu_sim[0] - ind_pmp[0]), int(mu_sim[-1] - ind_pmp[0]) + 1)
omega1 = omega0[0] + domega_pmp1

Dint = Dint - Dint[int(mu0)]

Ptot = jnp.sum(Ppmp)

dt = 1.0
t_end = t_end * tR
t_ramp = t_end

# %%
Nt = int(7e5)#int(jnp.round(t_ramp / tR / dt))
theta = jnp.linspace(0, 2 * jnp.pi, len(mu))
del_omega_tot = jnp.abs(del_omega_end) + jnp.abs(del_omega_init)
del_omega_perc = -1 * jnp.sign(del_omega_end + del_omega_init) * (jnp.abs(del_omega_end + del_omega_init) / 2) / del_omega_tot

t_sim = jnp.linspace(
    -t_ramp / 2 + del_omega_perc * t_ramp,
    t_ramp / 2 + del_omega_perc * t_ramp,
    Nt
)

# Vectorized creation of del_omega_all
xx = jnp.arange(1, Nt + 1, dtype=jnp.float64)
ind_sweep_set = set([int(idx) for idx in ind_sweep])

@vmap
def create_sweep_array(ii):
    """Create detuning sweep for each pump"""
    is_swept = jnp.isin(ii, jnp.array(list(ind_sweep_set)))
    swept_array = del_omega_init + xx / Nt * (del_omega_end - del_omega_init)
    static_array = jnp.ones(Nt) * del_omega_init
    return jnp.where(is_swept, swept_array, static_array)

del_omega_all = create_sweep_array(jnp.arange(len(fpmp)))

# %%
def setup_input_fields():
    """Setup input fields - vectorized"""
    
    def create_ein_row(ii):
        row = jnp.zeros(len(mu), dtype=jnp.complex128)
        pump_idx = int(mu0 + ind_pmp[ii])
        value = jnp.sqrt(Ppmp[ii]) * len(mu)
        return row.at[pump_idx].set(value)
    
    Ein = vmap(create_ein_row)(jnp.arange(len(fpmp)))
    
    def create_ain_row(ii, ein_row):
        return jnp.fft.ifft(jnp.fft.fftshift(ein_row)) * jnp.exp(1j * phi_pmp[ii])
    
    Ain = vmap(create_ain_row)(jnp.arange(len(fpmp)), Ein)
    
    return Ein, Ain


Ein, Ain = setup_input_fields()
u0 = DKS_init

# %%
saved_data = {
    'u_probe': jnp.zeros((num_probe, len(u0)), dtype=jnp.complex128),
    'driving_force': jnp.zeros((num_probe, len(u0)), dtype=jnp.complex128),
    'detuning': jnp.zeros(num_probe),
    't_sim': jnp.zeros(num_probe),
    'kappa_ext': kext,
    'kappa_0': k0,
    'alpha': alpha,
}

Dint_shift = jnp.fft.ifftshift(Dint)

# %%
# JIT-compiled operators with pytree support

@jit
def FFT_Lin_operator(it, alpha, Dint_shift, del_omega_all, tR):
    """Linear operator in frequency domain"""
    return (-alpha / 2) + 1j * (Dint_shift - del_omega_all[0, it]) * tR


@jit
def NL_operator(uu, gamma, L):
    """Nonlinear operator"""
    return -1j * (gamma * L * jnp.abs(uu) ** 2)


@jit
def compute_driving_force_single(ii, Ain_ii, del_omega_all_ii, mu0, ind_pmp_ii, Dint, del_omega_all_0_it, t_sim_it):
    """Vectorized driving force computation for single pump index"""
    sigma = lax.cond(
        ii > 0,
        lambda _: (2 * del_omega_all_ii + Dint[int(mu0 + ind_pmp_ii)] - 0.5 * del_omega_all_0_it) * t_sim_it,
        lambda _: 0.0,
        None
    )
    return -1j * Ain_ii * jnp.exp(1j * sigma)


@partial(jit, static_argnums=(1,))
def Fdrive_optimized(it, fpmp_len):
    """Optimized JIT-compiled driving force"""
    Force = jnp.zeros(len(theta), dtype=jnp.complex128)
    
    def add_force(ii, Force_acc):
        sigma = lax.cond(
            ii > 0,
            lambda _: (2 * del_omega_all[ii, it] + Dint[int(mu0 + ind_pmp[ii])] - 0.5 * del_omega_all[0, it]) * t_sim[it],
            lambda _: 0.0,
            None
        )
        return Force_acc - 1j * Ain[ii] * jnp.exp(1j * sigma)
    
    return lax.fori_loop(0, fpmp_len, add_force, Force)


# %%
@jit
def FFT_fwd(field):
    """Vectorized FFT"""
    return jnp.fft.fft(field)


@jit
def FFT_inv(field):
    """Vectorized IFFT"""
    return jnp.fft.ifft(field)


@jit
def convergence_check(A_prop, A_h_prop):
    """Compute convergence error"""
    return jnp.linalg.norm(A_prop - A_h_prop, 2) / (jnp.linalg.norm(A_h_prop, 2) + 1e-10)


# %%
@jit
def ssfm_step_jitted(A0, it, Fdrive_val, L_h_prop_val, NL_h_prop_0_val, tol, max_iter):
    """Fully JIT-compiled SSFM step with while_loop for convergence"""
    
    # Linear Part
    A0_updated = A0 + Fdrive_val * jnp.sqrt(kext) * dt
    A_L_h_prop = FFT_inv(FFT_fwd(A0_updated) * L_h_prop_val)
    
    # Convergence iteration using while_loop
    def body_fn(carry):
        A_h_prop, iteration = carry
        NL_h_prop_1 = NL_operator(A_h_prop, gamma, L)
        NL_prop = (NL_h_prop_0_val + NL_h_prop_1) * dt / 2
        A_prop = FFT_inv(FFT_fwd(A_L_h_prop * jnp.exp(NL_prop)) * L_h_prop_val)
        err = convergence_check(A_prop, A_h_prop)
        return (A_prop, iteration + 1), err
    
    def cond_fn(carry):
        A_prop, iteration = carry[0]
        err = carry[1]
        return (err > tol) & (iteration < max_iter)
    
    initial_carry = ((A_h_prop := A0_updated.copy(), 0), tol + 1.0)
    (A_final, _), err_final = lax.while_loop(cond_fn, body_fn, initial_carry)
    
    return A_final, err_final


# %%
@partial(jit, static_argnums=(1, 2))
def ssfm_step_wrapper(A0, it, fpmp_len):
    """Wrapper for JIT compilation with static arguments"""
    Fdrive_val = Fdrive_optimized(it, fpmp_len)
    L_h_prop_val = jnp.exp(FFT_Lin_operator(it, alpha, Dint_shift, del_omega_all, tR) * dt / 2)
    NL_h_prop_0_val = NL_operator(A0, gamma, L)
    
    A0_new, err = ssfm_step_jitted(A0, it, Fdrive_val, L_h_prop_val, NL_h_prop_0_val, tol, max_iter)
    return A0_new, err


def ssfm_step(A0, it):
    """SSFM step with error handling"""
    A0_new, err = ssfm_step_wrapper(A0, int(it), len(fpmp))
    return A0_new


# %%
@jit
def SaveStatus_Callback_jitted(it, Nt, u0, driving_force_input):
    """JIT-compiled callback for saving data"""
    probe_idx = int(it * num_probe / Nt)
    should_save = probe_idx < num_probe
    
    return (
        jnp.where(should_save, u0, jnp.zeros_like(u0)),
        jnp.where(should_save, driving_force_input, jnp.zeros_like(driving_force_input)),
        probe_idx
    )


def SaveData(saved_data):
    """Convert JAX arrays to numpy and save"""
    saved_data_numpy = {
        'u_probe': np.array(saved_data['u_probe']),
        'driving_force': np.array(saved_data['driving_force']),
        'detuning': np.array(saved_data['detuning']),
        't_sim': np.array(saved_data['t_sim']),
        'kappa_ext': np.array(saved_data['kappa_ext']),
        'kappa_0': np.array(saved_data['kappa_0']),
        'alpha': np.array(saved_data['alpha']),
    }
    savemat('SSFM_half_data.mat', saved_data_numpy)


# %%
def MainSolver(Nt, saved_data, u0):
    """Main SSFM solver loop"""
    
    for it in tqdm(range(Nt), ncols=120, desc="SSFM Integration"):
        u0 = ssfm_step(u0, it)
        
        # Optional: Save data at probe intervals (can be optimized with vmap)
        # if it * num_probe / Nt > param['probe']:
        #     probe_idx = int(it * num_probe / Nt)
        #     if probe_idx < num_probe:
        #         saved_data['u_probe'] = saved_data['u_probe'].at[probe_idx, :].set(u0)
        #         saved_data['detuning'] = saved_data['detuning'].at[probe_idx].set(del_omega_all[0, it])
        #         saved_data['t_sim'] = saved_data['t_sim'].at[probe_idx].set(t_sim[it])
        #         saved_data['driving_force'] = saved_data['driving_force'].at[probe_idx, :].set(Fdrive_optimized(it, len(fpmp)))
    
    return u0, saved_data


# %%
# Run the solver
print("Starting SSFM solver with GPU acceleration...")
u0_final, saved_data_final = MainSolver(Nt, saved_data, u0)
print(f"SSFM completed. Final state shape: {u0_final.shape}")

# Optionally save data
# SaveData(saved_data_final)