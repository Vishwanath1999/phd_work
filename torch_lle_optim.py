# %%
# optimized_env_loader.py
import os
import math
import numpy as np
import torch
from scipy.io import loadmat

H_BAR = 1.054571817e-34

# ---------------------------
# Utilities: sanitize loaded dict and convert to tensors on device
# ---------------------------

def _is_numpy_string_array(x: np.ndarray) -> bool:
    return isinstance(x, np.ndarray) and (x.dtype.kind in ("U", "S") or x.dtype == object)

def _numpy_to_tensor(x: np.ndarray, device: torch.device):
    """
    Convert numeric numpy array (real or complex) to torch tensor on device.
    """
    if np.iscomplexobj(x):
        return torch.tensor(x, dtype=torch.complex128, device=device)
    else:
        return torch.tensor(x, dtype=torch.float64, device=device)

def sanitize_value(v, device: torch.device):
    """
    Turn a variety of Python/NumPy/Torch objects into a torch tensor on device
    when appropriate. Strings and generic objects are returned unchanged.
    """
    # Already a torch tensor -> move to device
    if isinstance(v, torch.Tensor):
        return v.to(device)

    # Numpy arrays -> decide dtype
    if isinstance(v, np.ndarray):
        # If it's an array of strings/object that somehow remained, try to parse numeric strings
        if _is_numpy_string_array(v):
            # attempt to parse elements into numeric -> shape preserved
            flat = v.flatten()
            numeric = []
            for item in flat:
                s = str(item).strip()
                # try complex
                try:
                    c = complex(s)
                    numeric.append(c)
                    continue
                except Exception:
                    pass
                # try float
                try:
                    f = float(s)
                    numeric.append(f)
                    continue
                except Exception:
                    pass
                # if fails, return original python list of strings
                return v.tolist()
            arr = np.array(numeric).reshape(v.shape)
            return _numpy_to_tensor(arr, device)

        # normal numeric numpy array
        return _numpy_to_tensor(v, device)

    # Python scalar numeric -> tensor
    if isinstance(v, (float, int, np.floating, np.integer)):
        return torch.tensor(v, dtype=torch.float64, device=device)

    # Complex scalar
    if isinstance(v, complex) or isinstance(v, np.complexfloating):
        return torch.tensor(v, dtype=torch.complex128, device=device)

    # Python list: try to convert numeric lists to tensors, else return list
    if isinstance(v, list):
        try:
            arr = np.asarray(v)
            if arr.dtype == object:
                # mixed list -> leave as-is
                return v
            return _numpy_to_tensor(arr, device)
        except Exception:
            return v

    # Leave strings and other types as-is
    return v

def load_and_sanitize_pt(pt_path: str, device: torch.device):
    """
    Load a .pt file (dict) and sanitize values into torch tensors on device
    where possible. Returns a dict-like object.
    """
    assert os.path.exists(pt_path), f"{pt_path} not found"
    raw = torch.load(pt_path, map_location='cpu')  # load on CPU first
    out = {}
    for k, v in raw.items():
        out[k] = sanitize_value(v, device)
    return out

# ---------------------------
# Physics kernels (unchanged)
# ---------------------------
@torch.jit.script
def noise_op(h_bar: float, fpmp: torch.Tensor, mu_len: int, device: torch.device):
    omega1 = 2.0 * torch.pi * (fpmp[0] if isinstance(fpmp, torch.Tensor) else float(fpmp))
    Ephoton = h_bar * omega1
    phase = 2.0 * torch.pi * torch.rand(mu_len, dtype=torch.float64, device=device)  # Explicit dtype
    arr = torch.rand(mu_len, dtype=torch.float64, device=device)  # Explicit dtype
    Enoise = arr * torch.sqrt(Ephoton / 2.0) * torch.exp(1j * phase) * mu_len
    return torch.fft.ifftshift(torch.fft.ifft(Enoise))

@torch.jit.script
def lin_operator(alpha, Dint_shift, del_omega, tR):
    return (-alpha / 2.0) + 1j * (Dint_shift - del_omega) * tR

@torch.jit.script
def nl_operator(u, gamma, L):
    return -1j * (gamma * L * torch.square(torch.abs(u)))

@torch.jit.script
def vectorized_fdrive(Ain: torch.Tensor, del_omega_scalar: float, tR: float, fpmp: torch.Tensor, phi_pmp: torch.Tensor, device: torch.device):
    n_pumps = Ain.shape[0]
    mu_len = Ain.shape[1]
    sigma = torch.zeros((n_pumps,), dtype=torch.complex128, device=device)  # Explicitly use device
    for i in range(1, n_pumps):
        sigma[i] = (2.0 * del_omega_scalar) * tR
    ph = torch.exp(1j * sigma).unsqueeze(1)  # (n_pumps, 1)
    forced = -1j * torch.sum(Ain * ph, dim=0)
    forced = forced + noise_op(H_BAR, fpmp, mu_len, device)  # Pass device explicitly
    return forced

@torch.jit.script
def ssfm_step_core(A0, alpha, Dint_shift, del_omega, tR, gamma, L, max_iter, tol, dt, kext, Fdrive_val):
    A = A0 + Fdrive_val * torch.sqrt(kext) * dt
    Lh = torch.exp(lin_operator(alpha, Dint_shift, del_omega, tR) * (dt / 2.0))
    A_L = torch.fft.ifft(torch.fft.fft(A) * Lh)
    NL0 = nl_operator(A, gamma, L)
    A_prev = A.clone()
    for _ in range(max_iter):
        NL1 = nl_operator(A_prev, gamma, L)
        NLm = (NL0 + NL1) * (dt / 2.0)
        A_prop = torch.fft.ifft(torch.fft.fft(A_L * torch.exp(NLm)) * Lh)
        num = torch.linalg.vector_norm(A_prop - A_prev, ord=2)
        den = torch.linalg.vector_norm(A_prev, ord=2)
        rel = num / (den + 1e-12)
        if rel < tol:
            return A_prop
        A_prev = A_prop
    return A_prev

# ---------------------------
# Environment loader and runner (keeps your state formulation)
# ---------------------------
class OptimizedEnv:
    def __init__(self, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # load sanitized .pt files (assumes you've already converted .mat -> .pt)
        disp = load_and_sanitize_pt('disp_copy.pt', self.device)
        res  = load_and_sanitize_pt('res_copy.pt', self.device)
        sim  = load_and_sanitize_pt('sim_copy.pt', self.device)

        # map to tensors ensuring types and shapes
        # Dispersion
        assert 'D1' in disp, "D1 missing in disp_copy.pt"
        self.D1 = disp['D1'].squeeze().to(self.device, dtype=torch.float64)
        self.FSR = (self.D1 / (2.0 * math.pi)).to(self.device)
        # try keys for Dint variants
        if 'Dint_new' in disp:
            self.Dint_new = disp['Dint_new'].squeeze().to(self.device, dtype=torch.float64)
        elif 'Dint' in disp:
            self.Dint_new = disp['Dint'].squeeze().to(self.device, dtype=torch.float64)
        else:
            raise KeyError("No Dint_new or Dint in disp file")
        mu_len = self.D1.numel()
        center_idx = mu_len // 2
        self.Dint = (self.Dint_new - self.Dint_new[center_idx]).to(self.device)
        self.Dint_shift = torch.fft.ifftshift(self.Dint)

        # Resonator
        self.R = res['R'].squeeze().to(self.device, dtype=torch.float64)
        self.gamma = res['gamma'].squeeze().to(self.device, dtype=torch.float64)
        self.Qi = res['Qi'].squeeze().to(self.device, dtype=torch.float64)
        self.Qc = res['Qc'].squeeze().to(self.device, dtype=torch.float64)
        self.L = 2.0 * math.pi * self.R

        # Simulation
        self.fpmp = sim['f_pmp'].squeeze().to(self.device, dtype=torch.float64)
        self.Pin = sim['Pin'] if isinstance(sim['Pin'], torch.Tensor) else torch.tensor(sim['Pin'], dtype=torch.float64, device=self.device)
        self.phi_pmp = sim['phi_pmp'] if isinstance(sim['phi_pmp'], torch.Tensor) else torch.tensor(sim['phi_pmp'], dtype=torch.float64, device=self.device)
        # ind_pmp may be float/double tensor -> convert to long
        ind_pmp_raw = sim.get('ind_pmp', None)
        if ind_pmp_raw is None:
            raise KeyError("ind_pmp missing in sim file")
        self.ind_pmp = ind_pmp_raw.to(self.device) if isinstance(ind_pmp_raw, torch.Tensor) else torch.tensor(ind_pmp_raw, dtype=torch.long, device=self.device)

        # DKS_init: may be complex tensor already
        DKS_init_raw = sim.get('DKS_init', None)
        if DKS_init_raw is None:
            self.DKS_init = torch.zeros(mu_len, dtype=torch.complex128, device=self.device)
        else:
            if isinstance(DKS_init_raw, torch.Tensor):
                # ensure complex dtype
                if not torch.is_complex(DKS_init_raw):
                    self.DKS_init = DKS_init_raw.to(self.device, dtype=torch.float64).to(torch.complex128)
                else:
                    self.DKS_init = DKS_init_raw.to(self.device)
            elif isinstance(DKS_init_raw, np.ndarray):
                self.DKS_init = _numpy_to_tensor(DKS_init_raw, self.device).to(torch.complex128)
            else:
                raise TypeError("Unsupported DKS_init type")

        # Derived params
        omega0 = 2.0 * math.pi * (self.fpmp if self.fpmp.ndim > 0 else self.fpmp.unsqueeze(0))
        self.tR = (1.0 / self.FSR)
        kext = (omega0[0] / self.Qc) * self.tR
        k0 = (omega0[0] / self.Qi) * self.tR
        self.kext = kext
        self.alpha = k0 + kext
        self.un_norm_kappa = 2.0 * math.pi * (self.fpmp.flatten()[0] / self.Qi + self.fpmp.flatten()[0] / self.Qc)

        # mu grid (from sim['mucenter'] or fallback)
        mu_sim = sim.get('mucenter', None)
        if mu_sim is not None:
            mu_sim_t = mu_sim if isinstance(mu_sim, torch.Tensor) else torch.tensor(mu_sim, device=self.device)
            mu_start = int(mu_sim_t[0].item())
            mu_end = int(mu_sim_t[1].item())
            self.mu = torch.arange(mu_start, mu_end + 1, device=self.device)
            self.mu0 = int((self.mu == 0).nonzero()[0].item()) if (self.mu == 0).any() else (self.mu.numel() // 2)
        else:
            self.mu = torch.arange(0, mu_len, device=self.device)
            self.mu0 = mu_len // 2

        # Prepare Ain/Ein (complex)
        n_pumps = 1 if (self.fpmp.ndim == 0) else self.fpmp.numel()
        mu_len = self.mu.numel()
        self.Ain = torch.zeros((n_pumps, mu_len), dtype=torch.complex128, device=self.device)
        self.Ein = torch.zeros_like(self.Ain)
        Pin_v = self.Pin if (isinstance(self.Pin, torch.Tensor) and self.Pin.ndim > 0) else (torch.tensor([self.Pin], device=self.device) if not isinstance(self.Pin, torch.Tensor) else self.Pin.unsqueeze(0))
        phi_v = self.phi_pmp if (isinstance(self.phi_pmp, torch.Tensor) and self.phi_pmp.ndim > 0) else (torch.tensor([self.phi_pmp], device=self.device) if not isinstance(self.phi_pmp, torch.Tensor) else self.phi_pmp.unsqueeze(0))
        for i in range(n_pumps):
            idx = int(self.mu0 + int(self.ind_pmp[i].item()))
            self.Ein[i, idx] = torch.sqrt(Pin_v[i]) * mu_len
            self.Ain[i] = torch.fft.ifft(torch.fft.fftshift(self.Ein[i])) * torch.exp(-1j * phi_v[i])

        # thermal & control defaults (kept as in your env)
        self.tau0 = 100e-9
        self.xi = -4.5e4
        self.delta_theta = torch.tensor(0.0, dtype=torch.float64, device=self.device)

        self.del_omega_init = (7.0 * self.un_norm_kappa).to(self.device)
        self.del_omega_end = (-9.0 * self.un_norm_kappa).to(self.device)
        self.current_del_omega = self.del_omega_init.clone()

        self.seq_len = 100
        self.p_max = 0.2
        self.p_min = 0.05
        self.ctrl_freq = 100
        self.max_steps = int(4e5)
        self.Nt = self.max_steps

        # histories (keep on CPU for logging)
        self.ecav_state = np.zeros((self.seq_len, mu_len), dtype=float)
        self.pcav_hist = []

        # warm-up small number of steps for quick init
        self.reset(steps=500)

    def reset(self, steps: int = None):
        self.state = self.DKS_init.clone().to(torch.complex128)
        self.step_cntr = 0
        self.delta_theta = torch.tensor(0.0, dtype=torch.float64, device=self.device)
        self.current_del_omega = self.del_omega_init.clone()
        self.t_sim_step = 0.0

        self.power = torch.tensor([0.05], dtype=torch.float64, device=self.device)
        Pin_v = self.power
        for ii in range(self.Ain.size(0)):
            idx = int(self.mu0 + int(self.ind_pmp[ii].item()))
            self.Ein[ii] = torch.zeros_like(self.Ein[ii])
            self.Ein[ii, idx] = torch.sqrt(Pin_v[0]) * self.mu.numel()
            self.Ain[ii] = torch.fft.ifft(torch.fft.fftshift(self.Ein[ii])) * torch.exp(-1j * self.phi_pmp[ii])

        init_steps = steps if (steps is not None) else 500
        for idx in range(init_steps):
            r = torch.rand(1, device=self.device).item()
            mul = 1.0 if r > 0.66 else (-1.0 if r > 0.33 else 0.0)
            dd = self.current_del_omega + mul * (1.0/self.Nt) * (self.del_omega_end - self.del_omega_init)
            self.current_del_omega = dd
            Fdrive_val = vectorized_fdrive(self.Ain, self.current_del_omega + (self.delta_theta/self.tR), self.tR, self.fpmp if self.fpmp.ndim>0 else self.fpmp.unsqueeze(0), self.phi_pmp, self.device)
            u0 = ssfm_step_core(self.state, self.alpha, self.Dint_shift, self.current_del_omega + (self.delta_theta/self.tR), self.tR, self.gamma, self.L, 5, 1e-3, 1.0, self.kext, Fdrive_val)
            self.state = u0
            P_avg = torch.mean(torch.abs(u0)**2)
            ddt = -self.delta_theta / self.tau0 + self.xi * P_avg
            self.delta_theta += (1.0 * self.tR) * ddt
            self.t_sim_step += float(self.tR)

        Acav = torch.sqrt(self.alpha/2.0) * self.state * torch.exp(1j * torch.tensor(math.pi, device=self.device)) / math.sqrt(self.mu.numel())
        Acav_np = Acav.detach().cpu().numpy()
        self.ecav_state = np.zeros((self.seq_len, self.mu.numel()), dtype=float)
        self.pcav_hist = []
        return self.state, Acav_np, self.ecav_state.copy(), np.array(self.pcav_hist)

    def rescale_and_quantize(self, action, lower_limit, upper_limit, step_size):
        a = float(max(min(action, 1.0), -1.0))
        value = lower_limit + (a + 1.0) * (upper_limit - lower_limit) / 2.0
        q = round(value / step_size) * step_size
        return q

    def rescale_power(self, power, lower_limit, upper_limit, step_size):
        p = float(max(min(power, 1.0), -1.0))
        value = lower_limit + (p + 1.0) * (upper_limit - lower_limit) / 2.0
        q = round(value / step_size) * step_size
        return q

    def step(self, state, action, desired_spectrum):
        pow_val = self.rescale_power(float(action[0]), self.p_min, self.p_max, 0.001)
        self.power = torch.tensor([pow_val], dtype=torch.float64, device=self.device)

        for ii in range(self.Ain.size(0)):
            idx = int(self.mu0 + int(self.ind_pmp[ii].item()))
            self.Ein[ii] = torch.zeros_like(self.Ein[ii])
            self.Ein[ii, idx] = torch.sqrt(self.power[0]) * self.mu.numel()
            self.Ain[ii] = torch.fft.ifft(torch.fft.fftshift(self.Ein[ii])) * torch.exp(-1j * self.phi_pmp[ii])

        delta_omega = (2.0 * math.pi) * self.rescale_and_quantize(float(action[1]), -2e6, 2e6, 1e4)
        delta_omega = torch.tensor(delta_omega, dtype=torch.float64, device=self.device)

        for _ in range(self.ctrl_freq):
            new_del = self.current_del_omega + delta_omega
            self.current_del_omega = torch.clamp(new_del, min=min(self.del_omega_end, self.del_omega_init), max=max(self.del_omega_end, self.del_omega_init))
            full_det = self.current_del_omega + (self.delta_theta / self.tR)
            Fdrive_val = vectorized_fdrive(self.Ain, full_det, self.tR, self.fpmp if self.fpmp.ndim>0 else self.fpmp.unsqueeze(0), self.phi_pmp, self.device)
            u0 = ssfm_step_core(state, self.alpha, self.Dint_shift, full_det, self.tR, self.gamma, self.L, 5, 1e-3, 1.0, self.kext, Fdrive_val)
            state = u0
            P_avg = torch.mean(torch.abs(u0)**2)
            ddt = -self.delta_theta / self.tau0 + self.xi * P_avg
            self.delta_theta += (1.0 * self.tR) * ddt
            self.step_cntr += 1
            self.t_sim_step += float(self.tR)

        self.next_state = state.clone()
        Acav = torch.sqrt(self.alpha/2.0) * state * torch.exp(1j * torch.tensor(math.pi, device=self.device)) / math.sqrt(self.mu.numel())
        Ecav = torch.fft.fftshift(torch.fft.fft(Acav)) / math.sqrt(self.mu.numel())
        cav = Fdrive_val * torch.sqrt(1.0 - self.kext)
        wg = torch.sqrt(self.kext) * state * torch.exp(1j * torch.tensor(math.pi, device=self.device))
        Awg = (wg + cav) / math.sqrt(self.mu.numel())
        Ewg = torch.fft.fftshift(torch.fft.fft(Awg)) / math.sqrt(self.mu.numel())

        Ecav_dBm = 10.0 * torch.log10(torch.abs(Ewg)**2 + 1e-20) + 30.0
        Ecav_dBm = torch.clamp(Ecav_dBm, min=-60.0)

        ecav_cpu = Ecav_dBm.detach().cpu().numpy()
        self.ecav_state = np.roll(self.ecav_state, -1, axis=0)
        self.ecav_state[-1, :] = ecav_cpu

        desired_spectrum_dBm = 10.0 * torch.log10(torch.abs(desired_spectrum.to(self.device))**2 + 1e-20) + 30.0
        desired_spectrum_dBm = torch.clamp(desired_spectrum_dBm, min=-60.0)

        center_idx = self.mu.numel() // 2
        mask = (Ecav_dBm > -60.0)
        mask[center_idx] = False
        width = torch.sum(mask.float()).item() / 300.0

        stacked = torch.stack([Ecav_dBm, desired_spectrum_dBm])
        corr = torch.corrcoef(stacked)[0, 1].item()

        sym = 0.0
        try:
            left = Ecav_dBm[:center_idx][Ecav_dBm[:center_idx] > -55.0]
            right = Ecav_dBm[center_idx+1:][Ecav_dBm[center_idx+1:] > -55.0]
            minlen = min(left.numel(), right.numel())
            if minlen >= 2:
                r = torch.corrcoef(torch.stack((left[:minlen], torch.flip(right[:minlen], dims=[0]))))[0, 1].item()
                sym = float(r)
        except Exception:
            sym = 0.0

        reward = 2.0 * width + 2.0 * corr + 2.0 * sym
        reward -= 0.5 * float(torch.std(self.power))

        achieved = False
        if torch.linalg.vector_norm(desired_spectrum_dBm - Ecav_dBm, ord=2) < 50.0 or corr > 0.9:
            achieved = True
            reward += 2.0

        terminal = False
        if (self.step_cntr - 0) >= int(0.5 * self.Nt) and corr < 0.25:
            terminal = True
            reward += -5.0

        done = (self.step_cntr + 1 >= self.max_steps) or terminal

        Acav_np = Acav.detach().cpu().numpy()
        return self.next_state, reward, done, terminal, achieved, Acav_np, self.ecav_state.copy(), Ewg.detach().cpu().numpy()

# ---------------------------
# Unit tests (adapted)
# ---------------------------
def test_reset(env):
    state, Acav_np, ecav_state, pcav = env.reset()
    assert torch.is_tensor(state)
    assert isinstance(Acav_np, np.ndarray)
    assert isinstance(ecav_state, np.ndarray)
    print("Reset test passed.")

def test_step(env):
    state, _, _, _ = env.reset()
    action = torch.tensor([0.0, 0.0])
    desired = torch.zeros(env.mu.numel(), dtype=torch.complex128).to(env.device)
    out = env.step(state, action, desired)
    assert isinstance(out, tuple) and len(out) == 8
    print("Step test passed.")

def test_gpu_consistency(env):
    state, _, _, _ = env.reset()
    action = torch.tensor([0.0, 0.0]).to(env.device)
    desired = torch.zeros(env.mu.numel(), dtype=torch.complex128).to(env.device)
    next_state, reward, done, terminal, achieved, Acav_np, ecav_state, Ewg_np = env.step(state, action, desired)
    assert torch.isfinite(next_state).all()
    assert np.isfinite(Acav_np).all()
    print("GPU consistency test passed.")
# %%
import time
if __name__ == "__main__":
    print("Instantiating env (this may take a moment)...")
    env = OptimizedEnv(device='cuda')
    test_reset(env)
    test_step(env)
    test_gpu_consistency(env)
    print("All unit tests passed.")

    # Measure time for one episode
    print("Running one episode for max_steps...")
    state, _, _, _ = env.reset()
    action = torch.tensor([0.0, 0.0]).to(env.device)
    desired = torch.zeros(env.mu.numel(), dtype=torch.complex128).to(env.device)

    start_time = time.time()
    for _ in range(env.max_steps):
        state, reward, done, terminal, achieved, Acav_np, ecav_state, Ewg_np = env.step(state, action, desired)
        if done:
            break
    end_time = time.time()

    elapsed_time = end_time - start_time
    print(f"Time taken for one episode with {env.max_steps} steps: {elapsed_time:.2f} seconds.")
# %%
