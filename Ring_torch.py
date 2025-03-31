import torch
import numpy as np
from scipy.constants import c, hbar, pi
from torch.fft import fft, ifft
from tqdm import tqdm

class Ring:
    def __init__(self, ring_parameters, device="cuda"):
        # Ensure device compatibility (defaults to GPU if available)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # Convert parameters to tensors and send them to the chosen device
        self.N = ring_parameters['N']
        self.n0 = ring_parameters['n0']
        self.n2 = ring_parameters['n2']
        self.FSR = ring_parameters['FSR']
        self.lambda0 = ring_parameters['lambda0']
        self.kappa = ring_parameters['kappa']
        self.eta = ring_parameters['eta']
        self.Veff = ring_parameters['Veff']
        self.D2 = ring_parameters['D2']
        self.D3 = ring_parameters['D3']
        self.Pin = ring_parameters['Pin']
        
        # Calculate derived parameters and convert them to PyTorch tensors
        self.Tr = 1 / self.FSR
        self.freq0 = c / self.lambda0
        self.omega0 = 2 * pi * self.freq0
        self.g = hbar * self.omega0**2 * c * self.n2 / (self.n0**2 * self.Veff)
        self.kappa = 2 * pi * self.kappa

        # Initialize tensors for fields and dispersion on the specified device
        self.f = np.zeros(self.N, dtype=np.complex128)
        self.f[self.N // 2] = np.sqrt(
            (8 * self.g * self.eta / self.kappa**2) * (self.Pin / (hbar * self.omega0))
        )
        self.f = torch.tensor(self.f, dtype=torch.complex128, device=self.device)
        
        # Dispersion terms and mode numbers
        self.D2 = 2 * pi * self.D2
        self.D3 = 2 * pi * self.D3
        self.d2 = (2 / self.kappa) * self.D2
        self.d3 = (2 / self.kappa) * self.D3
        self.mu = torch.arange(-(self.N - 1) / 2, ((self.N - 1) / 2) + 1, 1, device=self.device)
        self.dint = ((self.d2 / 2) * self.mu**2 + (self.d3 / 6) * self.mu**3).to(self.device)

    def numerical_simulation(self, parameters, simulation_options, dseta_forward=torch.tensor([]), amu_forward=torch.tensor([]), theta_forward=torch.tensor([])):
        # Retrieve simulation options
        Effects = simulation_options['Effects']
        Noise = simulation_options['Noise']
        
        # Retrieve simulation parameters
        dseta_start = parameters['dseta_start']
        dseta_end = parameters['dseta_end']
        dseta_step = parameters['dseta_step']
        roundtrips_step = parameters['roundtrips_step']
        Amu0 = parameters.get('Amu0', None)
        
        # Set up optional effects
        if Effects == 'Thermal':
            theta0 = parameters.get('theta0', 0)
            tauT = parameters['tauT']
            n2T = parameters['n2T']
        elif Effects == 'Avoided mode crossings':
            theta0, tauT, n2T = 0, 0.1, 0
            mode_perturbated = parameters['mode_perturbated']
            self.dint[self.N // 2 + mode_perturbated] = 0
        else:
            theta0, tauT, n2T = 0, 0.1, 0

        # Initialize variables and tensors on the GPU
        dseta = torch.arange(dseta_start, dseta_end + dseta_step, dseta_step, device=self.device)
        tau_step = (self.kappa / 2) * self.Tr * roundtrips_step
        amu = torch.zeros((dseta.size(0), self.N), dtype=torch.complex128, device=self.device)
        amu[0, :] = ifft(torch.tensor(np.sqrt(2 * self.g / self.kappa), device=self.device) * torch.tensor(Amu0, dtype=torch.complex128, device=self.device)) if amu_forward.numel() == 0 else amu_forward[torch.abs(dseta_forward - dseta_start).argmin(), :]
        theta = torch.zeros(dseta.size(0), dtype=torch.complex128, device=self.device)
        theta[0] = theta0 if theta_forward.numel() == 0 else theta_forward[torch.abs(dseta_forward - dseta_start).argmin()]
        aux = 1j if Effects == 'Thermal' else 0

        # Adaptive RK4 parameters
        abs_tol, rel_tol = 1e-6, 1e-6
        max_factor, min_factor = 2.0, 0.1

        def LLE_step(amu_, theta_, dseta_current):
            damu_ = -(1 + 1j * (dseta_current + self.dint)) * amu_ + \
                    1j * ifft(torch.abs(fft(amu_))**2 * fft(amu_)) + self.f
            dtheta_ = (2 / (self.kappa * 0.1)) * (torch.sum(torch.abs(amu_)**2) - theta_)
            return damu_, dtheta_

        def adaptive_RK4_step(amu_, theta_, dseta_current, dt):
            k1_amu, k1_theta = LLE_step(amu_, theta_, dseta_current)
            k2_amu, k2_theta = LLE_step(amu_ + 0.5 * dt * k1_amu, theta_ + 0.5 * dt * k1_theta, dseta_current)
            k3_amu, k3_theta = LLE_step(amu_ + 0.5 * dt * k2_amu, theta_ + 0.5 * dt * k2_theta, dseta_current)
            k4_amu, k4_theta = LLE_step(amu_ + dt * k3_amu, theta_ + dt * k3_theta, dseta_current)

            full_step_amu = amu_ + (dt / 6) * (k1_amu + 2 * k2_amu + 2 * k3_amu + k4_amu)
            full_step_theta = theta_ + (dt / 6) * (k1_theta + 2 * k2_theta + 2 * k3_theta + k4_theta)

            # Half-step method for error estimation
            half_dt = dt / 2
            half_step_amu, half_step_theta = amu_, theta_
            for _ in range(2):  # two half-steps
                k1_amu, k1_theta = LLE_step(half_step_amu, half_step_theta, dseta_current)
                k2_amu, k2_theta = LLE_step(half_step_amu + 0.5 * half_dt * k1_amu, half_step_theta + 0.5 * half_dt * k1_theta, dseta_current)
                k3_amu, k3_theta = LLE_step(half_step_amu + 0.5 * half_dt * k2_amu, half_step_theta + 0.5 * half_dt * k2_theta, dseta_current)
                k4_amu, k4_theta = LLE_step(half_step_amu + half_dt * k3_amu, half_step_theta + half_dt * k3_theta, dseta_current)
                half_step_amu += (half_dt / 6) * (k1_amu + 2 * k2_amu + 2 * k3_amu + k4_amu)
                half_step_theta += (half_dt / 6) * (k1_theta + 2 * k2_theta + 2 * k3_theta + k4_theta)

            # Compute error and adaptive factor
            error_amu = torch.abs(full_step_amu - half_step_amu).max()
            error_theta = torch.abs(full_step_theta - half_step_theta).max()
            max_error = max(error_amu.item(), error_theta.item())
            scale_factor = 0.9 * (abs_tol / max_error)**0.2 if max_error != 0 else max_factor
            scale_factor = min(max(max_factor, scale_factor), min_factor)

            # Decide whether to accept step or not
            if max_error < abs_tol + rel_tol * max(full_step_amu.abs().max(), full_step_theta.abs().max()):
                return full_step_amu, full_step_theta, dt * scale_factor  # Accept with scaled step size
            else:
                return amu_, theta_, dt * scale_factor  # Reject and retry with reduced step size

        # Main simulation loop
        for i in tqdm(range(dseta.size(0) - 1),ncols=120):
            dseta_current = dseta[i]
            dt = tau_step  # Initial step size guess

            # Adaptive RK4 with step size adjustment
            amu_temp, theta_temp = amu[i, :], theta[i]
            while True:
                amu_temp, theta_temp, new_dt = adaptive_RK4_step(amu_temp, theta_temp, dseta_current, dt)
                if new_dt >= dt:  # Step accepted
                    amu[i + 1, :], theta[i + 1] = amu_temp, theta_temp
                    dt = new_dt  # Update step size for next iteration
                    break
                dt = new_dt  # Step rejected, retry with smaller step

        return dseta.cpu().numpy(), amu.cpu().numpy(), theta.cpu().numpy()
