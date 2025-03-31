import Ring_torch as rng
import numpy as np
import matplotlib.pyplot as plt

parameters_ring1 =  {  
                        'N': 511, # Number of modes. It must be odd!
                        'n0': 2.4, # Refractive index
                        'n2': 2.4e-19, # Nonlinear reftactive index [m^2/W]
                        'FSR': 100e9, # Free Spectral Range [Hz]
                        'lambda0': 1553.4e-9, # CW pump wavelength [m]
                        'kappa': 3e8, # Optical linewidth [Hz]
                        'eta': 0.5, # Coupling efficiency
                        'Veff': 1e-15, # Effective mode volume [m^3]
                        'D2': 2.5e6, # Second order dispersion [Hz],
                        'D3': 0, # Third order dispersion [Hz]
                        'Pin': 2 # Pump power [W]
                    }
ring1 = rng.Ring(parameters_ring1) # Init Ring class

simulation_options_ring1 =  { 
                                'Effects': None, # None, "Thermal" or "Avoided mode crossings"
                                'Noise': False # True or False (White noise)
                            }

forward_parameters_ring1 =  {
                                'dseta_start': -10, # Normalized detuning start
                                'dseta_end': 45, # Normalized detuning end
                                'dseta_step': 0.01, # Tuning step
                                'roundtrips_step': 1, # Roundtrips per tuning step
                                # 'Amu0': psi_init1 # Initial field
                                'Amu0': np.random.randn(parameters_ring1['N']) + 1j * np.random.randn(parameters_ring1['N']), # Initial field
                            }
dseta_forward_ring1, amu_forward_ring1, _ = ring1.numerical_simulation(forward_parameters_ring1, simulation_options_ring1) # Run forward simulation