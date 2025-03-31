# %% [markdown]
# ## Getting started

# %% [markdown]
# Let's start by importing all the pacakges. If error occurs, please make sure to run `pip install -r requirements.txt`, file which can be find at the root of this repository

# %%
import os 
import sys
import inspect

currentdir = os.getcwd()
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

import pyLLE
import numpy as np
from scipy import constants as cts
import pandas as pd
import pickle as pkl
import plotly.graph_objs as go
import plotly.graph_objs
import plotly.io as pio
import time
pio.renderers.default = "notebook"
pio.templates.default = "seaborn"

# %% [markdown]
# Here we will define the parameters. 2 dictionaries are needed, the resonator and the simulations. Novelty of the version 4 is that the pump power and the pump frequency can be lists to simulate a multi pump system: 

# %%
res = dict(
        R=23e-6, 
        Qi=1e6, 
        Qc=1e6, 
        γ=3.2, 
        dispfile="./RW1000_H430.csv"
)

sim = dict(
    Pin=[140e-3], 
    f_pmp=[283e12],
    φ_pmp=[0], 
    δω=[None], 
    Tscan=0.7e6,
    μ_sim=[-220, 220],
    δω_init= 1e9 * 2 * np.pi,
    δω_end= -6.5e9 * 2 * np.pi,
    num_probe = 7000, 
)


# %%
solver = pyLLE.LLEsolver(sim=sim, res=res,debug=False)
solver.Analyze()
solver.Setup(verbose = True)
solver.SolveTemporal(bin = "julia-1.11.1/bin/julia", verbose=True)
solver.RetrieveData()

# %%
ref_spectrum = np.zeros(441)
def calculate_mse(ref_spectrum, spectrum):
    return np.linalg.norm(ref_spectrum - spectrum, ord=2)

# %%
s = solver.sol.Ewg[:,3345]
pred_spec = 10*np.log10(np.abs(s)) + 30
import matplotlib.pyplot as plt
plt.plot(pred_spec)

# %%
target_spectrum = pred_spec

# %%
from scipy.optimize import differential_evolution

# %%
# Define the simulation and resonator setup
res = dict(
    R=23e-6,
    Qi=1e6,
    Qc=1e6,
    γ=3.2,
    dispfile="./RW1000_H430.csv"
)

sim_template = dict(
    Pin=[190e-3],  # Placeholder, to be optimized
    f_pmp=[283e12],
    φ_pmp=[np.pi],  # Placeholder, to be optimized
    δω=[None],
    Tscan=0.7e6,
    μ_sim=[-220, 220],
    δω_init=1e9 * 2 * np.pi,
    δω_end=-6.5e9 * 2 * np.pi,
    num_probe=7000,
)

# Define the fitness function
def fitness_function(params):
    """
    Compute the fitness score for given Pin and φ_pmp.
    Arguments:
        params: Array containing [Pin, φ_pmp].
    Returns:
        Mean squared error (MSE) between the simulated and reference spectrum.
    """
    Pin = [params[0]]  # Optimized pump power
    φ_pmp = [params[1]]  # Optimized phase

    # Update the simulation parameters
    sim = sim_template.copy()
    sim['Pin'] = Pin
    sim['φ_pmp'] = φ_pmp

    # Solve the LLE
    solver = pyLLE.LLEsolver(sim=sim, res=res, debug=False)
    solver.Analyze()
    solver.Setup(verbose=False)
    solver.SolveTemporal(bin="julia-1.11.1/bin/julia", verbose=False)
    solver.RetrieveData()
    s = solver.sol.Ewg[:,3345]
    spectrum = 10*np.log10(np.abs(s)) + 30

    # Calculate the fitness score (MSE)
    mse = np.linalg.norm(target_spectrum - spectrum, ord=2)
    print(f"Pin: {Pin}, φ_pmp: {φ_pmp}, MSE: {mse}")
    return mse

# Define bounds for Pin and φ_pmp
bounds = [
    (0.001, 0.5),  # Pump power bounds (adjust based on your system)
    (0, 2*np.pi)  # Phase bounds
]

# Run the Differential Evolution algorithm
result = differential_evolution(
    fitness_function,
    bounds,
    strategy='best1bin',
    maxiter=100,
    popsize=15,
    tol=1e-6,
    mutation=(0.5, 1),
    recombination=0.7,
    seed=42
)

# Extract the results
optimal_params = result.x
optimal_fitness = result.fun
print(f"Optimal Parameters: {optimal_params}")
print(f"Optimal Fitness (MSE): {optimal_fitness}")

# Update sim with optimal parameters
optimized_Pin = [optimal_params[0]]
optimized_φ_pmp = [optimal_params[1]]
print(f"Optimized Pump Power (Pin): {optimized_Pin}")
print(f"Optimized Phase (φ_pmp): {optimized_φ_pmp}")


