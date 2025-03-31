
# Reinforcement Learning for Microresonator Frequency Comb Control

This repository contains code and resources for training and evaluating reinforcement learning (RL) agents to control microresonator frequency combs. The project leverages deep reinforcement learning techniques, such as Dueling Double Deep Q-Networks (DDQN), to optimize the detuning and pump power of microresonators for achieving desired spectral properties.

## Repository Structure

```
.
├── README.md                     # Project documentation
├── rl_mrr_all_ddqn.py            # Main script for RL training and evaluation
├── actor_critic.py               # Actor-Critic implementation
├── dueling_dqn.py                # Dueling DDQN agent implementation
├── desired_spec.mat              # Desired spectrum for training
├── disp.mat                      # Dispersion parameters
├── res.mat                       # Resonator parameters
├── sim.mat                       # Simulation parameters
├── My_DKS_init_a3c.mat           # Initial conditions for DKS
├── maddpg_results/               # Directory for saving results (plots, models, etc.)
├── tmp/                          # Directory for saving model checkpoints
└── other_files...                # Additional scripts and resources
```

## Features

- **Reinforcement Learning Environment**: The `RL_MRR_Env` class in [`rl_mrr_all_ddqn.py`](rl_mrr_all_ddqn.py) simulates the microresonator dynamics and provides an interface for RL agents.
- **Dueling DDQN Agent**: The [`dueling_dqn.py`](dueling_dqn.py) script implements a Dueling Double Deep Q-Network for efficient learning.
- **Custom Reward Function**: The reward function combines spectral mean squared error (MSE) and power cavity (Pcav) MSE to guide the agent toward optimal control.
- **Visualization**: The repository includes scripts for visualizing training progress, spectral evolution, and rewards.

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure the following files are available in the repository:
   - `disp.mat`
   - `res.mat`
   - `sim.mat`
   - `desired_spec.mat`
   - `My_DKS_init_a3c.mat`

## Usage

### Training the RL Agent

Run the [`rl_mrr_all_ddqn.py`](rl_mrr_all_ddqn.py) script to train the Dueling DDQN agent:
```bash
python rl_mrr_all_ddqn.py
```

The script initializes the environment, trains the agent, and logs the results using [Weights & Biases](https://wandb.ai/).

### Visualizing Results

After training, the script generates plots for:
- Spectral evolution
- Rewards vs. iterations
- Actions taken by the agent
- Comparison of obtained and desired spectra

These plots are saved in the `maddpg_results/` directory.

### Testing the RL Agent

To test the trained agent, modify the script to load the saved model and evaluate its performance on the environment.

## Key Components

### RL Environment (`RL_MRR_Env`)

The environment simulates the dynamics of a microresonator, including:
- Dispersion parameters (`disp.mat`)
- Resonator parameters (`res.mat`)
- Simulation parameters (`sim.mat`)

It provides methods for resetting the environment, stepping through actions, and computing rewards.

### Dueling DDQN Agent

The agent uses a Dueling Double Deep Q-Network to learn optimal control policies. Key features include:
- Experience replay
- Target network updates
- Epsilon-greedy exploration

### Reward Function

The reward function combines:
- Spectral MSE: Measures the difference between the desired and obtained spectra.
- Pcav MSE: Measures the deviation of the power cavity from a reference.

## Results

The repository includes scripts for visualizing the training and testing results. Example plots include:
- Spectral evolution over tuning steps
- Rewards vs. iterations
- Actions taken by the agent
- Comparison of obtained and desired spectra

## Dependencies

- Python 3.9+
- PyTorch
- NumPy
- SciPy
- Matplotlib
- tqdm
- Weights & Biases (wandb)
- fastdtw
- numba

## Acknowledgments

This project was developed as part of research on microresonator frequency comb control using reinforcement learning. Special thanks to the contributors and collaborators.