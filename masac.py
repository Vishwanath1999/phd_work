import numpy as np
import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, Categorical

class ReplayBuffer:
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
        self.action_memory = np.zeros((self.mem_size, n_actions), dtype=np.float32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = done

        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, states_, dones

    
class ActorNetworkContinuous(nn.Module):
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims, n_actions, max_action, min_action, name='cont_actor', chkpt_dir='tmp/sac'):
        super(ActorNetworkContinuous, self).__init__()
        self.checkpoint_file = os.path.join(chkpt_dir, name+'_sac')
        self.fc1 = nn.Linear(*input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.mu = nn.Linear(fc2_dims, n_actions)
        self.sigma = nn.Linear(fc2_dims, n_actions)

        self.max_action = max_action
        self.min_action = min_action

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state):
        prob = self.fc1(state)
        prob = F.elu(prob)
        prob = self.fc2(prob)
        prob = F.elu(prob)

        mu = self.mu(prob)
        sigma = self.sigma(prob)

        sigma = T.clamp(sigma, min=-20, max=2).exp()

        return mu, sigma

    def sample_action(self, state, reparam=False, deterministic=True):
        mu, sigma = self.forward(state)

        if deterministic:
            action = T.tanh(mu)
            # rescale between min and max action
            rescaled_action = self.min_action + (action + 1.) * (self.max_action - self.min_action)/2
            return rescaled_action, None
        else:
            probabilities = Normal(mu, sigma)
            actions = probabilities.rsample() if reparam else probabilities.sample()
            action = T.tanh(actions)
            log_probs = probabilities.log_prob(actions)
            log_probs -= T.log(1 - action.pow(2) + 1e-6)
            log_probs = log_probs.sum(1, keepdim=True)
            rescaled_action = self.min_action + (action + 1.) * (self.max_action - self.min_action)/2
            return rescaled_action, log_probs
    
class ActorNetworkDiscrete(nn.Module):
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims, n_actions, name='disc_actor', chkpt_dir='tmp/sac'):
        super(ActorNetworkDiscrete, self).__init__()
        self.checkpoint_file = os.path.join(chkpt_dir, name+'_sac')
        self.fc1 = nn.Linear(*input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.pi = nn.Linear(fc2_dims, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state):
        prob = self.fc1(state)
        prob = F.elu(prob)
        prob = self.fc2(prob)
        prob = F.elu(prob)

        pi = self.pi(prob)
        return pi
    
    def sample_action(self, state, reparam=False, deterministic=False):
        pi = self.forward(state)
        probs = F.gumbel_softmax(pi, hard=deterministic, tau=1)
        dist = Categorical(probs)
        action = dist.sample()
        log_probs = dist.log_prob(action).sum(dim=-1, keepdim=False)
        return action, log_probs, probs
    
    
class MixerNetwork(nn.Module):
    def __init__(self, beta, input_dims, fc_dims, output_dims, name='mixer', chkpt_dir='tmp/sac'):
        super(MixerNetwork, self).__init__()
        self.checkpoint_file = os.path.join(chkpt_dir, name+'_sac')
        
        self.fc1s = nn.ModuleList([nn.Linear(input_dims[idx], fc_dims) for idx in range(len(input_dims))])
        self.fc2s = nn.ModuleList([nn.Linear(fc_dims, fc_dims) for _ in range(len(input_dims))])
        self.qs = nn.ModuleList([nn.Linear(fc_dims, output_dims[idx]) for idx in range(len(input_dims))])

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
    
    # def forward(self, states, cont_actions, disc_actions):
        