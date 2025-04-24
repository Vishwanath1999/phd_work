import torch as T
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import os
import numpy as np

class PrioritizedReplayBuffer:
    def __init__(self, max_size, input_shape, alpha=0.6):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.alpha = alpha
        self.epsilon = 1e-6  # small constant to avoid zero priority

        self.state_memory = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
        self.action_memory = np.zeros((self.mem_size,), dtype=np.int64)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)

        self.priorities = np.zeros((self.mem_size,), dtype=np.float32)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = done

        max_prio = self.priorities.max() if self.mem_cntr > 0 else 1.0
        self.priorities[index] = max_prio  # new sample gets max priority

        self.mem_cntr += 1

    def sample_buffer(self, batch_size, beta=0.4):
        max_mem = min(self.mem_cntr, self.mem_size)
        if max_mem == 0:
            raise ValueError("Cannot sample from an empty buffer!")

        # Compute probabilities with alpha
        scaled_priorities = self.priorities[:max_mem] ** self.alpha
        sample_probs = scaled_priorities / scaled_priorities.sum()

        indices = np.random.choice(max_mem, batch_size, p=sample_probs)
        
        # Importance-sampling weights
        total = max_mem
        weights = (total * sample_probs[indices]) ** (-beta)
        weights /= weights.max()  # Normalize

        states = self.state_memory[indices]
        actions = self.action_memory[indices]
        rewards = self.reward_memory[indices]
        states_ = self.new_state_memory[indices]
        dones = self.terminal_memory[indices]

        return states, actions, rewards, states_, dones, indices, weights

    def update_priorities(self, indices, td_errors):
        for idx, td_err in zip(indices, td_errors):
            self.priorities[idx] = abs(td_err) + self.epsilon
            # Ensure priorities are non-zero and positive
            self.priorities[idx] = max(self.priorities[idx], self.epsilon)

# Noisy Linear Layer
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma_init=0.017):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight_mu = nn.Parameter(T.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(T.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', T.empty(out_features, in_features))

        self.bias_mu = nn.Parameter(T.empty(out_features))
        self.bias_sigma = nn.Parameter(T.empty(out_features))
        self.register_buffer('bias_epsilon', T.empty(out_features))

        self.sigma_init = sigma_init
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init)

        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma_init)

    def reset_noise(self):
        self.weight_epsilon.normal_()
        self.bias_epsilon.normal_()

    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)

# Noisy Dueling DQN with GRU Encoder
class NoisyDuelingDQN(nn.Module):
    def __init__(self, alpha, n_actions, input_dims, fc_dims=256, name='noisy_dueling_dqn', chkpt_dir='./tmp/dueling_ddqn'):
        super(NoisyDuelingDQN, self).__init__()
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)

        self.seq_encoder = nn.GRU(input_size=input_dims[1], hidden_size=fc_dims, batch_first=True)

        # Noisy layers for value and advantage streams
        self.fc_v = NoisyLinear(fc_dims, fc_dims)
        self.V = NoisyLinear(fc_dims, 1)
        self.fc_a = NoisyLinear(fc_dims, fc_dims)
        self.A = NoisyLinear(fc_dims, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        _, h = self.seq_encoder(state)  # state: [batch, time, feature]
        x = h[0]  # final hidden state from GRU
        x_v = F.relu(self.fc_v(x))
        V = self.V(x_v)

        x_a = F.relu(self.fc_a(x))
        A = self.A(x_a)

        return V, A

    def reset_noise(self):
        self.fc_v.reset_noise()
        self.V.reset_noise()
        self.fc_a.reset_noise()
        self.A.reset_noise()

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file, weights_only=True))

class DuelingDQN(nn.Module):
    def __init__(self, alpha, n_actions, input_dims, fc_dims=256, name='dueling_dqn', chkpt_dir='./tmp/dueling_ddqn'):
        super(DuelingDQN, self).__init__()
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)

        self.seq_encoder = nn.GRU(input_size=input_dims[1], hidden_size=fc_dims, batch_first=True)
        
        self.fc_v = nn.Linear(fc_dims, fc_dims)
        self.fc_a = nn.Linear(fc_dims, fc_dims)
        self.V = nn.Linear(fc_dims, 1)
        self.A = nn.Linear(fc_dims, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
    
    def forward(self, state):
        _, h = self.seq_encoder(state)
        x_v = F.relu(self.fc_v(h[0]))
        x_a = F.relu(self.fc_a(h[0]))
        V = self.V(x_v)
        A = self.A(x_a)

        return V, A
    
    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)
    
    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file, weights_only=True))
    

class Agent:
    def __init__(self, n_actions, input_dims, run_name, lr=1e-4, mem_size=int(1e6), batch_size=128, gamma=0.99,\
                 eps_min=0.01, warm_up=10000, replace=1000, checkpoint_dir='./tmp/dueling_ddqn', fc_dims=256, eval=False,
                 use_noisy_layer=False):
        
        self.gamma = gamma
        self.epsilon = 1.0
        self.eps_min = eps_min
        self.warm_up = warm_up
        self.n_actions = n_actions
        self.eps_dec = (1-self.eps_min)/self.warm_up
        self.batch_size = batch_size
        self.replace_target_cnt = replace
        self.run_name = run_name
        self.use_noisy_layer = use_noisy_layer
        
        if eval == False:
            self.memory = PrioritizedReplayBuffer(max_size=mem_size, input_shape=input_dims, alpha=0.6)

        if self.use_noisy_layer==False:
            self.q_eval = DuelingDQN(lr, n_actions, input_dims, fc_dims=fc_dims, name=run_name+'_dueling_dqn_eval', chkpt_dir=checkpoint_dir)
            self.q_next = DuelingDQN(lr, n_actions, input_dims, fc_dims=fc_dims, name=run_name+'_dueling_dqn_next', chkpt_dir=checkpoint_dir)
        elif self.use_noisy_layer == True:
            self.q_eval = NoisyDuelingDQN(lr, n_actions, input_dims, fc_dims=fc_dims, name=run_name+'_noisy_dueling_dqn_eval', chkpt_dir=checkpoint_dir)
            self.q_next = NoisyDuelingDQN(lr, n_actions, input_dims, fc_dims=fc_dims, name=run_name+'_noisy_dueling_dqn_next', chkpt_dir=checkpoint_dir)
    
    def choose_action(self, state, deterministic=False):
        state = T.tensor(np.array([state]), dtype=T.float).to(self.q_eval.device)
        if deterministic:
            self.q_eval.eval()
            _, advantage = self.q_eval.forward(state)
            action = T.argmax(advantage).item()
        elif self.use_noisy_layer==False:
            if np.random.random() > self.epsilon:
                _, advantage = self.q_eval.forward(state)
                action = T.argmax(advantage).item()
            else:
                action = np.random.choice(self.n_actions)
        else:
            self.q_eval.reset_noise()
            _, advantage = self.q_eval.forward(state)
            action = T.argmax(advantage).item()
        return action
    
    def remember(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)
    
    def decay_epsilon(self):
        if self.epsilon > self.eps_min:
            self.epsilon -= self.eps_dec
        else:
            self.epsilon = self.eps_min
    
    def replace_target_network(self, step):
        if step % self.replace_target_cnt == 0:
            self.q_next.load_state_dict(self.q_eval.state_dict())
    
    def save_model(self):
        self.q_eval.save_checkpoint()
        self.q_next.save_checkpoint()
    
    def load_model(self):
        self.q_eval.load_checkpoint()
        self.q_next.load_checkpoint()
    
    def get_beta(self,current_step, beta_start=0.4, beta_frames=400000):
        beta = beta_start + (1.0 - beta_start) * min(current_step, beta_frames) / beta_frames
        return beta

    
    def learn(self, step):
        if self.memory.mem_cntr < self.batch_size:
            return
        
        self.replace_target_network(step)

        beta = self.get_beta(step)
        state, action, reward, state_, done, indices_per, weights = self.memory.sample_buffer(self.batch_size, beta=beta)


        states = T.tensor(state, dtype=T.float).to(self.q_eval.device)
        states_ = T.tensor(state_, dtype=T.float).to(self.q_eval.device)
        actions = T.tensor(action, dtype=T.int64).to(self.q_eval.device)
        rewards = T.tensor(reward, dtype=T.float).to(self.q_eval.device)
        dones = T.tensor(done).to(self.q_eval.device)
        weights_ = T.tensor(weights, dtype=T.float).to(self.q_eval.device)
        indices = np.arange(self.batch_size)

        if self.use_noisy_layer == True:
            self.q_eval.reset_noise()
            self.q_next.reset_noise()

        V_s, A_s = self.q_eval.forward(states)
        V_s_, A_s_ = self.q_next.forward(states_)
        V_s_eval, A_s_eval = self.q_eval.forward(states_)

        q_pred = T.add(V_s, (A_s-A_s.mean(dim=1, keepdim=True)))[indices, actions]
        q_next = T.add(V_s_, (A_s_-A_s_.mean(dim=1, keepdim=True)))
        q_eval = T.add(V_s_eval, (A_s_eval-A_s_eval.mean(dim=1, keepdim=True)))

        max_actions = T.argmax(q_eval, dim=1)
        
        q_target = rewards + self.gamma * q_next[indices, max_actions] * (1-dones.int())
        self.q_eval.optimizer.zero_grad()
        loss = (weights_*(q_target - q_pred)**2).mean()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_eval.parameters(), 1)
        self.q_eval.optimizer.step()

        if self.use_noisy_layer == False:
            self.decay_epsilon()

        # Update priorities
        td_errors = (q_target - q_pred).abs().cpu().detach().numpy()
        self.memory.update_priorities(indices_per, td_errors)

        return loss.item(), q_eval.mean(dim=0).cpu().detach().numpy(), weights

