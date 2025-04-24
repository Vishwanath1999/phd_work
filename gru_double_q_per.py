import torch as T
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import os

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


class DQN(nn.Module):
    def __init__(self, alpha, n_actions, input_dims, fc_dims=256, name='dueling_dqn', chkpt_dir='./tmp/dueling_ddqn'):
        super(DQN, self).__init__()
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)

        self.seq_encoder = nn.GRU(input_size=input_dims[1], hidden_size=fc_dims, batch_first=True)
        
        self.fc_q = nn.Linear(fc_dims, fc_dims)
        self.Q = nn.Linear(fc_dims, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
    
    def forward(self, state):
        _, h = self.seq_encoder(state)
        x_q = F.relu(self.fc_q(h[0]))
        Q = self.Q(x_q)

        return Q
    
    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)
    
    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file, weights_only=True))
    

class Agent:
    def __init__(self, n_actions, input_dims, run_name, lr=1e-4, mem_size=int(1e6), batch_size=128, gamma=0.99,\
                 eps_min=0.01, warm_up=10000, replace=1000, checkpoint_dir='./tmp/dueling_ddqn', fc_dims=256, eval_mode=False):
        
        self.gamma = gamma
        self.epsilon = 1.0
        self.eps_min = eps_min
        self.warm_up = warm_up
        self.n_actions = n_actions
        self.eps_dec = (1-self.eps_min)/self.warm_up
        self.batch_size = batch_size
        self.replace_target_cnt = replace
        self.run_name = run_name
        
        if eval_mode == False:
            self.memory = PrioritizedReplayBuffer(max_size=mem_size, input_shape=input_dims, alpha=0.6)

        self.q_eval = DQN(lr, n_actions, input_dims, fc_dims=fc_dims, name=run_name+'_dueling_dqn_eval', chkpt_dir=checkpoint_dir)
        self.q_next = DQN(lr, n_actions, input_dims, fc_dims=fc_dims, name=run_name+'_dueling_dqn_next', chkpt_dir=checkpoint_dir)

        
    
    def choose_action(self, state, deterministic=False):
        state = T.tensor(np.array([state]), dtype=T.float).to(self.q_eval.device)
        if deterministic:
            q = self.q_eval.forward(state)
            action = T.argmax(q).item()
        else:
            if np.random.random() > self.epsilon:
                q = self.q_eval.forward(state)
                action = T.argmax(q).item()
            else:
                action = np.random.choice(self.n_actions)

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
    
    def get_beta(self,current_step, beta_start=0.4, beta_frames=500000):
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

        q_pred = self.q_eval.forward(states)[indices, actions]
        q_next = self.q_next.forward(states_)
        q_eval = self.q_eval.forward(states_)

        max_actions = T.argmax(q_eval, dim=1)
        
        q_target = rewards + self.gamma * q_next[indices, max_actions] * (1-dones.int())
        self.q_eval.optimizer.zero_grad()
        loss = (weights_*(q_target - q_pred)**2).mean()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_eval.parameters(), 0.5)
        self.q_eval.optimizer.step()
        self.decay_epsilon()

        # Update priorities
        td_errors = (q_target - q_pred).abs().cpu().detach().numpy()
        self.memory.update_priorities(indices_per, td_errors)
        return loss.item(), q_eval.mean(dim=0).cpu().detach().numpy(),weights

