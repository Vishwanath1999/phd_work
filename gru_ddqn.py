import torch as T
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import os
from copy import deepcopy

class ReplayBuffer:
    def __init__(self, max_size, input_shape):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
        self.action_memory = np.zeros((self.mem_size,), dtype=np.int64)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)

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
        self.loss = nn.MSELoss()
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
        self.load_state_dict(T.load(self.checkpoint_file))
    

class Agent:
    def __init__(self, n_actions, input_dims, run_name, lr=1e-4, mem_size=int(1e6), batch_size=128, gamma=0.99,\
                 eps_min=0.01, warm_up=10000, replace=1000, checkpoint_dir='./tmp/dueling_ddqn', fc_dims=256):
        
        self.gamma = gamma
        self.epsilon = 1.0
        self.eps_min = eps_min
        self.warm_up = warm_up
        self.n_actions = n_actions
        self.eps_dec = (1-self.eps_min)/self.warm_up
        self.batch_size = batch_size
        self.replace_target_cnt = replace
        self.run_name = run_name
        
        self.memory = ReplayBuffer(mem_size, input_dims)

        self.q_eval = DuelingDQN(lr, n_actions, input_dims, fc_dims=fc_dims, name=run_name+'_dueling_dqn_eval', chkpt_dir=checkpoint_dir)
        self.q_next = DuelingDQN(lr, n_actions, input_dims, fc_dims=fc_dims, name=run_name+'_dueling_dqn_next', chkpt_dir=checkpoint_dir)

        
    
    def choose_action(self, state, deterministic=False):
        state = T.tensor(np.array([state]), dtype=T.float).to(self.q_eval.device)
        if deterministic:
            _, advantage = self.q_eval.forward(state)
            action = T.argmax(advantage).item()
        else:
            if np.random.random() > self.epsilon:
                _, advantage = self.q_eval.forward(state)
                action = T.argmax(advantage).item()
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
    
    def learn(self, step):
        if self.memory.mem_cntr < self.batch_size:
            return
        
        self.replace_target_network(step)

        state, action, reward, state_, done = self.memory.sample_buffer(self.batch_size)

        states = T.tensor(state, dtype=T.float).to(self.q_eval.device)
        states_ = T.tensor(state_, dtype=T.float).to(self.q_eval.device)
        actions = T.tensor(action, dtype=T.int64).to(self.q_eval.device)
        rewards = T.tensor(reward, dtype=T.float).to(self.q_eval.device)
        dones = T.tensor(done).to(self.q_eval.device)
        indices = np.arange(self.batch_size)

        V_s, A_s = self.q_eval.forward(states)
        V_s_, A_s_ = self.q_next.forward(states_)
        V_s_eval, A_s_eval = self.q_eval.forward(states_)

        q_pred = T.add(V_s, (A_s-A_s.mean(dim=1, keepdim=True)))[indices, actions]
        q_next = T.add(V_s_, (A_s_-A_s_.mean(dim=1, keepdim=True)))
        q_eval = T.add(V_s_eval, (A_s_eval-A_s_eval.mean(dim=1, keepdim=True)))

        max_actions = T.argmax(q_eval, dim=1)
        
        q_target = rewards + self.gamma * q_next[indices, max_actions] * (1-dones.int())
        self.q_eval.optimizer.zero_grad()
        loss = self.q_eval.loss(q_target, q_pred)#.to(self.q_eval.device)
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_eval.parameters(), 1)
        self.q_eval.optimizer.step()
        self.decay_epsilon()
        return loss.item(), q_eval.mean(dim=0).cpu().detach().numpy()

