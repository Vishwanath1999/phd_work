import torch as T
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import os
import numpy as np
from torch.distributions import Categorical

class ReplayBuffer:
    def __init__(self, max_size, input_shape):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
        self.log_probs = np.zeros((self.mem_size,), dtype=np.float32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)

    def store_transition(self, state, log_prob, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.log_probs[index] = log_prob
        self.terminal_memory[index] = done

        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_memory[batch]
        log_prob = self.log_probs[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        dones = self.terminal_memory[batch]

        return states, log_prob, rewards, states_, dones

class ActorCriticNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc_dims=256, n_actions=3, name='actor_critic', chkpt_dir='./tmp/actor_critic'):
        super(ActorCriticNetwork, self).__init__()

        self.seq_encoder = nn.GRU(input_dims[1], hidden_size=fc_dims, batch_first=True)

        
        # self.pi = nn.Sequential(
        #     nn.Linear(fc_dims, fc_dims),
        #     nn.ELU(),
        #     nn.Linear(fc_dims, n_actions),
        #     nn.Softmax(dim=-1)
        # )

        # self.v = nn.Sequential(
        #     nn.Linear(fc_dims, fc_dims),
        #     nn.ELU(),
        #     nn.Linear(fc_dims,1)
        # )
        self.fc_pi = nn.Linear(fc_dims, fc_dims)
        self.pi = nn.Linear(fc_dims, n_actions)

        self.fc_v = nn.Linear(fc_dims, fc_dims)
        self.v = nn.Linear(fc_dims,1)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)
    
    def forward(self, state):
        _, h = self.seq_encoder(state)
        x = h[0]
        
        x_pi = F.elu(self.fc_pi(x))
        pi = self.pi(x_pi)

        x_v = F.elu(self.fc_v(x))
        v = self.v(x_v)
        return pi,v

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)
    
    def load_checkpoint(self):
        print('... loading checkpoint ...')
        with T.no_grad():
            self.load_state_dict(T.load(self.checkpoint_file, weights_only=True))

class Agent():
    def __init__(self, lr, input_dims, n_actions=3,gamma=0.99, fc_dims=256,\
                 batch_size=128, mem_size=int(1e6), eval_mode=False, run_name='ac'):
        self.gamma = gamma
        self.batch_size = batch_size
        self.run_name = run_name
        
        if eval_mode == False:
            self.memory = ReplayBuffer(mem_size, input_dims)
        
        self.actor_critic = ActorCriticNetwork(lr, input_dims, fc_dims, n_actions, name=run_name)
    
    def remember(self, state, prob, reward, state_, done):
        self.memory.store_transition(state, prob, reward, state_, done)
        
    def choose_action(self, obs):
        state = T.tensor(np.array([obs]),dtype=T.float32).to(self.actor_critic.device)

        probs, _ = self.actor_critic(state)
        probs = F.softmax(probs, dim=-1)
        action_probs = Categorical(probs)
        action = action_probs.sample()
        log_probs = action_probs.log_prob(action)

        return action.item(), log_probs.item()

    def learn(self):

        if self.memory.mem_cntr < self.batch_size:
            return
        
        states, log_probs, rewards, new_states, dones = self.memory.sample_buffer(self.batch_size)

        states = T.tensor(states).to(self.actor_critic.device)
        log_probs = T.tensor(log_probs).to(self.actor_critic.device)
        rewards = T.tensor(rewards).to(self.actor_critic.device)
        dones = T.tensor(dones).to(self.actor_critic.device)
        states_ = T.tensor(new_states).to(self.actor_critic.device)

        self.actor_critic.optimizer.zero_grad()
        _, critic_value = self.actor_critic.forward(states)
        _, critic_value_ = self.actor_critic.forward(states_)

        critic_value = critic_value.view(-1)
        critic_value_ = critic_value_.view(-1)

        delta = rewards.view(-1) + self.gamma*critic_value_*(1-dones.int())
        pi_loss = -T.mean(log_probs.view(-1)*(delta-critic_value))
        v_loss = F.mse_loss(delta, critic_value)
        (pi_loss + v_loss).backward()
        nn.utils.clip_grad_norm_(self.actor_critic.parameters(), 0.5)
        self.actor_critic.optimizer.step()

        return pi_loss.item(), v_loss.item()

    def save_model(self):
        self.actor_critic.save_checkpoint()
    
    def load_model(self):
        self.actor_critic.load_checkpoint()



