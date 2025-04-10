import torch as T
import torch.nn as nn
import torch.nn.functional as F
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


class ActorNetwork(nn.Module):
    def __init__(self, alpha, n_actions, input_dims, fc_dims=256, name='actor', chkpt_dir='./tmp/softq'):
        super(ActorNetwork, self).__init__()
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)

        self.fc1 = nn.Linear(*input_dims, fc_dims)
        self.fc2 = nn.Linear(fc_dims, fc_dims)
        self.probs = nn.Linear(fc_dims, n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
    
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.probs(x), dim=-1)
        return x

    def sample_action(self, state, deterministic=False):
        probs = self.forward(state)

        if deterministic:
            action = T.argmax(probs,dim=-1).item()
            return action,None

        dist = Categorical(probs)
        # sample action
        action = dist.sample()
        action_probs = probs + 1e-8
        log_pi = T.log(action_probs)        
        return action, probs, log_pi

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)
    
    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, n_actions, fc_dims=256, name='critic', chkpt_dir='./tmp/softq'):
        super(CriticNetwork, self).__init__()
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)

        self.fc1 = nn.Linear(*input_dims, fc_dims)
        self.fc2 = nn.Linear(fc_dims, fc_dims)
        self.q = nn.Linear(fc_dims, n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
    
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        q = self.q(x)
        return q

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)
    
    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

class Agent:
    def __init__(self, alpha, beta, input_dims, n_actions, mem_size=int(1e6), batch_size=64,
                 gamma=0.99, tau=0.005,run_name='sac'):

        self.run_name = run_name
        self.gamma = gamma
        self.tau = tau
        self.n_actions = n_actions
        self.batch_size = batch_size

        self.memory = ReplayBuffer(mem_size, input_dims)

        self.actor = ActorNetwork(alpha, n_actions, input_dims, name=run_name+'_actor')

        self.critic_1 = CriticNetwork(beta, input_dims, n_actions, name=run_name+'_critic_1')
        self.critic_2 = CriticNetwork(beta, input_dims, n_actions, name=run_name+'_critic_2')

        self.target_critic_1 = CriticNetwork(beta, input_dims, n_actions, name=run_name+'_target_critic_1')
        self.target_critic_2 = CriticNetwork(beta, input_dims, n_actions, name=run_name+'_target_critic_2')

        self.update_network_parameters(tau=1)

        self.target_ent_coef = -0.98*np.log(1/n_actions)
        self.log_ent_coef = T.log(T.ones(1,device=self.actor.device)).requires_grad_(True)
        self.ent_coef_optim = T.optim.Adam([self.log_ent_coef],lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
    
    def choose_action(self, obs, deterministic=False):
        state = T.tensor(np.array([obs]), dtype=T.float).to(self.device)
        actions,_,_ = self.actor.sample_action(state, deterministic)
        return actions.item()

    def remember(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)

    def learn(self):
        state, action, reward, state_, done = \
            self.memory.sample_buffer(self.batch_size)
        state = T.tensor(state, dtype=T.float).to(self.device)
        action = T.tensor(action, dtype=T.float).to(self.device)
        reward = T.tensor(reward, dtype=T.float).to(self.device)
        state_ = T.tensor(state_, dtype=T.float).to(self.device)
        done = T.tensor(done).to(self.device)

        with T.no_grad():
            _, probs_, log_pi_ = self.actor.sample_action(state_)
            
            ent_coef = T.exp(self.log_ent_coef)

            q1_ = self.target_critic_1(state_)
            q2_ = self.target_critic_2(state_)
            q_ = T.min(q1_, q2_)
            q_next = probs_*(q_ - ent_coef*log_pi_)
            q_next = q_next.sum(dim=-1)

            target = reward + (1-done.int())*self.gamma*q_next
        
        q1 = self.critic_1(state).gather(1, action.long().unsqueeze(-1)).squeeze(-1)
        q2 = self.critic_2(state).gather(1, action.long().unsqueeze(-1)).squeeze(-1)
        
        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()
        critic_1_loss = F.mse_loss(q1, target)
        critic_2_loss = F.mse_loss(q2, target)
        critic_loss = critic_1_loss + critic_2_loss
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic_1.parameters(), 0.5)
        nn.utils.clip_grad_norm_(self.critic_2.parameters(), 0.5)
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        _, probs, log_pi = self.actor.sample_action(state)
        q1 = self.critic_1(state)
        q2 = self.critic_2(state)
        q = T.min(q1, q2)
        actor_loss = probs*(self.log_ent_coef.exp()*log_pi - q)
        actor_loss = actor_loss.sum(dim=-1).mean()
        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.actor.optimizer.step()

        log_action_probs = T.sum(log_pi*probs, dim=-1)

        self.ent_coef_optim.zero_grad()
        ent_coef_loss = -(self.log_ent_coef*(log_action_probs + self.target_ent_coef).detach()).mean()
        ent_coef_loss.backward()
        self.ent_coef_optim.step()

        self.update_network_parameters()

        return critic_loss.item(), actor_loss.item(), ent_coef_loss.item(), self.log_ent_coef.exp().item()

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau
    
        target_critic_1_params = self.target_critic_1.named_parameters()
        critic_1_params = self.critic_1.named_parameters()

        target_critic_2_params = self.target_critic_2.named_parameters()
        critic_2_params = self.critic_2.named_parameters()

        target_critic_1_state_dict = dict(target_critic_1_params)
        critic_1_state_dict = dict(critic_1_params)

        target_critic_2_state_dict = dict(target_critic_2_params)
        critic_2_state_dict = dict(critic_2_params)

        for name in critic_1_state_dict:
            critic_1_state_dict[name] = tau*critic_1_state_dict[name].clone() + \
                    (1-tau)*target_critic_1_state_dict[name].clone()

        for name in critic_2_state_dict:
            critic_2_state_dict[name] = tau*critic_2_state_dict[name].clone() + \
                    (1-tau)*target_critic_2_state_dict[name].clone()

        self.target_critic_1.load_state_dict(critic_1_state_dict)
        self.target_critic_2.load_state_dict(critic_2_state_dict)

    def save_model(self):
        print('... saving models ...')
        self.actor.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()
        self.target_critic_1.save_checkpoint()
        self.target_critic_2.save_checkpoint()
    
    def load_model(self):
        print('... loading models ...')
        self.actor.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.critic_2.load_checkpoint()
        self.target_critic_1.load_checkpoint()
        self.target_critic_2.load_checkpoint()