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

    
class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, fc_dims=128, n_actions=1, max_action=1, name='cont_actor', chkpt_dir='tmp/masac'):
        super(ActorNetwork, self).__init__()
        self.checkpoint_file = os.path.join(chkpt_dir, name+'_sac')

        self.seq_encoder = nn.GRU(input_size=input_dims[1], hidden_size=fc_dims, batch_first=True)
        self.mu = nn.Linear(fc_dims, n_actions)
        self.sigma = nn.Linear(fc_dims, n_actions)

        self.max_sigma = 2
        self.min_sigma = -20

        self.max_action = max_action

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state):
        _, h = self.seq_encoder(state)
        mu = self.mu(h[0])
        sigma = self.sigma(h[0])

        sigma = T.clamp(sigma, min=self.min_sigma, max=self.max_sigma).exp()

        return mu, sigma

    def sample_action(self, state, reparam=False, deterministic=True):
        mu, sigma = self.forward(state)
        
        if deterministic:
            return T.tanh(mu), None
        else:
            pi_dist = Normal(mu, sigma)
            pi_action = pi_dist.rsample() if reparam else pi_dist.sample()
            log_probs = pi_dist.log_prob(pi_action).sum(axis=1)
            corr_fact = (2*(np.log(2) - pi_action - F.softplus(-2*pi_action))).sum(axis=1)
            log_probs -= corr_fact
            pi_action = T.tanh(pi_action) * self.max_action
            return pi_action, log_probs
    
    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)
    
    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file, weights_only=True))
    

class  CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, fc_dims=128, n_actions=1, name='critic', chkpt_dir='tmp/masac'):
        super(CriticNetwork, self).__init__()
        self.checkpoint_file = os.path.join(chkpt_dir, name+'_sac')

        self.seq_encoder = nn.GRU(input_size=input_dims[1], hidden_size=fc_dims, batch_first=True)
        
        self.fc = nn.Linear(fc_dims + n_actions, fc_dims)
        self.q1 = nn.Linear(fc_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state, action):
        _, h = self.seq_encoder(state)
        q = T.cat([h[0], action], dim=1)
        q = F.elu(self.fc(q))
        q1 = self.q1(q)
        return q1

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)
    
    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file, weights_only=True))

class SACAgent:
    def __init__(self, alpha, beta, input_dims, n_actions, max_action=1, gamma=0.99, tau=0.005,
                  max_size=1000000, fc_dims=256, name='sac',
                 chkpt_dir='tmp/masac', eval_mode=False):
        self.gamma = gamma
        self.tau = tau
        if not eval_mode:
            self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.learn_step_counter = 0
        self.max_action = max_action

        self.actor = ActorNetwork(alpha, input_dims, fc_dims=fc_dims,
                                            n_actions=n_actions, max_action=max_action,
                                            name=name+'_actor', chkpt_dir=chkpt_dir)

        self.critic_1 = CriticNetwork(beta, input_dims, fc_dims=fc_dims,
                                      n_actions=n_actions, name=name+'_critic_1', chkpt_dir=chkpt_dir)

        self.critic_2 = CriticNetwork(beta, input_dims, fc_dims=fc_dims,
                                      n_actions=n_actions, name=name+'_critic_2', chkpt_dir=chkpt_dir)

        self.target_critic_1 = CriticNetwork(beta, input_dims, fc_dims=fc_dims,
                                             n_actions=n_actions, name=name+'_target_critic_1', chkpt_dir=chkpt_dir)

        self.target_critic_2 = CriticNetwork(beta, input_dims, fc_dims=fc_dims,
                                             n_actions=n_actions, name=name+'_target_critic_2', chkpt_dir=chkpt_dir)

        self.update_network_parameters(tau=1)

        self.target_ent_coef = -np.prod(n_actions)
        self.log_ent_coef = T.log(T.ones(1,device=self.actor.device)).requires_grad_()
        self.ent_coef_optimizer = optim.Adam([self.log_ent_coef], lr=alpha)

        self.ent_coef_max, self.ent_coef_min = 1, 1e-4

    def choose_action(self, state, deterministic=False):
        action = self.actor.sample_action(state, deterministic)[0]
        return action.cpu().detach().numpy()[0]

    def store_transition(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)
    
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
    
    def save_models(self):
        print('... saving models ...')
        self.actor.save_checkpoint()
    
    def load_models(self):
        print('... loading models ...')
        self.actor.load_checkpoint()
        # freeze the weights
        for param in self.actor.parameters():
            param.requires_grad = False

class MASAC:
    def __init__(self, alpha, beta, input_dims, n_actions, gamma=0.99, chkpt_dir='tmp/masac',
                 max_action=1, tau=0.005, batch_size=64, mem_size=1000000, name='masac',
                 fc_dims=256, eval_mode=False):
        
        self.power_agent = SACAgent(alpha, beta, input_dims, n_actions[0], max_action=max_action,
                            gamma=gamma, tau=tau, max_size=mem_size, fc_dims=fc_dims, name=name+'_power',
                            chkpt_dir=chkpt_dir, eval_mode=eval_mode)
       
        self.det_agent = SACAgent(alpha, beta, input_dims, n_actions[1], max_action=max_action,
                            gamma=gamma, tau=tau, max_size=mem_size, fc_dims=fc_dims, name=name+'_detuning',
                            chkpt_dir=chkpt_dir, eval_mode=eval_mode)
       
        self.agents = [self.power_agent, self.det_agent]

        self.gamma = gamma
        self.batch_size = batch_size
        
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.det_memory_ready = False
        self.power_memory_ready = False
    

    def choose_action(self, state, deterministic=False):
        actions = []
        state = T.tensor(np.array([state]), dtype=T.float).to(self.device)
        for agent in self.agents:
            action = agent.choose_action(state, deterministic)
            actions.append(action)
        return np.array(actions).flatten()
    
    def remember_power(self, state, action, reward, state_, done):
        self.power_agent.store_transition(state, action, reward, state_, done)
        if self.power_agent.memory.mem_cntr > 4*self.batch_size:
            self.power_memory_ready = True
    
    def remember_detuning(self, state, action, reward, state_, done):
        self.det_agent.store_transition(state, action, reward, state_, done)
        if self.det_agent.memory.mem_cntr > 4*self.batch_size:
            self.det_memory_ready = True
    
    def save_models(self):
        for agent in self.agents:
            agent.save_models()
        print('... saving models ...')
    
    def load_models(self):
        for agent in self.agents:
            agent.load_models()
        print('... loading models ...')
    
    def learn(self):

        if not self.det_memory_ready:
            return
        
        cl,al,el,ec = [],[],[],[]
        
        # sample from the memory and learn
        det_states, det_actions, det_rewards, det_states_, det_dones = self.det_agent.memory.sample_buffer(self.batch_size)

        det_states = T.tensor(det_states, dtype=T.float).to(self.device)
        det_actions = T.tensor(det_actions, dtype=T.float).to(self.device)
        det_rewards = T.tensor(det_rewards, dtype=T.float).to(self.device)
        det_states_ = T.tensor(det_states_, dtype=T.float).to(self.device)
        det_dones = T.tensor(det_dones).to(self.device)

        det_ent_coef = self.det_agent.log_ent_coef.detach().exp()
        det_ent_coef = T.clamp(det_ent_coef, min=self.det_agent.ent_coef_min, max=self.det_agent.ent_coef_max)

        ec.append(det_ent_coef.item())

        with T.no_grad():
            det_actions_, det_log_probs_ = self.det_agent.actor.sample_action(det_states_)
            det_q1_ = self.det_agent.target_critic_1(det_states_, det_actions_)
            det_q2_ = self.det_agent.target_critic_2(det_states_, det_actions_)
            det_q_ = T.min(det_q1_, det_q2_).view(-1)
            target_det_q = det_rewards + (1-det_dones.int()) * self.gamma * (det_q_ - det_ent_coef * det_log_probs_)
        
        # update critic networks
        self.det_agent.critic_1.optimizer.zero_grad()
        self.det_agent.critic_2.optimizer.zero_grad()
        det_q1 = self.det_agent.critic_1(det_states, det_actions).view(-1)
        det_q2 = self.det_agent.critic_2(det_states, det_actions).view(-1)
        det_q_loss = 0.5 * (F.mse_loss(det_q1, target_det_q) + F.mse_loss(det_q2, target_det_q))
        det_q_loss.backward()
        # clip gradients
        nn.utils.clip_grad_norm_(self.det_agent.critic_1.parameters(), 0.5)
        nn.utils.clip_grad_norm_(self.det_agent.critic_2.parameters(), 0.5)
        self.det_agent.critic_1.optimizer.step()
        self.det_agent.critic_2.optimizer.step()

        cl.append(det_q_loss.item())

        if self.power_memory_ready:
            # update critic networks for power agent
            power_states, power_actions, power_rewards, power_states_, power_dones = self.power_agent.memory.sample_buffer(self.batch_size)

            power_states = T.tensor(power_states, dtype=T.float).to(self.device)
            power_actions = T.tensor(power_actions, dtype=T.float).to(self.device)
            power_rewards = T.tensor(power_rewards, dtype=T.float).to(self.device)
            power_states_ = T.tensor(power_states_, dtype=T.float).to(self.device)
            power_dones = T.tensor(power_dones).to(self.device)

            power_ent_coef = self.power_agent.log_ent_coef.detach().exp()
            power_ent_coef = T.clamp(power_ent_coef, min=self.power_agent.ent_coef_min, max=self.power_agent.ent_coef_max)
            ec.append(power_ent_coef.item())

            with T.no_grad():
                power_actions_, power_log_probs_ = self.power_agent.actor.sample_action(power_states_)
                power_q1_ = self.power_agent.target_critic_1(power_states_, power_actions_)
                power_q2_ = self.power_agent.target_critic_2(power_states_, power_actions_)
                power_q_ = T.min(power_q1_, power_q2_).view(-1)
                target_power_q = power_rewards + (1-power_dones.int()) * self.gamma * (power_q_ - power_ent_coef * power_log_probs_)
            
            # update critic networks
            self.power_agent.critic_1.optimizer.zero_grad()
            self.power_agent.critic_2.optimizer.zero_grad()
            power_q1 = self.power_agent.critic_1(power_states, power_actions).view(-1)
            power_q2 = self.power_agent.critic_2(power_states, power_actions).view(-1)
            power_q_loss = 0.5 * (F.mse_loss(power_q1, target_power_q) + F.mse_loss(power_q2, target_power_q))
            power_q_loss.backward()
            # clip gradients
            nn.utils.clip_grad_norm_(self.power_agent.critic_1.parameters(), 0.5)
            nn.utils.clip_grad_norm_(self.power_agent.critic_2.parameters(), 0.5)
            self.power_agent.critic_1.optimizer.step()
            self.power_agent.critic_2.optimizer.step()
            cl.append(power_q_loss.item())
        else:
            cl.append(0)
            ec.append(0)
        
        # update actor networks
        self.det_agent.actor.optimizer.zero_grad()
        det_actions, det_log_probs = self.det_agent.actor.sample_action(det_states, reparam=True)
        det_q1 = self.det_agent.critic_1(det_states, det_actions)
        det_q2 = self.det_agent.critic_2(det_states, det_actions)
        det_q = T.min(det_q1, det_q2).view(-1)
        det_actor_loss = (det_ent_coef * det_log_probs - det_q).mean()
        det_actor_loss.backward()
        # clip gradients
        nn.utils.clip_grad_norm_(self.det_agent.actor.parameters(), 0.5)
        self.det_agent.actor.optimizer.step()

        al.append(det_actor_loss.item())

        if self.power_memory_ready:
            self.power_agent.actor.optimizer.zero_grad()
            power_actions, power_log_probs = self.power_agent.actor.sample_action(power_states, reparam=True)
            power_q1 = self.power_agent.critic_1(power_states, power_actions)
            power_q2 = self.power_agent.critic_2(power_states, power_actions)
            power_q = T.min(power_q1, power_q2).view(-1)
            power_actor_loss = (power_ent_coef * power_log_probs - power_q).mean()
            power_actor_loss.backward()
            # clip gradients
            nn.utils.clip_grad_norm_(self.power_agent.actor.parameters(), 0.5)
            self.power_agent.actor.optimizer.step()
            al.append(power_actor_loss.item())
        else:
            al.append(0)
        
        # update target networks
        self.power_agent.update_network_parameters()
        self.det_agent.update_network_parameters()

        # update entropy coefficient
        det_ent_coef_loss = -(self.det_agent.log_ent_coef * (det_log_probs + self.det_agent.target_ent_coef).detach()).mean()
        self.det_agent.ent_coef_optimizer.zero_grad()
        det_ent_coef_loss.backward()
        self.det_agent.ent_coef_optimizer.step()
        el.append(det_ent_coef_loss.item())

        if self.power_memory_ready:
            power_ent_coef_loss = -(self.power_agent.log_ent_coef * (power_log_probs + self.power_agent.target_ent_coef).detach()).mean()
            self.power_agent.ent_coef_optimizer.zero_grad()
            power_ent_coef_loss.backward()
            self.power_agent.ent_coef_optimizer.step()
            el.append(power_ent_coef_loss.item())
        else:
            el.append(0)

        return cl, al, el, ec
        