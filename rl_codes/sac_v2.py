import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, Beta
import os

class ReplayBuffer():
    def __init__(self, input_shape, n_actions, max_size=int(1e6)):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.n_actions = n_actions
        self.input_shape = input_shape
        self.state_memory = np.zeros((self.mem_size,*input_shape),dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size,*input_shape),dtype=np.float32)
        self.action_memory = np.zeros((self.mem_size,n_actions))
        self.reward_memory = np.zeros((self.mem_size,),dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)

    def store_transition(self,state,action,reward,state_,done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.new_state_memory[index] = state_
        self.terminal_memory[index] = done

        self.mem_cntr += 1
    
    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        dones = self.terminal_memory[batch]

        return states,actions,rewards,states_,dones


class Actor(nn.Module):
    def __init__(self, input_dim, lr=3e-4, fc_dim=128, output_dim=1, max_action=1, dist='normal', name='sac',chkpt_dir='./tmp/sac'):
        super(Actor, self).__init__()
        self.max_action = max_action
        self.dist = dist
        self.seq_encoder = nn.GRU(input_dim[1], fc_dim, batch_first=True)
        self.attn = nn.Linear(fc_dim, 1)
        self.fc1 = nn.Linear(fc_dim, fc_dim)
        self.mu_layer = nn.Linear(fc_dim, output_dim)
        self.log_std_layer = nn.Linear(fc_dim, output_dim)
        self.log_std_min = -20
        self.log_std_max = 2

        self.chkpt_dir = chkpt_dir
        self.chkpt_file = os.path.join(chkpt_dir, name+'_actor')

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        gru_out,_ = self.seq_encoder(state)
        attn_scores = self.attn(gru_out)
        attn_weights = T.softmax(attn_scores, dim=1)
        context = T.sum(attn_weights * gru_out, dim=1)
        x = F.relu(self.fc1(context))
        if self.dist == 'normal':
            mu = self.mu_layer(x)
            log_std = self.log_std_layer(x)
            std = T.clamp(log_std, self.log_std_min, self.log_std_max).exp()
        else:
            mu = 1 + F.softplus(self.mu_layer(x))
            std = 1 + F.softplus(self.log_std_layer(x))
        
        return mu, std
    
    def sample_action(self, state, reparam=False,deterministic=False):
        mu, std = self.forward(state)
        if deterministic:
            if self.dist == 'normal':
                return T.tanh(mu) * self.max_action, None
            else:
                mean = mu/(mu+std)
                return (2*mean - 1)*self.max_action, None
        
        if self.dist == 'normal':
            pi_dist = Normal(mu, std)
            pi_action = pi_dist.rsample() if reparam else pi_dist.sample()
            log_prob = pi_dist.log_prob(pi_action).sum(axis=1)
            corr_fact = (2*(np.log(2) - pi_action - F.softplus(-2*pi_action))).sum(axis=1)
            log_prob -= corr_fact
            pi_action = T.tanh(pi_action) * self.max_action
        else:
            pi_dist = Beta(mu, std)
            pi_action = pi_dist.rsample() if reparam else pi_dist.sample()
            log_prob = pi_dist.log_prob(pi_action).sum(axis=1)
            pi_action = (2*pi_action - 1)*self.max_action
        return pi_action, log_prob
    
    def save_checkpoint(self):
        T.save(self.state_dict(), self.chkpt_file)
    
    
    def load_checkpoint(self):
        if os.path.exists(self.chkpt_file):
            self.load_state_dict(T.load(self.chkpt_file, weights_only=True))
        else:
            print('No checkpoint found at {}'.format(self.chkpt_file))
    
    def save_jit(self, example_input=None, filename=None):
        """
        Save the model as a TorchScript JIT file.
        If example_input is provided, uses tracing; otherwise, uses scripting.
        """
        if filename is None:
            filename = self.chkpt_file + "_jit.pt"
        self.eval()
        if example_input is not None:
            # Trace the model (requires example input)
            scripted = T.jit.trace(self, example_input.to(self.device))
        else:
            # Script the model (no example input needed, but all control flow must be TorchScript compatible)
            scripted = T.jit.script(self)
        T.jit.save(scripted, filename)
        print(f"JIT model saved to {filename}")

    @staticmethod
    def load_jit(filename, device=None):
        """
        Load a TorchScript JIT model from file.
        Returns the loaded scripted model.
        """
        if device is None:
            device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        scripted = T.jit.load(filename, map_location=device)
        scripted.eval()
        print(f"JIT model loaded from {filename}")
        return scripted

class CriticNetwork(nn.Module):
    def __init__(self, input_dim, n_actions, lr=3e-4, fc_dim=128, name='sac', chkpt_dir='./tmp/sac'):
        super(CriticNetwork, self).__init__()
        
        self.seq_encoder = nn.GRU(input_dim[1], fc_dim, batch_first=True)
        self.attn = nn.Linear(fc_dim, 1)
        self.fc1 = nn.Linear(fc_dim+n_actions, fc_dim)
        self.q_layer = nn.Linear(fc_dim, 1)

        self.chkpt_dir = chkpt_dir
        self.chkpt_file = os.path.join(chkpt_dir, name)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, action):
        gru_out,_ = self.seq_encoder(state)
        attn_scores = self.attn(gru_out)
        attn_weights = T.softmax(attn_scores, dim=1)
        context = T.sum(attn_weights * gru_out, dim=1)
        x = T.cat([context, action], dim=1)
        x = F.relu(self.fc1(x))
        q_value = self.q_layer(x)
        return q_value

    def save_checkpoint(self):
        T.save(self.state_dict(), self.chkpt_file)
    
    def load_checkpoint(self):
        if os.path.exists(self.chkpt_file):
            self.load_state_dict(T.load(self.chkpt_file, weights_only=True))
        else:
            print('No checkpoint found at {}'.format(self.chkpt_file))


class SACAgent:
    def __init__(self, input_dim, n_actions, run_name, alpha=3e-4, beta=3e-4, gamma=0.99, tau=0.005, 
                 mem_size=int(1e6), batch_size=256, max_action=1, dist='normal', eval_mode=False):
        self.input_dim = input_dim
        self.n_actions = n_actions
        self.run_name = run_name
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.tau = tau
        self.mem_size = mem_size
        self.batch_size = batch_size
        self.max_action = max_action
        self.dist = dist

        self.actor = Actor(input_dim, lr=alpha, output_dim=n_actions, max_action=max_action, dist=dist,name=run_name+'_actor')
        self.critic_1 = CriticNetwork(input_dim, n_actions=n_actions, lr=beta, name=run_name+'_critic_1')
        self.critic_2 = CriticNetwork(input_dim, n_actions=n_actions, lr=beta, name=run_name+'_critic_2')
        
        self.target_critic_1 = CriticNetwork(input_dim, n_actions=n_actions, lr=beta, name=run_name+'_target_critic_1')
        self.target_critic_2 = CriticNetwork(input_dim, n_actions=n_actions, lr=beta, name=run_name+'_target_critic_2')

        self.update_network_parameters(tau=1)
        self.target_ent_coef = -np.prod(n_actions)
        self.log_ent_coef = T.log(T.ones(1,device=self.actor.device)).requires_grad_()
        self.ent_coef_optimizer = optim.Adam([self.log_ent_coef], lr=alpha)

        self.ent_coef_max, self.ent_coef_min = 1, 1e-4

        if eval_mode == False:
            self.memory = ReplayBuffer(input_dim, n_actions, max_size=mem_size)
    
    def choose_action(self, state, deterministic=False):
        state = T.tensor(np.array([state]), dtype=T.float).to(self.actor.device)
        action, _ = self.actor.sample_action(state, deterministic)
        return action.cpu().detach().numpy()[0]

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)
    
    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return
        
        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)
        state = T.tensor(state, dtype=T.float).to(self.actor.device)
        action = T.tensor(action, dtype=T.float).to(self.actor.device)
        reward = T.tensor(reward, dtype=T.float).to(self.actor.device)
        new_state = T.tensor(new_state, dtype=T.float).to(self.actor.device)
        done = T.tensor(done).to(self.actor.device)

        ent_coef = self.log_ent_coef.detach().exp()
        ent_coef = T.clamp(ent_coef, self.ent_coef_min, self.ent_coef_max)

        with T.no_grad():
            actions_, log_probs_ = self.actor.sample_action(new_state)
            q1_ = self.target_critic_1(new_state, actions_)
            q2_ = self.target_critic_2(new_state, actions_)
            critic_value_ = T.min(q1_, q2_).view(-1)
            target_value = reward + (1-done.int()) * self.gamma * (critic_value_ - ent_coef * log_probs_)
        
        # Update Critics
        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()
        q1 = self.critic_1(state, action).view(-1)
        q2 = self.critic_2(state, action).view(-1)
        critic_loss = 0.5*(F.mse_loss(q1, target_value) + F.mse_loss(q2, target_value))
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic_1.parameters(), 0.5)
        nn.utils.clip_grad_norm_(self.critic_2.parameters(), 0.5)
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        # Update Actor
        actions, log_probs = self.actor.sample_action(state, reparam=True)
        q1 = self.critic_1(state, actions)
        q2 = self.critic_2(state, actions)
        critic_value = T.min(q1, q2).view(-1)
        actor_loss = (ent_coef * log_probs - critic_value).mean()
        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.actor.optimizer.step()

        # Update Entropy Coefficient
        ent_coef_loss = -(self.log_ent_coef * (log_probs + self.target_ent_coef).detach()).mean()
        self.ent_coef_optimizer.zero_grad()
        ent_coef_loss.backward()
        self.ent_coef_optimizer.step()

        # Update Target Networks
        self.update_network_parameters()
        return critic_loss.item(), actor_loss.item(), ent_coef_loss.item(), ent_coef.item()

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
        