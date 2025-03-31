import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn

class OUActionNoise(object):
    def __init__(self, mu, sigma=0.15, theta=.2, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'Ornstein-Uhlenbeck Action-noise(mu={}, sigma={})'.format(
                                                            self.mu, self.sigma)

class ReplayBuffer:
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
        self.action_memory = np.zeros((self.mem_size, n_actions), dtype=np.float32)
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


class ContinuousPolicy(nn.Module):
    def __init__ (self, alpha, input_dims, fc_dims, n_actions, name, chkpt_dir='tmp/maddpg'):
        super(ContinuousPolicy, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, name+'_cont_policy')
        self.fc1 = nn.Linear(*input_dims, fc_dims)
        self.bn1 = nn.LayerNorm(fc_dims)
        self.fc2 = nn.Linear(fc_dims, fc_dims)
        self.bn2 = nn.LayerNorm(fc_dims)
        self.pi = nn.Linear(fc_dims, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.to(self.device)
    
    def forward(self, state):
        x = F.elu(self.bn1(self.fc1(state)))
        x = F.elu(self.bn2(self.fc2(x)))
        x = torch.tanh(self.pi(x))
        return x
    
    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)
    
    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))

class DiscretePolicy(nn.Module):
    def __init__ (self, alpha, input_dims, fc_dims, n_actions, name, chkpt_dir='tmp/maddpg'):
        super(DiscretePolicy, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, name+'_disc_policy')
        self.fc1 = nn.Linear(*input_dims, fc_dims)
        self.bn1 = nn.LayerNorm(fc_dims)
        self.fc2 = nn.Linear(fc_dims, fc_dims)
        self.bn2 = nn.LayerNorm(fc_dims)
        self.pi = nn.Linear(fc_dims, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.to(self.device)
    
    def forward(self, state):
        x = F.elu(self.bn1(self.fc1(state)))
        x = F.elu(self.bn2(self.fc2(x)))
        x = F.softmax(self.pi(x),dim=-1)
        return x
    
    def save_checkpoint(self):
        
        torch.save(self.state_dict(), self.checkpoint_file)
    
    def load_checkpoint(self):
        
        self.load_state_dict(torch.load(self.checkpoint_file))

class Critic(nn.Module):
    def __init__(self, beta, input_dims, fc_dims, n_actions, name, chkpt_dir='tmp/maddpg'):
        super(Critic, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, name+'_critic')
        self.fc1 = nn.Linear(input_dims, fc_dims)
        self.bn1 = nn.LayerNorm(fc_dims)
        self.fc2 = nn.Linear(fc_dims, fc_dims)
        self.bn2 = nn.LayerNorm(fc_dims)
        self.q = nn.Linear(fc_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.to(self.device)
    
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.elu(self.bn1(self.fc1(x)))
        x = F.elu(self.bn2(self.fc2(x)))
        x = self.q(x)
        return x
    
    def save_checkpoint(self):
        
        torch.save(self.state_dict(), self.checkpoint_file)
    
    def load_checkpoint(self):
        
        self.load_state_dict(torch.load(self.checkpoint_file))

class ContinuousAgent:
    def __init__(self, alpha, beta, input_dims, critic_dims, tau=1e-3, gamma=0.99, n_actions=2, fc_dims=256):
        self.gamma = gamma
        self.tau = tau
        self.n_actions = n_actions

        self.actor = ContinuousPolicy(alpha, input_dims, fc_dims, n_actions, name='actor')
        self.critic_1 = Critic(beta, critic_dims, fc_dims, n_actions, name='critic_1')
        self.critic_2 = Critic(beta, critic_dims, fc_dims, n_actions, name='critic_2')

        self.target_actor = ContinuousPolicy(alpha, input_dims, fc_dims, n_actions, name='target_actor')
        self.target_critic_1 = Critic(beta, critic_dims, fc_dims, n_actions, name='target_critic_1')
        self.target_critic_2 = Critic(beta, critic_dims, fc_dims, n_actions, name='target_critic_2')

        # self.noise = OUActionNoise(mu=np.zeros(n_actions))

        self.update_network_parameters(tau=1)
    
    def choose_action(self, state, deterministic=False):
        state = torch.tensor(np.array([state]), dtype=torch.float).to(self.actor.device)
        mu = self.actor(state)
        if deterministic:
            return mu.cpu().detach().numpy()[0]
        else:
            noise = np.random.normal(loc = 0, scale=0.15, size=self.n_actions)#self.noise()
            action = mu.cpu().detach().numpy()[0] + noise
            return action
    
    
    def save_models(self):
        print('... saving models ...')
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()
        self.target_critic_1.save_checkpoint()
        self.target_critic_2.save_checkpoint()
    
    def load_models(self):
        print('... loading models ...')
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.critic_2.load_checkpoint()
        self.target_critic_1.load_checkpoint()
        self.target_critic_2.load_checkpoint()
    
    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        actor_params = self.actor.named_parameters()
        critic_1_params = self.critic_1.named_parameters()
        critic_2_params = self.critic_2.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_1_params = self.target_critic_1.named_parameters()
        target_critic_2_params = self.target_critic_2.named_parameters()

        critic_1 = dict(critic_1_params)
        critic_2 = dict(critic_2_params)
        actor = dict(actor_params)
        target_actor = dict(target_actor_params)
        target_critic_1 = dict(target_critic_1_params)
        target_critic_2 = dict(target_critic_2_params)

        for name in critic_1:
            critic_1[name] = tau*critic_1[name].clone() + \
                    (1-tau)*target_critic_1[name].clone()

        for name in critic_2:
            critic_2[name] = tau*critic_2[name].clone() + \
                    (1-tau)*target_critic_2[name].clone()

        for name in actor:
            actor[name] = tau*actor[name].clone() + \
                    (1-tau)*target_actor[name].clone()

        self.target_critic_1.load_state_dict(critic_1)
        self.target_critic_2.load_state_dict(critic_2)
        self.target_actor.load_state_dict(actor)
            

class DiscreteAgent:
    def __init__(self, alpha, beta, input_dims, critic_dims, tau=1e-3, gamma=0.99, n_actions=3, fc_dims=256):
        self.gamma = gamma
        self.tau = tau
        self.n_actions = n_actions

        self.actor = DiscretePolicy(alpha, input_dims, fc_dims, n_actions, name='actor')
        self.critic_1 = Critic(beta, critic_dims, fc_dims, n_actions, name='critic_1')
        self.critic_2 = Critic(beta, critic_dims, fc_dims, n_actions, name='critic_2')

        self.target_actor = DiscretePolicy(alpha, input_dims, fc_dims, n_actions, name='target_actor')
        self.target_critic_1 = Critic(beta, critic_dims, fc_dims, n_actions, name='target_critic_1')
        self.target_critic_2 = Critic(beta, critic_dims, fc_dims, n_actions, name='target_critic_2')

        self.update_network_parameters(tau=1)
    
    def choose_action(self, state, deterministic=False):
        state = torch.tensor(np.array([state]), dtype=torch.float).to(self.actor.device)
        probs = self.actor(state)
        if deterministic:
            return torch.argmax(probs).cpu().detach().numpy()
        else:
            action = probs + torch.rand(probs.shape).to(self.actor.device)
            return action.cpu().detach().numpy()[0]
    
    def save_models(self):
        print('... saving models ...')
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()
        self.target_critic_1.save_checkpoint()
        self.target_critic_2.save_checkpoint()
    
    def load_models(self):
        print('... loading models ...')
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.critic_2.load_checkpoint()
        self.target_critic_1.load_checkpoint()
        self.target_critic_2.load_checkpoint()
    
    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        actor_params = self.actor.named_parameters()
        critic_1_params = self.critic_1.named_parameters()
        critic_2_params = self.critic_2.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_1_params = self.target_critic_1.named_parameters()
        target_critic_2_params = self.target_critic_2.named_parameters()

        critic_1 = dict(critic_1_params)
        critic_2 = dict(critic_2_params)
        actor = dict(actor_params)
        target_actor = dict(target_actor_params)
        target_critic_1 = dict(target_critic_1_params)
        target_critic_2 = dict(target_critic_2_params)

        for name in critic_1:
            critic_1[name] = tau*critic_1[name].clone() + \
                    (1-tau)*target_critic_1[name].clone()

        for name in critic_2:
            critic_2[name] = tau*critic_2[name].clone() + \
                    (1-tau)*target_critic_2[name].clone()

        for name in actor:
            actor[name] = tau*actor[name].clone() + \
                    (1-tau)*target_actor[name].clone()

        self.target_critic_1.load_state_dict(critic_1)
        self.target_critic_2.load_state_dict(critic_2)
        self.target_actor.load_state_dict(actor)


class MATD3PG:
    def __init__(self, alpha, beta, input_dims, tau=5e-3, gamma=0.99, cont_n_actions=1, disc_n_actions=3,\
          fc_dims=256, batch_size=128, max_size=int(1e6), update_interval=2, warmup=5000, noise = 0.15):
        
        self.update_interval = update_interval
        critic_dims = input_dims[0] + cont_n_actions + disc_n_actions
        self.agents = [ContinuousAgent(alpha, beta, input_dims,critic_dims, tau, gamma, cont_n_actions, fc_dims),
                        DiscreteAgent(alpha, beta, input_dims,critic_dims, tau, gamma, disc_n_actions, fc_dims)]
        
        self.memory = ReplayBuffer(max_size, input_dims, cont_n_actions+disc_n_actions)
        self.batch_size = batch_size
        self.gamma = gamma
        self.warmup = warmup
        self.noise = noise
    
    def reset_action_noise(self):
        self.agents[0].noise.reset()        
    
    def choose_action(self, state, deterministic=False):
        actions = []
        if self.memory.mem_cntr < self.warmup:
            # output random actions
            action_1 = np.random.uniform(-1, 1, size=(1,))
            action_2 = np.random.uniform(-1, 1, size=(3,))
            actions = [action_1, action_2]
        else:
            for agent in self.agents:
                actions.append(agent.choose_action(state, deterministic))
        return actions
    
    def remember(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)
    
    def save_models(self):
        for agent in self.agents:
            agent.save_models()
    
    def load_models(self):
        for agent in self.agents:
            agent.load_models()
    
    def learn(self, step):

        if self.memory.mem_cntr < self.batch_size:
            return

        states, actions, rewards, states_, dones = self.memory.sample_buffer(self.batch_size)
        states = torch.tensor(states, dtype=torch.float).to(self.agents[0].actor.device)
        states_ = torch.tensor(states_, dtype=torch.float).to(self.agents[0].actor.device)
        actions = torch.tensor(actions, dtype=torch.float).to(self.agents[0].actor.device)
        rewards = torch.tensor(rewards, dtype=torch.float).to(self.agents[0].actor.device)
        dones = torch.tensor(dones).to(self.agents[0].actor.device)

        all_agent_new_actions = []

        for idx, agent in enumerate(self.agents):
            new_pi = agent.target_actor.forward(states_) + torch.clamp(torch.normal(0, 0.2, size=agent.target_actor.forward(states_).shape, device=agent.actor.device), -0.5, 0.5)
            if idx == 0:
                new_pi = torch.clamp(new_pi, -1, 1)
            all_agent_new_actions.append(new_pi)

        all_agent_new_actions = torch.cat(all_agent_new_actions, dim=1)

        action_cntr = 0

        al,cl=[],[]

        for agent_idx, agent in enumerate(self.agents):
            
            with torch.no_grad():
                q1_ = agent.target_critic_1.forward(states_, all_agent_new_actions).view(-1)
                q2_ = agent.target_critic_2.forward(states_, all_agent_new_actions).view(-1)
                q_ = torch.min(q1_, q2_)
                target = rewards + self.gamma * q_ * (1-dones.int())
            
            # critic update
            agent.critic_1.optimizer.zero_grad()
            agent.critic_2.optimizer.zero_grad()
            q1 = agent.critic_1.forward(states, actions).view(-1)
            q2 = agent.critic_2.forward(states, actions).view(-1)
            q1_loss = F.mse_loss(q1, target)
            q2_loss = F.mse_loss(q2, target)
            critic_loss = 0.5*(q1_loss + q2_loss)
            critic_loss.backward(retain_graph=True)
            nn.utils.clip_grad_norm_(agent.critic_1.parameters(), 0.5)
            nn.utils.clip_grad_norm_(agent.critic_2.parameters(), 0.5)
            cl.append(critic_loss.item())
            agent.critic_1.optimizer.step()
            agent.critic_2.optimizer.step()

            if step % self.update_interval == 0:
                # actor update
                oa = actions.clone()
                oa[:, action_cntr:action_cntr+agent.n_actions] = agent.actor.forward(states)
                action_cntr += agent.n_actions
                actor_loss = -torch.mean(agent.critic_1.forward(states, oa).flatten())
                agent.actor.optimizer.zero_grad()
                actor_loss.backward(retain_graph=True)
                nn.utils.clip_grad_norm_(agent.actor.parameters(), 0.5)
                al.append(actor_loss.item())
                agent.actor.optimizer.step()
        
                agent.update_network_parameters()
        
        return al, cl
