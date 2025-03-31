import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class RewardScaler:
    def __init__(self, alpha=0.01):  # alpha controls update speed
        self.mean = 0
        self.var = 1
        self.alpha = alpha  # Controls how fast we update mean/std

    def update(self, reward):
        self.mean = self.alpha * reward + (1 - self.alpha) * self.mean
        self.var = self.alpha * (reward - self.mean) ** 2 + (1 - self.alpha) * self.var

    def normalize(self, reward):
        return (reward - self.mean) / (self.var ** 0.5 + 1e-6)
    
    def reset(self):
        self.mean = 0
        self.var = 1

class GenericNetwork(nn.Module):
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims,
                 n_actions):
        super(GenericNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, observation):
        state = T.Tensor(observation).to(self.device)
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class ContinuousNetwork(nn.Module):
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims,
                 n_actions):
        super(ContinuousNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.mu = nn.Linear(self.fc2_dims, n_actions)
        self.sigma = nn.Linear(self.fc2_dims, n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, observation):
        state = T.Tensor(observation).to(self.device)
        x = F.tanh(self.fc1(state))
        x = F.tanh(self.fc2(x))
        mu = self.mu(x)
        sigma = self.sigma(x)
        sigma = T.clamp(sigma, min=1e-5, max=0.5).exp()
        return mu,sigma

class CombinedNW(nn.Module):
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims,
                 disc_n_actions, cont_n_actions):
        super(CombinedNW, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.disc_n_actions = disc_n_actions
        self.cont_n_actions = cont_n_actions
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.fc2_dims)

        self.disc_op = nn.Linear(self.fc2_dims, disc_n_actions)
        self.v = nn.Linear(self.fc2_dims, 1)

        self.cont_mu = nn.Linear(self.fc2_dims, cont_n_actions)
        self.cont_sigma = nn.Linear(self.fc2_dims, cont_n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, observation):
        state = T.Tensor(observation).to(self.device)
        x = F.tanh(self.fc1(state))
        x = F.tanh(self.fc2(x))
        x = F.tanh(self.fc3(x))
        
        disc_op = self.disc_op(x)
        v = self.v(x)
        
        cont_mu = self.cont_mu(x)
        cont_sigma = self.cont_sigma(x)
        cont_sigma = T.clamp(cont_sigma, min=1e-5, max=0.5).exp()
        return disc_op, v, cont_mu, cont_sigma

class Agent(object):
    """ Agent class for use with separate actor and critic networks.
        This is appropriate for very simple environments, such as the mountaincar
    """
    def __init__(self, alpha, beta, input_dims, gamma=0.99,
                 layer1_size=256, layer2_size=256, disc_n_actions=2, cont_n_actions=1):
        self.gamma = gamma
        self.disc_actor = GenericNetwork(alpha, input_dims, layer1_size,
                                    layer2_size, n_actions=disc_n_actions)
        self.cont_actor = ContinuousNetwork(alpha, input_dims, layer1_size,
                                    layer2_size, n_actions=cont_n_actions)
        self.critic = GenericNetwork(beta, input_dims, layer1_size,
                                     layer2_size, n_actions=1)
        self.disc_log_probs = None
        self.cont_log_probs = None

    # def choose_action(self, observation):
    #     obs = self.disc_actor.forward(observation)
    #     global_action = np.array([])
    #     # print(obs.shape)
    #     probabilities = F.softmax(obs, dim=-1)
    #     action_probs = T.distributions.Categorical(probabilities)
    #     action = action_probs.sample()
    #     global_action = np.append(global_action, action.item())
    #     self.disc_log_probs = action_probs.log_prob(action)

    #     mu,sigma = self.cont_actor.forward(observation)
    #     action_probs = T.distributions.Normal(mu, sigma)
    #     action = action_probs.sample()
    #     print(action)
    #     global_action = np.append(global_action, action.item())
    #     self.cont_log_probs = action_probs.log_prob(action)

    #     return global_action

    def choose_action(self, observation, deterministic=False):
        obs = self.disc_actor.forward(observation)
        global_action = np.array([])
        # Discrete action sampling
        probabilities = F.log_softmax(obs,dim=0)
        # probabilities = T.clamp(logits=probabilities, 1e-8, 1.0)
        action_probs = T.distributions.Categorical(logits=probabilities)
        action = action_probs.sample()
        global_action = np.append(global_action, action.item())
        self.disc_log_probs = action_probs.log_prob(action)

        # Continuous action sampling
        if deterministic:
            mu, _ = self.cont_actor.forward(observation)
            raw_action = mu
        else:
            mu, sigma = self.cont_actor.forward(observation)
            action_probs = T.distributions.Normal(mu, sigma)
            raw_action = action_probs.rsample()  # rsample() allows gradients to flow

        # Apply tanh squashing
        squashed_action = T.tanh(raw_action)
        a, b = 0.001, 0.5  # Define action bounds
        scaled_action = a + 0.5 * (squashed_action + 1) * (b - a)

        # Log probability correction
        log_prob_raw = action_probs.log_prob(raw_action).sum(dim=-1)  # Sum over action dims
        log_prob_correction = T.log(1 - squashed_action.pow(2) + 1e-6).sum(dim=-1)  # Log derivative
        corrected_log_prob = log_prob_raw - log_prob_correction  # Apply correction

        global_action = np.append(global_action, scaled_action.cpu().detach().numpy())  
        self.cont_log_probs = corrected_log_prob  # Store corrected log-prob

        return global_action


    def learn(self, state, reward, new_state, done):
        
        critic_value_ = self.critic.forward(new_state)
        critic_value = self.critic.forward(state)
        reward = T.tensor(reward, dtype=T.float).to(self.disc_actor.device)

        delta = reward + self.gamma*critic_value_*(1-int(done)) - critic_value

        
        # self.critic.optimizer.zero_grad()
        # self.actor.optimizer.zero_grad()
        # actor_loss = -self.log_probs * delta
        # critic_loss = delta**2
        # loss = actor_loss + critic_loss
        # loss.backward()
        # # gradient clipping
        # nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1)
        # nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1)
        # self.critic.optimizer.step()
        # self.actor.optimizer.step()
        
        self.critic.optimizer.zero_grad()
        critic_loss = delta**2
        critic_loss.backward(retain_graph=True)
        nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1)
        self.critic.optimizer.step()

        self.disc_actor.optimizer.zero_grad()
        disc_actor_loss = -self.disc_log_probs * delta.detach()
        disc_actor_loss.backward()
        nn.utils.clip_grad_norm_(self.disc_actor.parameters(), max_norm=0.5)
        self.disc_actor.optimizer.step()

        self.cont_actor.optimizer.zero_grad()
        cont_actor_loss = -self.cont_log_probs * delta.detach()
        cont_actor_loss.backward()
        nn.utils.clip_grad_norm_(self.cont_actor.parameters(), max_norm=1)
        self.cont_actor.optimizer.step()

        return disc_actor_loss.item(), cont_actor_loss.item(), critic_loss.item()


class NewAgent(object):
    """ Agent class for use with separate actor and critic networks.
        This is appropriate for very simple environments, such as the mountaincar
    """
    def __init__(self, alpha, input_dims, gamma=0.99,
                 layer1_size=256, layer2_size=256, disc_n_actions=2, cont_n_actions=1):
        self.gamma = gamma

        self.actor_critic = CombinedNW(alpha, input_dims, layer1_size, layer2_size, disc_n_actions, cont_n_actions)
        
        self.disc_log_probs = None
        self.cont_log_probs = None


    def choose_action(self, observation, deterministic=False):
        disc_op, _, cont_mu, cont_sigma = self.actor_critic.forward(observation)

        global_action = np.array([])
        # Discrete action sampling
        probabilities = F.log_softmax(disc_op,dim=0)
        # probabilities = T.clamp(logits=probabilities, 1e-8, 1.0)
        action_probs = T.distributions.Categorical(logits=probabilities)
        action = action_probs.sample()
        global_action = np.append(global_action, action.item())
        self.disc_log_probs = action_probs.log_prob(action)

        # Continuous action sampling
        if deterministic:
            raw_action = cont_mu
        else:
            action_probs = T.distributions.Normal(cont_mu, cont_sigma)
            raw_action = action_probs.rsample()  # rsample() allows gradients to flow

        # Apply tanh squashing
        squashed_action = T.tanh(raw_action)
        a, b = 0.001, 0.5  # Define action bounds
        scaled_action = a + 0.5 * (squashed_action + 1) * (b - a)

        # Log probability correction
        log_prob_raw = action_probs.log_prob(raw_action).sum(dim=-1)  # Sum over action dims
        log_prob_correction = T.log(1 - squashed_action.pow(2) + 1e-6).sum(dim=-1)  # Log derivative
        corrected_log_prob = log_prob_raw - log_prob_correction  # Apply correction

        global_action = np.append(global_action, scaled_action.cpu().detach().numpy())  
        self.cont_log_probs = corrected_log_prob  # Store corrected log-prob

        return global_action


    def learn(self, state, reward, new_state, done):
        
        _,critic_value_,_,_ = self.actor_critic.forward(new_state)
        _,critic_value,_,_ = self.actor_critic.forward(state)
        reward = T.tensor(reward, dtype=T.float).to(self.actor_critic.device)

        delta = reward + self.gamma*critic_value_*(1-int(done)) - critic_value

        
        # self.critic.optimizer.zero_grad()
        # self.actor.optimizer.zero_grad()
        # actor_loss = -self.log_probs * delta
        # critic_loss = delta**2
        # loss = actor_loss + critic_loss
        # loss.backward()
        # # gradient clipping
        # nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1)
        # nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1)
        # self.critic.optimizer.step()
        # self.actor.optimizer.step()
        
        self.actor_critic.optimizer.zero_grad()
        critic_loss = delta**2
        disc_actor_loss = -self.disc_log_probs * delta
        cont_actor_loss = -self.cont_log_probs * delta
        loss = critic_loss + disc_actor_loss + cont_actor_loss
        loss.backward()
        nn.utils.clip_grad_norm_(self.actor_critic.parameters(), max_norm=0.5)
        self.actor_critic.optimizer.step()

        # return disc_actor_loss.item(), cont_actor_loss.item(), critic_loss.item()