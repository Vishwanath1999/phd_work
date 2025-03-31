import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

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

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cuda:1')
        self.to(self.device)

    def forward(self, observation):
        state = T.Tensor(observation).to(self.device)
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class ActorCriticNetwork(nn.Module):
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims,
                 n_actions):
        super(ActorCriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.pi = nn.Linear(self.fc2_dims, n_actions)
        self.v = nn.Linear(self.fc2_dims, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, observation):
        state = T.Tensor(observation).to(self.device)
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        pi = self.pi(x)
        v = self.v(x)
        return (pi, v)

class Agent(object):
    """ Agent class for use with separate actor and critic networks.
        This is appropriate for very simple environments, such as the mountaincar
    """
    def __init__(self, alpha, beta, input_dims, gamma=0.99,
                 layer1_size=256, layer2_size=256, n_actions=2):
        self.gamma = gamma
        self.actor = GenericNetwork(alpha, input_dims, layer1_size,
                                    layer2_size, n_actions=n_actions)
        self.critic = GenericNetwork(beta, input_dims, layer1_size,
                                     layer2_size, n_actions=1)
        self.log_probs = None

    def choose_action(self, observation):
        obs = self.actor.forward(observation)
        # print(obs.shape)
        probabilities = F.softmax(obs, dim=-1)
        action_probs = T.distributions.Categorical(probabilities)
        action = action_probs.sample()
        self.log_probs = action_probs.log_prob(action)

        return action.item()

    def learn(self, state, reward, new_state, done):
        
        critic_value_ = self.critic.forward(new_state)
        critic_value = self.critic.forward(state)
        reward = T.tensor(reward, dtype=T.float).to(self.actor.device)

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

        self.actor.optimizer.zero_grad()
        actor_loss = -self.log_probs * delta.detach()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1)
        self.actor.optimizer.step()

        

class NewAgent(object):
    """ Agent class for use with a single actor critic network that shares
        the lowest layers. For use with more complex environments such as
        the discrete lunar lander
    """
    def __init__(self, alpha, input_dims, gamma=0.99,
                 layer1_size=256, layer2_size=256, n_actions=2):
        self.gamma = gamma
        self.actor_critic = ActorCriticNetwork(alpha, input_dims, layer1_size,
                                    layer2_size, n_actions=n_actions)

        self.log_probs = None

    def choose_action(self, observation):
        probabilities, _ = self.actor_critic.forward(observation)
        probabilities = F.softmax(probabilities,dim=0)
        action_probs = T.distributions.Categorical(probabilities)
        action = action_probs.sample()
        log_probs = action_probs.log_prob(action)
        self.log_probs = log_probs

        return action.item()

    def learn(self, state, reward, new_state, done):
        self.actor_critic.optimizer.zero_grad()

        _, critic_value_ = self.actor_critic.forward(new_state)
        _, critic_value = self.actor_critic.forward(state)
        reward = T.tensor(reward, dtype=T.float).to(self.actor_critic.device)

        delta = reward + self.gamma*critic_value_*(1-int(done)) - critic_value

        # actor_loss = -self.log_probs * delta
        # critic_loss = delta**2

        loss = delta**2 - self.log_probs*delta 

        loss.backward(retain_graph=True)

        # Gradient clipping
        nn.utils.clip_grad_norm_(self.actor_critic.parameters(), max_norm=1)

        self.actor_critic.optimizer.step()
    
    