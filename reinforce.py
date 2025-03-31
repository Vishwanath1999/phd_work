import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class PolicyNetwork(nn.Module):
    def __init__(self, ALPHA, input_dims, fc1_dims, fc2_dims,
                 n_actions):
        super(PolicyNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=ALPHA)
        self.model_file = 'REINFORCE' + '_'+str(ALPHA)+'_'+str(fc1_dims)+'_'+str(fc2_dims)+'.pth'

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        print('device:', self.device)
        self.to(self.device)

    def forward(self, observation):
        state = T.Tensor(observation).to(self.device)
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class PolicyGradientAgent(object):
    def __init__(self, ALPHA, input_dims, GAMMA=0.99, n_actions=2,
                 layer1_size=256, layer2_size=256):
        self.gamma = GAMMA
        self.reward_memory = []
        self.action_memory = []
        self.policy = PolicyNetwork(ALPHA, input_dims, layer1_size, layer2_size,
                                    n_actions)

    def choose_action(self, observation):
        probabilities = F.softmax(self.policy.forward(observation))
        action_probs = T.distributions.Categorical(probabilities)
        action = action_probs.sample()
        log_probs = action_probs.log_prob(action)
        self.action_memory.append(log_probs)

        return action.item()

    def store_rewards(self,reward):
        self.reward_memory.append(reward)

    def learn(self):
        self.policy.optimizer.zero_grad()
        # Assumes only a single episode for reward_memory
        G = np.zeros_like(self.reward_memory, dtype=np.float64)
        for t in range(len(self.reward_memory)):
            G_sum = 0
            discount = 1
            for k in range(t, len(self.reward_memory)):
                G_sum += self.reward_memory[k] * discount
                discount *= self.gamma
            G[t] = G_sum
        mean = np.mean(G)
        std = np.std(G) if np.std(G) > 0 else 1
        G = (G - mean) / std

        G = T.tensor(G, dtype=T.float).to(self.policy.device)

        loss = 0
        for g, logprob in zip(G, self.action_memory):
            loss += -g * logprob

        loss.backward()
        self.policy.optimizer.step()

        self.action_memory = []
        self.reward_memory = []
    
    def learn2(self):
        self.policy.optimizer.zero_grad()
        
        # Convert rewards to tensor
        rewards = np.array(self.reward_memory, dtype=np.float64)
        n = len(rewards)

        # Compute returns G using vectorized approach (O(n) instead of O(n²))
        G = np.zeros(n, dtype=np.float64)
        G_sum = 0
        for t in reversed(range(n)):
            G_sum = rewards[t] + self.gamma * G_sum
            G[t] = G_sum
        
        # Normalize returns
        G = (G - G.mean()) / (G.std() + 1e-8)  # Adding epsilon to avoid div-by-zero

        # Convert to PyTorch tensor
        G = T.tensor(G, dtype=T.float32, device=self.policy.device)

        # Compute loss in a vectorized manner
        logprobs = T.stack(self.action_memory)  # Convert list to tensor
        loss = -T.sum(G * logprobs)  # Vectorized loss computation

        # Backpropagate
        loss.backward()
        # clip gradient
        nn.utils.clip_grad_norm_(self.policy.parameters(), 1)
        self.policy.optimizer.step()

        # Clear memory
        self.action_memory = []
        self.reward_memory = []
    
    # function to save and load model
    def save_model(self):
        T.save(self.policy.state_dict(), self.policy.model_file)
    
    def load_model(self):
        self.policy.load_state_dict(T.load(self.policy.model_file))