# Code is heavily inspired by Morvan Zhou's code. Please check out
# his work at github.com/MorvanZhou/pytorch-A3C
import torch as T
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np

class SharedAdam(T.optim.Adam):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.99), eps=1e-8, weight_decay=0):
        super(SharedAdam, self).__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                
                # Ensure step is initialized as a singleton tensor
                if 'step' not in state or not isinstance(state['step'], T.Tensor):
                    state['step'] = T.tensor([0], dtype=T.float32, device=p.device)  # ✅ Ensure tensor format
                
                # Ensure optimizer states are stored as shared tensors
                state['exp_avg'] = T.zeros_like(p.data, memory_format=T.preserve_format)
                state['exp_avg_sq'] = T.zeros_like(p.data, memory_format=T.preserve_format)

                # Move to shared memory for multiprocessing
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()
                state['step'].share_memory_()  # ✅ Ensure shared memory compatibility

    def step(self, closure=None):
        """ Override step to correctly update state['step'] """
        loss = super().step(closure)
        
        # Manually increment the shared tensor
        for group in self.param_groups:
            for p in group['params']:
                if p in self.state:
                    self.state[p]['step'] += 1  # ✅ Correctly updating shared tensor
        
        return loss


class ActorCritic(nn.Module):
    def __init__(self, input_dims, n_actions, gamma=0.99):
        super(ActorCritic, self).__init__()

        self.gamma = gamma

        self.pi1 = nn.Linear(*input_dims, 128)
        self.v1 = nn.Linear(*input_dims, 128)
        self.pi = nn.Linear(128, n_actions)
        self.v = nn.Linear(128, 1)

        self.rewards = []
        self.actions = []
        self.states = []

    def remember(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    def clear_memory(self):
        self.states = []
        self.actions = []
        self.rewards = []

    def forward(self, state):
        pi1 = F.relu(self.pi1(state))
        v1 = F.relu(self.v1(state))

        pi = self.pi(pi1)
        v = self.v(v1)

        return pi, v

    def calc_R(self, done):
        # print(self.states)
        rewards = [r.item() for r in self.rewards]  # Convert each NumPy array to a float
        states = T.tensor(np.array(self.states), dtype=T.float)
        # rewards = T.tensor(rewards, dtype=T.float)
        _, v = self.forward(states)

        R = v[-1]*(1-int(done))

        batch_return = []
        for reward in rewards[::-1]:
            R = reward + self.gamma*R
            batch_return.append(R)
        batch_return.reverse()
        batch_return = T.tensor(batch_return, dtype=T.float)

        return batch_return

    def calc_loss(self, done):
        states = T.tensor(np.array(self.states), dtype=T.float)
        actions = T.tensor(self.actions, dtype=T.float)

        returns = self.calc_R(done)

        pi, values = self.forward(states)
        values = values.squeeze()
        critic_loss = (returns-values)**2

        probs = F.softmax(pi)
        dist = Categorical(probs)
        log_probs = dist.log_prob(actions)
        actor_loss = -log_probs*(returns-values)

        total_loss = (critic_loss + actor_loss).mean()
    
        return total_loss

    def choose_action(self, observation):
        state = T.tensor(np.array([observation]), dtype=T.float)
        pi, v = self.forward(state)
        probs = F.softmax(pi)
        dist = Categorical(probs)
        action = dist.sample().numpy()[0]

        return action

# class Agent(mp.Process):
#     def __init__(self, global_actor_critic, optimizer, input_dims, n_actions, 
#                 gamma, lr, name, global_ep_idx, env_id, desired_spectrum):
#         super(Agent, self).__init__()
#         self.local_actor_critic = ActorCritic(input_dims, n_actions, gamma)
#         self.global_actor_critic = global_actor_critic
#         self.name = 'w%02i' % name
#         self.episode_idx = global_ep_idx
#         self.env = RL_MRR_Env()
#         self.optimizer = optimizer
#         self.desired_spectrum = desired_spectrum

#     def run(self):
#         t_step = 1
#         while self.episode_idx.value < N_GAMES:
#             done = False
#             state, _, ecav = self.env.reset()
#             score = 0
#             self.local_actor_critic.clear_memory()
#             while not done:
#                 action = self.local_actor_critic.choose_action(ecav)
#                 # print('action:', action)
#                 state_, reward, done, _, ecav_ = self.env.step(state,action, self.desired_spectrum)
#                 score += reward
#                 self.local_actor_critic.remember(ecav, action, reward)
#                 if t_step % T_MAX == 0 or done:
#                     loss = self.local_actor_critic.calc_loss(done)
#                     self.optimizer.zero_grad()
#                     loss.backward()
#                     for local_param, global_param in zip(
#                             self.local_actor_critic.parameters(),
#                             self.global_actor_critic.parameters()):
#                         global_param._grad = local_param.grad
#                     self.optimizer.step()
#                     self.local_actor_critic.load_state_dict(
#                             self.global_actor_critic.state_dict())
#                     self.local_actor_critic.clear_memory()
#                 t_step += 1
#                 state = state_
#                 ecav = ecav_
#                 # acav = acav_
#             with self.episode_idx.get_lock():
#                 self.episode_idx.value += 1
#             print(self.name, 'episode ', self.episode_idx.value, 'reward %.1f' % score)

# if __name__ == '__main__':
#     lr = 1e-4
#     env_id = 'CartPole-v0'
#     n_actions = 2
#     input_dims = [4]
#     N_GAMES = 3000
#     T_MAX = 5
#     global_actor_critic = ActorCritic(input_dims, n_actions)
#     global_actor_critic.share_memory()
#     optim = SharedAdam(global_actor_critic.parameters(), lr=lr, 
#                         betas=(0.92, 0.999))
#     global_ep = mp.Value('i', 0)

#     workers = [Agent(global_actor_critic,
#                     optim,
#                     input_dims,
#                     n_actions,
#                     gamma=0.99,
#                     lr=lr,
#                     name=i,
#                     global_ep_idx=global_ep,
#                     env_id=env_id) for i in range(mp.cpu_count())]
#     [w.start() for w in workers]
#     [w.join() for w in workers]