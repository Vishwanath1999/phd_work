import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os

class OUActionNoise(object):
    '''
    Ornstein-Uhlenbeck process.
    This is used to add noise to the actions taken by the agent in order to encourage exploration.
    The noise is generated using the Ornstein-Uhlenbeck process, which is a type of stochastic process.
    This process is particularly useful in reinforcement learning for continuous action spaces,
    as it helps to explore the action space more effectively by adding temporally correlated noise.
    '''
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
    '''
    Replay buffer for experience replay.
    This is used to store the experiences of the agent, which are then sampled to train the agent.
    The replay buffer stores the state, action, reward, next state, and done flag for each experience.
    '''
    def __init__(self, max_size, input_shape, n_actions_disc, n_actions_cont):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
        self.cont_action_memory = np.zeros((self.mem_size, n_actions_cont), dtype=np.float32)
        self.disc_action_memory = np.zeros((self.mem_size, n_actions_disc), dtype=np.float32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)

    def store_transition(self, state, cont_action, disc_action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.cont_action_memory[index] = cont_action
        self.disc_action_memory[index] = disc_action
        self.terminal_memory[index] = done

        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_memory[batch]
        cont_actions = self.cont_action_memory[batch]
        disc_actions = self.disc_action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        dones = self.terminal_memory[batch]

        return states, cont_actions, disc_actions, rewards, states_, dones



class ActorNetwork(nn.Module):
    '''
    Actor network for the DDPG algorithm.
    The actor network is used to predict the action to take in a given state.
    The actor network takes the state as input and outputs the continuous and discrete actions.
    '''
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims, n_actions_cont, n_actions_disc\
                 , name, use_grad_inv=False, chkpt_dir='tmp/ddpg'):
        super(ActorNetwork, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, name+'_ddpg')
        self.fc1 = nn.Linear(*input_dims, fc1_dims)
        self.bn1 = nn.LayerNorm(fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.bn2 = nn.LayerNorm(fc2_dims)
        self.cont_probs = nn.Linear(fc2_dims, n_actions_cont)
        self.disc_probs = nn.Linear(fc2_dims, n_actions_disc)

        self.pmax = 0.5
        self.pmin = 0.01
        self.use_grad_inv = use_grad_inv


        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.bn1(self.fc1(state)))
        x = F.relu(self.bn2(self.fc2(x)))

        if self.use_grad_inv:
            cont_probs = T.tanh(self.cont_probs(x))
            # rescale the continuous action to be between 0.5 and 0.01
            cont_probs = self.pmin + 0.5 * (cont_probs + 1) * (self.pmax-self.pmin)
        else:
            cont_probs = self.cont_probs(x)

        disc_probs = self.disc_probs(x)

        return cont_probs, disc_probs

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)
    
    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))


class CriticNetwork(nn.Module):
    '''
    Critic network for the DDPG algorithm.
    The critic network is used to evaluate the action taken by the actor network.
    The critic network takes the state and action as input and outputs the Q-value.
    '''
    def __init__(self, beta, input_dims, fc1_dims, fc2_dims, n_actions_cont, n_actions_disc\
                 , name, chkpt_dir='tmp/ddpg'):
        super(CriticNetwork, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, name+'_ddpg')
        self.fc1 = nn.Linear(input_dims[0]+n_actions_cont+n_actions_disc, fc1_dims)
        self.bn1 = nn.LayerNorm(fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.bn2 = nn.LayerNorm(fc2_dims)
        self.q = nn.Linear(fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state, cont_action, disc_action):
        x = T.cat([state, cont_action, disc_action], 1)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        q = self.q(x)

        return q

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)
    
    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

class Agent:

    def __init__(self, alpha, beta, input_dims, tau=0.001, gamma=0.99, n_actions_cont=1, n_actions_disc=3\
                 , max_size=1000000, layer1_size=400, layer2_size=300, batch_size=64,warmup=10000,\
                    pmax=0.5, pmin=0.01, final_eps=0.05, use_grad_inv=False):
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_dims, n_actions_disc, n_actions_cont)
        self.batch_size = batch_size

        self.n_actions_cont = n_actions_cont
        self.n_actions_disc = n_actions_disc

        self.use_grad_inv = use_grad_inv

        self.actor = ActorNetwork(alpha, input_dims, layer1_size, layer2_size, n_actions_cont, n_actions_disc, 'actor', use_grad_inv)
        self.critic_1 = CriticNetwork(beta, input_dims, layer1_size, layer2_size, n_actions_cont, n_actions_disc, 'critic_1')
        self.target_actor = ActorNetwork(alpha, input_dims, layer1_size, layer2_size, n_actions_cont, n_actions_disc, 'target_actor', use_grad_inv)
        self.target_critic_1 = CriticNetwork(beta, input_dims, layer1_size, layer2_size, n_actions_cont, n_actions_disc, 'target_critic_1')
        self.critic_2 = CriticNetwork(beta, input_dims, layer1_size, layer2_size, n_actions_cont, n_actions_disc, 'critic_2')
        self.target_critic_2 = CriticNetwork(beta, input_dims, layer1_size, layer2_size, n_actions_cont, n_actions_disc, 'target_critic_2')

        self.noise = OUActionNoise(mu=np.zeros(n_actions_cont))

        self.epsilon = 1.0
        self.warmup = warmup
        self.pmax = pmax
        self.pmin = pmin
        self.final_eps = final_eps
        self.eps_annealing = (self.epsilon - self.final_eps)/self.warmup

        self.update_network_parameters(tau=1)
    
    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        critic_1_params = self.critic_1.named_parameters()
        target_critic_1_params = self.target_critic_1.named_parameters()
        critic_2_params = self.critic_2.named_parameters()
        target_critic_2_params = self.target_critic_2.named_parameters()
        
        actor_params = self.actor.named_parameters()
        target_actor_params = self.target_actor.named_parameters()

        critic_1_state_dict = dict(critic_1_params)
        target_critic_1_state_dict = dict(target_critic_1_params)
        critic_2_state_dict = dict(critic_2_params)
        target_critic_2_state_dict = dict(target_critic_2_params)
        actor_state_dict = dict(actor_params)
        target_actor_state_dict = dict(target_actor_params)

        for name in critic_1_state_dict:
            critic_1_state_dict[name] = tau*critic_1_state_dict[name].clone() + \
                                      (1-tau)*target_critic_1_state_dict[name].clone()
        for name in critic_2_state_dict:
            critic_2_state_dict[name] = tau*critic_2_state_dict[name].clone() + \
                                      (1-tau)*target_critic_2_state_dict[name].clone()
        for name in actor_state_dict:
            actor_state_dict[name] = tau*actor_state_dict[name].clone() + \
                                    (1-tau)*target_actor_state_dict[name].clone()
        
        self.target_critic_1.load_state_dict(critic_1_state_dict)
        self.target_critic_2.load_state_dict(critic_2_state_dict)
        self.target_actor.load_state_dict(actor_state_dict)
    
    def choose_action(self, state, step, deterministic=False):
        
        # for the first warmup steps choose random actions. For continuous actions its uniformly 
        # sampled from pmax to pmin. for disc actions use epsilon greedy to choose actions and 
        # anneal epsilon to final_eps within warmup number of steps.    

        self.actor.eval()
        state = T.tensor([state], dtype=T.float).to(self.actor.device)
        cont, disc = self.actor.forward(state)

        if deterministic:
            disc_action = T.argmax(disc, dim=-1)
            action = T.cat([disc_action, cont[0]], dim=-1)
            action = action.cpu().detach().numpy().flatten()
            return action

        if step < self.warmup:
            cont = T.tensor(np.random.uniform(self.pmin, self.pmax, (self.n_actions_cont,1))).to(self.actor.device)

        # choose discrete actions using epsilon greedy
        if np.random.random() > self.epsilon:
            disc_action = T.argmax(disc, dim=-1)
        else:
            #    choose a raoom action from disc_action
            disc_action = T.randint(0, self.n_actions_disc, (1,)).to(self.actor.device)

        # anneal epsilon
        self.epsilon -= self.eps_annealing
        self.epsilon = max(self.epsilon, self.final_eps)
        
        action = T.cat([disc_action, cont[0]], dim=-1)
        action = action.cpu().detach().numpy().flatten()
        if step < self.warmup:
            action_noise = action
        else:
            action_noise = action + np.array([0,self.noise()[0]])
        return action_noise, cont.flatten().detach().cpu().numpy() , disc.flatten().detach().cpu().numpy()
        
    def remember(self, state, cont_action, disc_action, reward, new_state, done):
        self.memory.store_transition(state, cont_action, disc_action, reward, new_state, done)

    def learn(self,step=None):

        if self.memory.mem_cntr < self.batch_size:
            return

        states, cont_actions, disc_actions, rewards, states_, dones = \
                self.memory.sample_buffer(self.batch_size)

        states = T.tensor(states, dtype=T.float).to(self.actor.device)
        cont_actions = T.tensor(cont_actions, dtype=T.float).to(self.actor.device).requires_grad_()
        disc_actions = T.tensor(disc_actions, dtype=T.float).to(self.actor.device).requires_grad_()
        rewards = T.tensor(rewards, dtype=T.float).to(self.actor.device)
        states_ = T.tensor(states_, dtype=T.float).to(self.actor.device)
        dones = T.tensor(dones).to(self.actor.device)


        cont_actions_, disc_actions_ = self.target_actor.forward(states_)
        q1_ = self.target_critic_1.forward(states_, cont_actions_, disc_actions_).view(-1)
        q2_ = self.target_critic_2.forward(states_, cont_actions_, disc_actions_).view(-1)
        critic_value_ = T.min(q1_, q2_)

        q1 = self.critic_1.forward(states, cont_actions, disc_actions).view(-1)
        q2 = self.critic_2.forward(states, cont_actions, disc_actions).view(-1)

        target = rewards.view(-1) + self.gamma*critic_value_*(1-dones.int())
        # target = target

        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()
        q1_loss = F.mse_loss(target, q1)
        q2_loss = F.mse_loss(target, q2)
        critic_loss = q1_loss + q2_loss
        critic_loss.backward()
        cl = critic_loss.item()
        # nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        # if step is not None and step<self.warmup:
        #     self.update_network_parameters(critic_only=True)
        #     return 0,cl
        
        self.actor.optimizer.zero_grad()
        cont_actions, disc_actions = self.actor.forward(states)
        cont_actions.retain_grad()
        actor_loss = -self.critic_1.forward(states, cont_actions, disc_actions).mean()
        # gradient inversion for bounding action space for continuous actions
        actor_loss.backward()
        if step%2 == 0:
            if self.use_grad_inv:
                if cont_actions.grad is not None:
                    with T.no_grad():
                        # print('inverting gradients')
                        grad = cont_actions.grad
                        # print(grad)
                        inverted_grad = T.where(
                        grad > 0,
                        grad * ((self.pmax - cont_actions) / (self.pmax - self.pmin + 1e-6)),
                        grad * ((cont_actions - self.pmin) / (self.pmax - self.pmin + 1e-6))
                        )
                        cont_actions.grad.copy_(inverted_grad)
            
            nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            self.actor.optimizer.step()

            al = actor_loss.item()
            self.update_network_parameters()
        else:
            al = 0
        return al, cl


    def save_models(self):
        print('... saving models ...')
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.target_critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()
        self.target_critic_2.save_checkpoint()

    def load_models(self):
        print('... loading models ...')
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.target_critic_1.load_checkpoint() 
        self.critic_2.load_checkpoint()
        self.target_critic_2.load_checkpoint()       