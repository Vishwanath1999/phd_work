# %%
import torch
import numpy as np
from scipy.io import loadmat, savemat
from scipy import constants as cts
import torch.types
from tqdm import tqdm
from ipywidgets import interact, widgets
import matplotlib.pyplot as plt
import pandas as pd
from typing import Callable

DEVICE = 'cpu'
C0 = 299792458
H_BAR = cts.hbar
# %%
class RL_MRR_Env():

    def __init__(self):
        super(RL_MRR_Env, self).__init__()

        self.step_cntr = 0
        
        self.disp = loadmat('disp.mat')
        self.res = loadmat('res.mat')
        self.sim = loadmat('sim.mat')


        self.sim['DKS_init'] = np.array([complex(x.strip()) for x in self.sim['DKS_init']])
        # Convert the complex numbers to np.complex64
        self.sim['DKS_init'] = self.sim['DKS_init'].astype(np.complex128)

        myDKS = loadmat('My_DKS_init_a3c.mat')
        self.sim['DKS_init'] =  myDKS['DKS_init'][0]

        self.sim['domega_init'] = myDKS['detuning']
        self.sim['domega_end'] = myDKS['detuning_end']

        self.disp_tensor = self.dict_to_tensor(self.disp)
        self.res_tensor = self.dict_to_tensor(self.res)
        self.sim.pop('domega', None)
        self.sim['domega'] = np.array(['None',0])
        self.sim_tensor = self.dict_to_tensor(self.sim)

        self.disp_tensor['D1'] = self.disp_tensor['D1'][0]
        self.disp_tensor['FSR'] = self.disp_tensor['FSR'][0]
        self.disp_tensor['FSR_center'] = self.disp_tensor['FSR_center'][0]

        for key in self.res_tensor:
            self.res_tensor[key] = self.res_tensor[key][0]

        # if ndim>1 then squeeze the tensor
        for key in self.sim_tensor:
            if self.sim_tensor[key].ndim > 1:
                self.sim_tensor[key] = self.sim_tensor[key][0]
        
        ng = self.disp_tensor['ng']
        R = self.res_tensor['R']
        gamma = self.res_tensor['gamma']
        self.gamma = gamma
        L = 2*torch.pi*R
        self.L = L

        Q0 = self.res_tensor['Qi']
        Qc = self.res_tensor['Qc']
        fpmp = self.sim_tensor['f_pmp']
        Ppmp = self.sim_tensor['Pin']
        phi_pmp = self.sim_tensor['phi_pmp']
        self.phi_pmp = phi_pmp
        num_probe = self.sim_tensor['num_probe']
        num_probe = num_probe[0].cpu().numpy().astype(int)
        fcenter = self.sim_tensor['f_center']

        D1 = self.disp_tensor['D1']
        FSR = D1/(2*torch.pi)
        omega0 = 2*torch.pi*fpmp
        omega_center = 2*torch.pi*fcenter

        tR = 1/FSR
        self.tR = tR
        T = 1*tR
        kext = (omega0[0]/Qc) * tR
        self.kext = kext
        k0 = (omega0[0]/Q0) * tR
        alpha = k0+kext
        self.alpha = alpha

        del_omega_init = self.sim_tensor['domega_init']
        self.del_omega_init = del_omega_init
        self.current_del_omega = del_omega_init
        del_omega_end = self.sim_tensor['domega_end']
        self.del_omega_end = del_omega_end

        del_omega_stop = self.sim_tensor['domega_stop']
        ind_sweep = self.sim_tensor['ind_pump_sweep'] 
        t_end = self.sim_tensor['Tscan']
        Dint = self.disp_tensor['Dint_new']
        self.Dint = Dint

        DKSinit_real = torch.real(self.sim_tensor['DKS_init'])
        if self.sim_tensor['DKS_init'].dtype == torch.complex128 or self.sim_tensor['DKS_init'].dtype == torch.complex64:
            DKSinit_imag = torch.imag(self.sim_tensor['DKS_init'])
        else:
            DKSinit_imag = torch.zeros_like(DKSinit_real, device=DEVICE)

        self.DKS_init = torch.complex(DKSinit_real, DKSinit_imag)
        del_omega = self.sim['domega']
        ind_pmp = [ii for ii in self.sim_tensor['ind_pmp'].int().cpu().numpy()]    
        self.ind_pmp = ind_pmp
        mu_sim = self.sim_tensor['mucenter']
        mu = torch.arange(mu_sim[0], mu_sim[1]+1, device=DEVICE)
        self.mu= mu
        # find center of mu
        mu0 = torch.where(mu == 0)[0][0].int().cpu().numpy()+1
        self.mu0 = mu0

        d_omega = 2*torch.pi*FSR * torch.arange(mu_sim[0], mu_sim[-1]+1, device=DEVICE)
        domega_pmp1 = 2*torch.pi*FSR * torch.arange(mu_sim[0]-ind_pmp[0]-1, mu_sim[-1]-ind_pmp[0], device=DEVICE)
        omega1 = omega0[0] + domega_pmp1

        self.Dint = Dint-Dint[mu0-1]
        Ptot = 0#torch.zeros(1, device=device)
        for ii in range(len(fpmp)):
            Ptot += Ppmp[ii]
        
        # fpmp = fpmp[0]
        Ain = torch.zeros(len(fpmp), len(mu),dtype=torch.complex128, device=DEVICE)
        Ein = torch.zeros(len(fpmp), len(mu),dtype=torch.complex128, device=DEVICE)

        for ii in range(len(fpmp)):
            Ein[ii,int(mu0+ind_pmp[ii])] = torch.sqrt(Ppmp[ii])*len(mu)
            Ain[ii] = torch.fft.ifft(torch.fft.fftshift(Ein[ii],dim=0),dim=0)*torch.exp(-1j*phi_pmp[ii])
        
        self.Ain = Ain

        self.Ptot_dist = torch.distributions.uniform.Uniform(0.01, 0.2)

        self.Dint_shift = torch.fft.ifftshift(self.Dint)

        dt = 1
        t_end  = self.sim_tensor['Tscan']*tR
        t_ramp = t_end
        Nt = int(3e5)
        self.max_steps = Nt

        self.del_omega_0 = del_omega_init + (1/Nt)*(del_omega_end - del_omega_init)

        del_omega_tot = torch.abs(del_omega_end)+torch.abs(del_omega_init)
        del_omega_perc = -1*torch.sign(del_omega_end+del_omega_init)*(torch.abs(del_omega_end+del_omega_init)/2)/del_omega_tot
        self.t_sim = torch.linspace(-t_ramp[0]/2 + del_omega_perc[0]*t_ramp[0], t_ramp[0]/2 + del_omega_perc[0]*t_ramp[0], Nt, device=DEVICE, dtype=torch.float64)
    
    def Fdrive(self, del_omega_all, t_sim, Ain, ind_pmp):
        Force = torch.zeros(len(self.mu), device=DEVICE, dtype=torch.complex128)
        for ii in range(len(ind_pmp)):
            if ii > 0:
                sigma = (2*del_omega_all + self.Dint[(self.mu0)+ind_pmp[ii]] - 0.5*self.del_omega_0)*t_sim
            else:
                sigma=torch.zeros(1, device=DEVICE)
            Force = Force - 1j*Ain[ii]*torch.exp(1j*sigma)
        return Force

    def dict_to_tensor(self,dic):
        dic.pop('__header__', None)
        dic.pop('__version__', None)
        dic.pop('__globals__', None)
        dic.pop('dispfile', None)

        tensor_dic = dict()
        # convert the dictionary to a tensor
        for key in dic:
            if isinstance(dic[key], np.ndarray):
                if key == 'DKS_init':
                    tensor_dic[key] = torch.tensor(dic[key], device=DEVICE, dtype=torch.complex64)
                else:
                    dic[key] = np.where(dic[key] == 'None', None, dic[key])
                    dic[key] = dic[key].astype(np.float64)
                    tensor_dic[key] = torch.tensor(dic[key], device=DEVICE)
            else:
                tensor_dic[key] = dic[key]
        return tensor_dic
    
    def tensor_to_dict(self,tensor):
        dic = dict()
        for key in tensor:
            # print(key)
            if isinstance(tensor[key], torch.Tensor):
                if tensor[key].dtype == torch.complex128:
                    dic[key] = tensor[key].cpu().numpy().astype(np.complex128)
                else:   
                    dic[key] = tensor[key].cpu().numpy().astype(np.float64)
            else:
                dic[key] = tensor[key]
        return dic
    
    @staticmethod
    @torch.jit.script
    def FFT_Lin(it: int, alpha: torch.Tensor, Dint_shift: torch.Tensor, del_omega_all: torch.Tensor, tR: torch.Tensor) -> torch.Tensor:
        '''
        Linear operator
        Input:
            it (int) : Time index
            alpha (torch.Tensor) : Linewidth enhancement factor
            Dint_shift (torch.Tensor) : Dispersive operator
            del_omega_all (torch.Tensor) : Detuning
            tR (torch.Tensor) : Round trip time
        Output:
            torch.Tensor : Linear operator
        '''
        return (-alpha / 2) + 1j * (Dint_shift - del_omega_all) * tR


    # Function for the Nonlinear operator
    @staticmethod
    @torch.jit.script
    def NL(uu: torch.Tensor, gamma: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
        '''
        Nonlinear operator
        Input:
            uu (torch.Tensor) : Input field
            gamma (torch.Tensor) : Nonlinear coefficient
            L (torch.Tensor) : Length of the resonator
        Output:
            torch.Tensor : Nonlinear operator
        '''
        return -1j * (gamma * L * torch.square(torch.abs(uu)) )
    
    # @staticmethod
    # @torch.jit.script
    def ssfm_step(self,A0: torch.Tensor, it: int, alpha: torch.Tensor, Dint_shift: torch.Tensor,
                del_omega_all: torch.Tensor, tR: torch.Tensor, gamma: torch.Tensor, L: torch.Tensor, 
                max_iter: int, tol: float, dt: int, kext: torch.Tensor, Fdrive_val:torch.Tensor,
                ) -> torch.Tensor:
        
        A0 = A0 + Fdrive_val * torch.sqrt(kext) * dt
        L_h_prop = torch.exp(self.FFT_Lin(it, alpha, Dint_shift, del_omega_all, tR) * dt / 2)
        A_L_h_prop = torch.fft.ifft(torch.fft.fft(A0) * L_h_prop)
        NL_h_prop_0 = self.NL(A0, gamma, L)
        A_h_prop = A0#.clone()
        A_prop = 0*A0.clone()
        # torch.zeros_like(A0, dtype=torch.complex128, device=A0.device)

        for _ in range(max_iter):
            err=0
            NL_h_prop_1 = self.NL(A_h_prop, gamma, L)
            NL_prop = (NL_h_prop_0 + NL_h_prop_1) * dt / 2
            A_prop = torch.fft.ifft(torch.fft.fft(A_L_h_prop * torch.exp(NL_prop)) * L_h_prop)
            err = torch.linalg.vector_norm(A_prop - A_h_prop, ord=2, dim=0) / torch.linalg.vector_norm(A_h_prop, ord=2, dim=0)
            if err < tol:
                return A_prop
            A_h_prop = A_prop
        err = torch.linalg.vector_norm(A_prop - A_h_prop, ord=2, dim=0) / torch.linalg.vector_norm(A_h_prop, ord=2, dim=0)
        raise RuntimeError(f"Convergence Error: {err}")
    
    def reset(self, steps=None):
        self.state = self.DKS_init
        self.step_cntr = 0

        Ppmp = self.Ptot_dist.sample().unsqueeze(0)
        # fpmp = fpmp[0]
        Ain = torch.zeros(1, len(self.mu),dtype=torch.complex128, device=DEVICE)
        Ein = torch.zeros(1, len(self.mu),dtype=torch.complex128, device=DEVICE)

        for ii in range(1):
            Ein[ii,int(self.mu0+self.ind_pmp[ii])] = torch.sqrt(Ppmp[ii])*len(self.mu)
            Ain[ii] = torch.fft.ifft(torch.fft.fftshift(Ein[ii],dim=0),dim=0)*torch.exp(-1j*self.phi_pmp[ii])
        
        self.Ain = Ain
        if steps is not None:
            steps_ = steps
        else:
            steps_ = int(1.5e5)
        # for idx in tqdm(range(int(1.5e5)), ncols=120):
        for idx in range(steps_):
            del_omega = self.current_del_omega + (1/self.max_steps)*(self.del_omega_end - self.del_omega_init)

            Fdrive_val = self.Fdrive(del_omega, self.t_sim[self.step_cntr], self.Ain, self.ind_pmp)
            u0 = self.ssfm_step(self.state, self.step_cntr, self.alpha, self.Dint_shift, del_omega, self.tR, self.gamma, \
                                self.L, 10, 1e-3, 1, self.kext, Fdrive_val)
            self.step_cntr += 1
            self.state = u0        
            self.current_del_omega = del_omega

        Acav = torch.sqrt(self.alpha/2)*self.state*np.exp(1j*torch.pi)/len(self.mu)
        Ecav = torch.fft.fftshift(torch.fft.fft(Acav))
        Ecav_dBm = 10*torch.log10(torch.abs(Ecav)**2)+30
        Ecav_dBm = torch.clamp(Ecav_dBm, min=-60, max=10)
        print('Reset...')
        return self.state, Acav.numpy(), Ecav_dBm.numpy()
    
    def step(self, state, action, desired_spectrum):
        Ppmp = torch.tensor(action[1:], dtype=torch.float64)
        Ain = torch.zeros(1, len(self.mu),dtype=torch.complex128, device=DEVICE)
        Ein = torch.zeros(1, len(self.mu),dtype=torch.complex128, device=DEVICE)

        for ii in range(1):
            Ein[ii,int(self.mu0+self.ind_pmp[ii])] = torch.sqrt(Ppmp[ii])*len(self.mu)
            Ain[ii] = torch.fft.ifft(torch.fft.fftshift(Ein[ii],dim=0),dim=0)*torch.exp(-1j*self.phi_pmp[ii])
        
        self.Ain = Ain

        if action[0] == 0:
            del_omega = self.current_del_omega - (1/self.max_steps)*(self.del_omega_end - self.del_omega_init)
        elif action[0] == 1:
            del_omega = self.current_del_omega + (1/self.max_steps)*(self.del_omega_end - self.del_omega_init)
        else:
            del_omega = self.current_del_omega

        Fdrive_val = self.Fdrive(del_omega, self.t_sim[self.step_cntr], self.Ain, self.ind_pmp)
        u0 = self.ssfm_step(state, self.step_cntr, self.alpha, self.Dint_shift, del_omega, self.tR, self.gamma, \
                            self.L, 10, 1e-3, 1, self.kext, Fdrive_val)
        
        self.step_cntr += 1
        self.current_del_omega = del_omega
        self.next_state = u0
        

        Acav = torch.sqrt(self.alpha/2)*u0*np.exp(1j*torch.pi)/len(self.mu)
        Ecav = torch.fft.fftshift(torch.fft.fft(Acav))

        Ecav_dBm = 10*torch.log10(torch.abs(Ecav)**2)+30
        Ecav_dBm = torch.clamp(Ecav_dBm, min=-60, max=10)
        desired_spectrum_dBm = 10*torch.log10(torch.abs(desired_spectrum)**2)+30
        desired_spectrum_dBm = torch.clamp(desired_spectrum_dBm, min=-60, max=10)

        if torch.allclose(Ecav_dBm, desired_spectrum_dBm, atol=1):
            achieved = True
        else:
            achieved = False
        

        # Calculate the reward as the l2 error norm between the desired spectrum and the current spectrum
        reward = -(torch.linalg.norm(Ecav-desired_spectrum, ord=2)/torch.linalg.norm(desired_spectrum, ord=2))
        # reward = - torch.linalg.norm((Ecav_dBm-desired_spectrum_dBm)/500, ord=2)
        if (self.step_cntr == self.max_steps) or (self.step_cntr == self.max_steps//2 and torch.mean(Ecav_dBm) < -60):
            done = True
        else:
            done = False
        
        return self.next_state, reward.numpy(), done, achieved, Acav.numpy(), desired_spectrum_dBm.numpy()

# %%
env = RL_MRR_Env()

# state, acav,ecav = env.reset()
# plt.plot(ecav)
# plt.show()
# %%
# state, acav = env.reset()
# s = (torch.sqrt(env.alpha/2)*state).cpu().numpy()*np.exp(1j*torch.pi)/len(env.mu)
# del_omega=[]
# del_omega.append(env.current_del_omega.cpu().numpy())
# %%
desired_spectrum = loadmat('desired_spec.mat')['Ecav'][0]
desired_spectrum = torch.tensor(desired_spectrum, device=DEVICE, dtype=torch.complex128)
# %%
# from maac import Agent, RewardScaler

# agent = Agent(alpha=1e-4, beta=5e-4, input_dims=[441], disc_n_actions=3, cont_n_actions=1)

# reward_scaler = RewardScaler(alpha=0.0005)
from mappo import MAPPO_Agent
agent = MAPPO_Agent(disc_n_actions=3, cont_n_actions=1, input_dims=[441], alpha=1e-4)
# %% AC
# n_games = 1000
# r_hist = []
# scaled_r_hist = []
# scores = []
# for i in range(n_games):
#     score = 0
#     done = False
#     state, acav, ecav = env.reset()
#     reward_scaler.reset()
#     while not done:
#         action = agent.choose_action(ecav/10)
#         # print(action)
#         next_state, reward, done, achieved, _, ecav_ = env.step(state, action, desired_spectrum)
#         reward_scaler.update(reward)
#         new_reward = reward_scaler.normalize(reward)
#         if achieved==True:
#             new_done = False
#             done = True
#         else:
#             new_done = done            
#         dl,al,cl = agent.learn(ecav/10, reward, ecav_/10, new_done)
#         state = next_state
#         ecav = ecav_
#         score += reward
#         r_hist.append(reward)
#         scaled_r_hist.append(new_reward)
#     scores.append(score)
#     print('episode ', i, 'score %.2f' % score)

# %% PPO training loop
n_games = 1000
r_hist = []
scaled_r_hist = []
scores = []
n_steps = 0
best_score = -np.inf
for i in range(n_games):
    score = 0
    done = False
    state, acav, ecav = env.reset()
    while not done:
        action, vals, probs = agent.choose_action(ecav/10)
        # print(action)
        next_state, reward, done, achieved, _, ecav_ = env.step(state, action, desired_spectrum)

        if achieved==True:
            new_done = False
            done = True
        else:
            new_done = done            
        agent.remember(ecav/10, action, probs, vals, reward, new_done)
        state = next_state
        ecav = ecav_
        score += reward
        r_hist.append(reward)
        n_steps += 1
        if n_steps % 64*10*2 == 0:
            agent.learn()
    scores.append(score)
    avg_score = np.mean(scores[-100:])
    print('episode ', i, 'score %.2f' % score)
    if avg_score > best_score:
        best_score = avg_score
        agent.save_models()
# %% save model
# torch.save(agent.actor.state_dict(), './rl_results/actor_power_det_ctrl.pth')
# %%
plt.figure(figsize=(10,6))
plt.plot(scores)
plt.grid()
plt.xlabel('Episodes')
plt.ylabel('Scores')
plt.title('Scores vs Episodes')
plt.show()
# %%
agent_frozen = agent
# freeze the actor network
for param in agent_frozen.actor.parameters():
    param.requires_grad = False
state, acav, ecav = env.reset()
r_hist = []
action_hist = []
acav_hist = []
score = 0
done = False
while not done:
    action = agent_frozen.choose_action(ecav/10, deterministic=True)
    next_state, reward, done, achieved, acav_, ecav_ = env.step(state, action, desired_spectrum)
    state = next_state
    ecav = ecav_
    score += reward
    r_hist.append(reward)
    action_hist.append(action)
    acav_hist.append(acav_)

print('Test score %.2f' % score)
# %%
import matplotlib.ticker as ticker

plt.figure(figsize=(14,4))
plt.imshow(np.abs(np.array(acav_hist).T), aspect='auto', cmap='jet')
# plt.xlabel('Iteration', fontsize=14)
# plt.ylabel('Tr', fontsize=14)
plt.colorbar(label='Power(dBm)')
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
# Set x-ticks to exponent format
formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(True)
formatter.set_powerlimits((-1, 1))
plt.gca().xaxis.set_major_formatter(formatter)
plt.savefig('./rl_results/ac_ecav_hist_spec_all_ctrl.png')
plt.show()

# %%
plt.figure(figsize=(14,4))
plt.imshow(np.abs(np.fft.fftshift(np.fft.fft(np.array(acav_hist)))).T, aspect='auto', cmap='jet')
# plt.xlabel('Iteration', fontsize=14)
# plt.ylabel('Tr', fontsize=14)
plt.colorbar(label='Power(dBm)')
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.savefig('./rl_results/ac_ecav_hist_ifft_spec_all_ctrl.png')
plt.show()
# %%
plt.figure(figsize=(10, 6))
plt.plot(r_hist)
plt.xlabel('Iteration')
plt.ylabel('Reward(Neg L2 Error)')
plt.grid()
plt.savefig('./rl_results/ac_rewards_spec_all_ctrl.png')
plt.show()
# %%
desired_spectrum_dBm = 10*torch.log10(torch.abs(desired_spectrum)**2)+30
plt.figure(figsize=(14,4))
plt.vlines(np.arange(len(ecav)), -100*np.ones(len(ecav)), ecav, \
           label='Obtained Spectrum')
plt.vlines(np.arange(len(desired_spectrum)), -100*np.ones(len(desired_spectrum)),\
            desired_spectrum_dBm, color='red', label='Desired Spectrum',alpha=0.5)
plt.xlabel('Mode no.', fontsize=14)
plt.ylabel('Power(dBm)', fontsize=14)
plt.grid()
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=14)
plt.savefig('./rl_results/ac_ecav_spec_all_ctrl.png')
plt.show()
# %%
plt.figure(figsize=(10, 6))
actions = np.array(action_hist[0:1])
plt.plot(actions, label='Actions', color='gray', alpha=0.5)

# Highlight different actions with different colors
plt.scatter(np.where(actions == 0), actions[actions == 0], color='red', label='Decrease Detuning', marker='o')
plt.scatter(np.where(actions == 1), actions[actions == 1], color='green', label='Increase Detuning', marker='x')
plt.scatter(np.where(actions == 2), actions[actions == 2], color='blue', label='No Update', marker='s')

plt.xlabel('Iteration', fontsize=14)
plt.ylabel('Action', fontsize=14)
plt.legend()
plt.grid()
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.savefig('./rl_results/ac_actions_spec_all_ctrl.png')
plt.show()