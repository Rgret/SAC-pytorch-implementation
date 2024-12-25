import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import os

class ActorCritic(nn.Module):
    def __init__(self, state_dims, action_dims, hidden_dims, max_action,
                  log_std_min=-20, log_std_max=2, ckpt_dir = "Network", name='AC', 
                  device = ('cuda:0' if torch.cuda.is_available() else 'cpu')):
        super().__init__()

        self.actor = Actor(state_dims, action_dims, hidden_dims,
                           max_action, name=name+'_Actor', ckpt_dir=ckpt_dir, device=device)
        
        self.q1 = Critic(state_dims, action_dims, hidden_dims, name=name+'_Critic_1',
                             ckpt_dir=ckpt_dir, device=device)
        
        self.q2 = Critic(state_dims, action_dims, hidden_dims, name=name+'_Critic_2',
                             ckpt_dir=ckpt_dir, device=device)
        
        self.device = device
        self.to(self.device)

    def act(self, state, deterministic = False, with_logprob = False):
        with torch.no_grad():
            a, _ = self.actor(state, deterministic, with_logprob)
            return a

    def save(self, path=None):
        self.actor.save_checkpoint(path)
        self.q1.save_checkpoint(path)
        self.q2.save_checkpoint(path)

    def load(self):
        self.actor.load_checkpoint()
        self.q1.load_checkpoint()
        self.q2.load_checkpoint()


class Actor(nn.Module):
    def __init__(self, state_dims, action_dims, hidden_dims, max_action, log_std_min=-2, log_std_max=1,
                 name='Actor', ckpt_dir='tmp', 
                 device = ('cuda:0' if torch.cuda.is_available() else 'cpu')):
        super(Actor, self).__init__()
        self.state_dims = state_dims
        self.action_dims = action_dims
        self.hidden_dims = hidden_dims
        self.max_action = max_action
        self.name = name
        self.ckpt_path = os.path.join(ckpt_dir, name)
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.device = device

        self.l1 = nn.Linear(state_dims, self.hidden_dims)
        self.l2 = nn.Linear(self.hidden_dims, self.hidden_dims)
        self.l3 = nn.Linear(self.hidden_dims, self.hidden_dims)

        self.mean = nn.Linear(self.hidden_dims, self.action_dims)
        self.sigma = nn.Linear(self.hidden_dims, self.action_dims)

        self.layers = {'l1': self.l1, 'l2':  self.l2, 'l3':  self.l3, 'mean':  self.mean, 'sigma':  self.sigma}

        self.to(self.device)
        self.prelu_weight = torch.as_tensor(0.25).to(self.device)

        for layer in [self.l1, self.l2, self.l3, self.mean, self.sigma]:
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, state, deterministic=False, with_logprob=True):
        l1 = F.tanhshrink(self.l1(state.to(self.device)))
        l2 = F.tanhshrink(self.l2(l1))
        l3 = F.tanhshrink(self.l3(l2))
        mean = self.mean(l3)
        sigma = self.sigma(l3)
        sigma = torch.clamp(sigma, min=self.log_std_min, max=self.log_std_max).to(self.device)
        std = torch.exp(sigma).to(self.device)

        #Pre-squashed distribution and sample
        pi_distribution = torch.distributions.normal.Normal(mean, std * 0.2)
        if deterministic:
            # testing agent
            action = mean
        else:
            action = pi_distribution.rsample()

        if with_logprob:
            # pure magic from SAC original paper, no idea honestly
            logp_pi = pi_distribution.log_prob(action).sum(axis=-1)
            logp_pi -= (2*(np.log(2) - action - F.softplus(-2*action))).sum(axis=-1)
        else:
            logp_pi = None

        action = torch.tanh(action)
        action = self.max_action * action

        return action, logp_pi

    def save_checkpoint(self, path=None):
        if path != None:
            torch.save(self.state_dict(), os.path.join(path, self.name))
        else:
            torch.save(self.state_dict(), self.ckpt_path)

    def load_checkpoint(self, gpu_to_cpu=False):
        if gpu_to_cpu:
            self.load_state_dict(torch.load(self.ckpt_path, map_location=lambda storage, loc: storage))
        else:
            self.load_state_dict(torch.load(self.ckpt_path))

class Critic(nn.Module):
    def __init__(self, state_dims, action_dims, hidden_dims, name='Critic', ckpt_dir='tmp', 
                    device = ('cuda:0' if torch.cuda.is_available() else 'cpu')):
        super(Critic, self).__init__()
        self.state_dims = state_dims
        self.action_dims = action_dims
        self.hidden_dims = hidden_dims
        self.name = name
        self.ckpt_path = os.path.join(ckpt_dir, name)

        # Q architecture
        self.l1 = nn.Linear(self.state_dims + self.action_dims, self.hidden_dims)
        self.l2 = nn.Linear(self.hidden_dims, self.hidden_dims)
        self.l3 = nn.Linear(self.hidden_dims, self.hidden_dims)
        self.q = nn.Linear(self.hidden_dims, 1)

        self.device = device
        self.to(self.device)

        for layer in [self.l1, self.l2, self.l3, self.q]:
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, state, action):
        state_action = torch.cat([state, action], -1).to(self.device)
        l1 = F.tanhshrink(self.l1(state_action))
        l2 = F.tanhshrink(self.l2(l1))
        l3 = F.tanhshrink(self.l3(l2))
        q = self.q(l3)

        return torch.squeeze(q, -1).to(self.device)

    def save_checkpoint(self, path=None):
        if path != None:
            torch.save(self.state_dict(), os.path.join(path, self.name))
        else:
            torch.save(self.state_dict(), self.ckpt_path)

    def load_checkpoint(self, gpu_to_cpu=False):
        if gpu_to_cpu:
            self.load_state_dict(torch.load(self.ckpt_path, map_location=lambda storage, loc: storage))
        else:
            self.load_state_dict(torch.load(self.ckpt_path))
