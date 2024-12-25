import torch
from agents import ActorCritic
from replay_buffer import ReplayBuffer
from logger import Logger
import torch.optim as optim
import itertools

# https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/sac/core.py
# https://spinningup.openai.com/en/latest/_modules/spinup/algos/pytorch/sac/sac.html
# https://spinningup.openai.com/en/latest/algorithms/sac.html

class SAC(object):
    def __init__(self, state_dims, action_dims, max_action, 
                hidden_dims=512, 
                buffer_size=1e6, lr=1e-5, alpha=.2, discount=0.9, 
                polyak=0.99, ckpt_dir='Network', update_every=1, learn_after=10000, 
                device = ('cuda:0' if torch.cuda.is_available() else 'cpu')):
        
        self.alpha = alpha
        self.polyak = polyak
        self.discount = discount
        self.update_every = update_every
        self.learn_after = learn_after
        self.current_step = 1
        self.device = device
        
        # Actor Critic network
        self.ac = ActorCritic(state_dims, action_dims, hidden_dims,
                              max_action, ckpt_dir=ckpt_dir, device=self.device)
        self.ac_target = ActorCritic(state_dims, action_dims, hidden_dims,
                              max_action, name='AC_Target', ckpt_dir=ckpt_dir, device=self.device)
        self.ac_target.load_state_dict(self.ac.state_dict())

        # Freeze updates of target neworks
        for p in self.ac_target.parameters():
            p.requires_grad = False

        self.q_params = itertools.chain(self.ac.q1.parameters(), self.ac.q2.parameters())
        self.a_optimizer = optim.Adam(self.ac.actor.parameters(), lr=lr)
        self.q_optimizer = optim.Adam(self.q_params, lr=lr)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(int(state_dims), int(action_dims), int(buffer_size))

        # Logger
        self.logger = Logger('Logs')


    def save(self, path=None):
        self.ac.save(path)
        self.ac_target.save(path)

    def load(self):
        self.ac.load()
        self.ac_target.load()

    def compute_loss_q(self, batch):
        state, action, next_state, reward, done = batch
        state = state.to(self.device)
        action = action.to(self.device)
        next_state = next_state.to(self.device)
        reward = reward.to(self.device)
        done = done.to(self.device)

        q1 = self.ac.q1(state, action).to(self.device)
        q2 = self.ac.q2(state, action).to(self.device)

        with torch.no_grad():
            # Target actions from current policy
            a2, logp_a2 = self.ac.actor(next_state)

            # Target Q-values
            q1_pi_targ = self.ac_target.q1(next_state, a2).to(self.device)
            q2_pi_targ = self.ac_target.q2(next_state, a2).to(self.device)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ).to(self.device)
            backup = reward + self.discount * (1 - done) * (q_pi_targ - self.alpha * logp_a2)
        
        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup)**2).mean()
        loss_q2 = ((q2 - backup)**2).mean()
        loss_q = loss_q1 + loss_q2

        # For loggging
        q_info = dict(Q1Vals=q1.cpu().detach().numpy(),
                        Q2Vals=q2.cpu().detach().numpy())
        
        return loss_q, q_info
        
    def compute_loss_pi(self, batch):
        state, _, _, _, _ = batch
        state = state.to(self.device)
        pi, logp_pi = self.ac.actor(state)
        q1_pi = self.ac.q1(state, pi).to(self.device)
        q2_pi = self.ac.q2(state, pi).to(self.device)
        q_pi = torch.min(q1_pi, q2_pi).to(self.device)

        # Entropy-regularized policy loss
        loss_pi = (self.alpha * logp_pi - q_pi).mean()

        # For logging
        pi_info = dict(LogPi=logp_pi.cpu().detach().numpy())
        return loss_pi, pi_info
        
    def update(self, batch_size):
        self.current_step += 1
        if self.current_step < self.learn_after or (self.current_step % self.update_every) != 0:
            return

        batch = self.replay_buffer.sample(batch_size)

        # Gradient descent for Q1 and Q2
        self.q_optimizer.zero_grad()
        loss_q, q_info = self.compute_loss_q(batch)
        loss_q.backward()
        self.q_optimizer.step()

        # For logging
        self.logger.store(LossQ=loss_q.item(), **q_info)

        # Freeze Q-Networks
        for p in self.q_params:
            p.requires_grad = False
        
        # Gradient descent for actor
        self.a_optimizer.zero_grad()
        loss_pi, pi_info = self.compute_loss_pi(batch)
        loss_pi.backward()
        self.a_optimizer.step()

        # Unfreeze Q-networks
        for p in self.q_params:
            p.requires_grad = True

        # For logging
        self.logger.store(LossPi=loss_pi.item(), **pi_info)

        # Update target networks by polyak averaging
        with torch.no_grad():
            for p, p_targ in zip(self.ac.parameters(), self.ac_target.parameters()):
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)
        
    def get_action(self, state, deterministic=False, with_logprob=False):
        return self.ac.act(torch.as_tensor(state, dtype=torch.float32), deterministic, with_logprob)
    