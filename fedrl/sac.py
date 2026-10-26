import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0


def mlp(in_dim, out_dim, hidden=(128, 128)):
    layers, d = [], in_dim
    for h in hidden:
        layers += [nn.Linear(d, h), nn.ReLU()]
        d = h
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=(128, 128)):
        super().__init__()
        self.net = mlp(obs_dim, 2 * act_dim, hidden)
        self.act_dim = act_dim

    def forward(self, obs):
        mu, log_std = self.net(obs).chunk(2, dim=-1)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mu, log_std

    def sample(self, obs):
        mu, log_std = self(obs)
        std = log_std.exp()
        dist = torch.distributions.Normal(mu, std)
        u = dist.rsample()
        a = torch.tanh(u)
        logp = dist.log_prob(u).sum(-1) - torch.log(1 - a.pow(2) + 1e-6).sum(-1)
        return a, logp

    def act(self, obs, deterministic=False):
        with torch.no_grad():
            mu, log_std = self(obs)
            if deterministic:
                return torch.tanh(mu)
            return torch.tanh(torch.distributions.Normal(mu, log_std.exp()).sample())


class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=(128, 128)):
        super().__init__()
        self.q1 = mlp(obs_dim + act_dim, 1, hidden)
        self.q2 = mlp(obs_dim + act_dim, 1, hidden)

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)


class SACAgent:
    def __init__(self, obs_dim, act_dim, lr=3e-4, gamma=0.99, tau=0.005,
                 hidden=(128, 128), device="cpu"):
        self.device = device
        self.gamma, self.tau = gamma, tau
        self.actor = Actor(obs_dim, act_dim, hidden).to(device)
        self.critic = Critic(obs_dim, act_dim, hidden).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        for p in self.critic_target.parameters():
            p.requires_grad_(False)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.target_entropy = -float(act_dim)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=lr)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, obs_np, deterministic=False):
        obs = torch.as_tensor(obs_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        return self.actor.act(obs, deterministic).squeeze(0).cpu().numpy()

    def update(self, batch, is_weights=None):
        """batch: dict of tensors (obs, act, rew, obs2, done). Returns per-sample |TD error|."""
        obs, act, rew, obs2, done = (batch[k] for k in ("obs", "act", "rew", "obs2", "done"))
        w = is_weights if is_weights is not None else torch.ones_like(rew)

        with torch.no_grad():
            a2, logp2 = self.actor.sample(obs2)
            q1t, q2t = self.critic_target(obs2, a2)
            target = rew + self.gamma * (1 - done) * (torch.min(q1t, q2t) - self.alpha * logp2)

        q1, q2 = self.critic(obs, act)
        td = 0.5 * (torch.abs(q1 - target) + torch.abs(q2 - target)).detach()
        critic_loss = (w * ((q1 - target) ** 2 + (q2 - target) ** 2)).mean()
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        a, logp = self.actor.sample(obs)
        q1pi, q2pi = self.critic(obs, a)
        actor_loss = (w * (self.alpha.detach() * logp - torch.min(q1pi, q2pi))).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        alpha_loss = -(self.log_alpha * (logp.detach() + self.target_entropy)).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        with torch.no_grad():
            for p, pt in zip(self.critic.parameters(), self.critic_target.parameters()):
                pt.mul_(1 - self.tau).add_(self.tau * p)

        return td.cpu().numpy()

    def param_count(self):
        return sum(p.numel() for p in self.actor.parameters()) + \
               sum(p.numel() for p in self.critic.parameters())


def fedavg_sync(agents):
    """Average actor / critic / target-critic parameters across agents, in place."""
    for attr in ("actor", "critic", "critic_target"):
        nets = [getattr(a, attr) for a in agents]
        avg = {k: torch.stack([n.state_dict()[k].float() for n in nets]).mean(0)
               for k in nets[0].state_dict()}
        for n in nets:
            n.load_state_dict(avg)
