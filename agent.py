import torch
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
from utils import discount_rewards


class Policy(torch.nn.Module):
    def __init__(self, state_space, action_space):
        super().__init__()
        self.state_space = state_space
        self.action_space = action_space
        self.hidden = 64
        self.fc1 = torch.nn.Linear(state_space, self.hidden)
        self.fc2_mean = torch.nn.Linear(self.hidden, action_space)
        # Task 1
        self.sigma = torch.tensor([5]).float()
        # Task 2 a
        # self.sigma = torch.tensor([10]).float()
        # Task 2 b
        # self.sigma = torch.nn.Parameter(torch.tensor([10]).float())
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear:
                torch.nn.init.normal_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, x, episode_number):
        x = self.fc1(x)
        x = F.relu(x)
        action_mean = self.fc2_mean(x)
        # Task 1
        sigma = self.sigma

        # Tasl 2 a
        # c = torch.tensor([0.0005])
        # k = torch.tensor([episode_number])
        # sigma = self.sigma * (torch.exp(-c * k))

        # Task 2 b
        # sigma = self.sigma

        # TODO: Instantiate and return a normal distribution
        action_dist = Normal(action_mean, sigma)

        return action_dist


class Agent(object):
    def __init__(self, policy):
        self.train_device = "cpu"
        self.policy = policy.to(self.train_device)
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=5e-3)
        self.gamma = 0.98
        self.states = []
        self.action_probs = []
        self.rewards = []

    def episode_finished(self, episode_number):
        action_probs = torch.stack(self.action_probs, dim=0) \
                .to(self.train_device).squeeze(-1)
        rewards = torch.stack(self.rewards, dim=0).to(self.train_device).squeeze(-1)
        self.states, self.action_probs, self.rewards = [], [], []

        # TODO: Compute discounted rewards (use the discount_rewards function)
        d_rewards = discount_rewards(rewards, self.gamma)


        # TODO: Compute the optimization term (T1)
        # Task 1 a
        # loss = -torch.mean(action_probs * d_rewards)

        # Task 1 b
        # b = 20
        # loss = -torch.sum(action_probs * (d_rewards - b))

        # Task 1 c
        reward_mean = torch.mean(d_rewards)
        reward_std = torch.std(d_rewards)
        norm_rewards = (d_rewards - reward_mean)/reward_std
        loss = -torch.mean(action_probs * norm_rewards)
        # TODO: Compute the gradients of loss w.r.t. network parameters (T1)
        loss.backward()

        # TODO: Update network parameters using self.optimizer and zero gradients (T1)
        self.optimizer.step()
        self.optimizer.zero_grad()

    def get_action(self, observation, episode_number, evaluation=False):
        x = torch.from_numpy(observation).float().to(self.train_device)

        # TODO: Pass state x through the policy network (T1)
        action_dist = self.policy.forward(x, episode_number)

        # TODO: Return mean if evaluation, else sample from the distribution
        if(evaluation):
            action = action_dist.mean
        else:
            action = action_dist.sample()

        # returned by the policy (T1)
        # TODO: Calculate the log probability of the action (T1)
        act_log_prob = action_dist.log_prob(action)

        return action, act_log_prob

    def store_outcome(self, observation, action_prob, action_taken, reward):
        self.states.append(observation)
        self.action_probs.append(action_prob)
        self.rewards.append(torch.Tensor([reward]))

