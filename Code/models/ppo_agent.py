import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()

        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )

        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, state):
        probs = self.actor(state)
        value = self.critic(state)
        return probs, value


class PPOAgent:
    def __init__(self, state_dim):
        self.actions = ["prewarm", "no_prewarm", "scale_up", "scale_down"]

        self.model = ActorCritic(state_dim, len(self.actions))
        self.optimizer = optim.Adam(self.model.parameters(), lr=3e-4)

        self.gamma = 0.99
        self.clip = 0.2

    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32)
        probs, _ = self.model(state)
        dist = Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    def update(self, states, actions, rewards, old_log_probs):

        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)

        returns = torch.tensor(returns, dtype=torch.float32)

        for _ in range(5):
            for i in range(len(states)):
                state = torch.tensor(states[i], dtype=torch.float32)
                action = torch.tensor(actions[i])

                probs, value = self.model(state)
                dist = Categorical(probs)

                new_log_prob = dist.log_prob(action)
                ratio = torch.exp(new_log_prob - old_log_probs[i])

                advantage = returns[i] - value

                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1-self.clip, 1+self.clip) * advantage

                loss = -torch.min(surr1, surr2) + 0.5 * advantage**2

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
