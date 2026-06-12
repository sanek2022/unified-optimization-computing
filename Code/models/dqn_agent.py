import torch
import torch.nn as nn
import torch.optim as optim
import random

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, x):
        return self.net(x)


class DQNAgent:
    def __init__(self, state_dim):
        self.actions = ["prewarm", "no_prewarm", "scale_up", "scale_down"]
        self.action_dim = len(self.actions)

        self.q_net = QNetwork(state_dim, self.action_dim)
        self.target_net = QNetwork(state_dim, self.action_dim)

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=1e-3)

        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.995

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)

        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float32)
            q_values = self.q_net(state)
            return torch.argmax(q_values).item()

    def train_step(self, batch):
        states, actions, rewards, next_states = batch

        states = torch.tensor(states, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        actions = torch.tensor(actions)

        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze()

        next_q = self.target_net(next_states).max(1)[0]

        target = rewards + self.gamma * next_q

        loss = nn.MSELoss()(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon *= self.epsilon_decay