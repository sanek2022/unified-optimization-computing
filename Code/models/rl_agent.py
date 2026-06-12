import random

class RLAgent:
    def __init__(self):
        self.actions = ["prewarm", "no_prewarm", "scale_up", "scale_down"]

    def build_state(self, lambda_pred, memory, P_cold):
        return {
            "lambda": lambda_pred,
            "memory": memory,
            "cold_prob": P_cold
        }

    def select_action(self, state):
        # Simple random (replace with DQN later)
        return random.choice(self.actions)

    def update(self, state, action, reward):
        # Placeholder for learning
        pass