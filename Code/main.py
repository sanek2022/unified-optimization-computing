from models.transformer import WorkloadPredictor
from models.rl_agent import RLAgent
from utils.data_loader import get_workload_data
from utils.environment import simulate_environment
from utils.metrics import compute_reward
from config import *

def main():

    predictor = WorkloadPredictor()
    agent = RLAgent()

    print("🚀 Starting Unified Cold-Start Optimization Simulation...\n")

    total_reward = 0

    memory = DEFAULT_MEMORY
    P_cold = 0

    for t in range(1, 50):

        # Step 1: Load workload
        workload = get_workload_data(t)

        # Step 2: Predict arrival rate
        lambda_pred = predictor.predict(workload)

        # Step 3: Build state
        state = agent.build_state(lambda_pred, memory, P_cold)

        # Step 4: Choose action
        action = agent.select_action(state)

        # Step 5: Simulate system
        latency, cost, energy, P_cold, memory = simulate_environment(action, lambda_pred)

        # Step 6: Compute reward
        reward = compute_reward(latency, cost, energy, W1, W2, W3)

        total_reward += reward

        print(f"Step {t}")
        print(f"  Lambda: {lambda_pred:.3f}")
        print(f"  Action: {action}")
        print(f"  Latency: {latency:.2f}")
        print(f"  Cost: {cost:.2f}")
        print(f"  Energy: {energy:.2f}")
        print(f"  Reward: {reward:.2f}\n")

    print("✅ Simulation Complete")
    print(f"Total Reward: {total_reward:.2f}")


if __name__ == "__main__":
    main()