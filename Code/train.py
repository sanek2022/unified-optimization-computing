import math
from pathlib import Path

import pandas as pd

from config import DEFAULT_MEMORY, W1, W2, W3
from models.rl_agent import RLAgent
from models.transformers import WorkloadPredictor
from utils.data_loader import get_sequence, load_dataset
from utils.environment import simulate_environment
from utils.metrics import compute_reward


def _generate_synthetic_workload(length=200):
    """Fallback workload if no dataset file is present."""
    return [
        max(0.05, 0.8 + 0.4 * math.sin(i / 7.0) + 0.2 * math.cos(i / 13.0))
        for i in range(length)
    ]


def _load_workload_data(project_root):
    dataset_path = project_root / "data" / "workload.csv"
    if dataset_path.exists():
        return load_dataset(str(dataset_path))
    return _generate_synthetic_workload()


def main():
    project_root = Path(__file__).resolve().parent.parent
    results_dir = project_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    data = _load_workload_data(project_root)
    predictor = WorkloadPredictor()
    agent = RLAgent()

    log_data = []
    episodes = 50
    window = 10

    print("🚀 Starting training...")

    for episode in range(episodes):
        total_reward = 0.0
        latency_list = []
        cost_list = []
        energy_list = []

        memory = DEFAULT_MEMORY
        p_cold = 0.0

        for t in range(window, len(data)):
            workload_seq = get_sequence(data, t, window=window)
            lambda_pred = predictor.predict(workload_seq)

            state = agent.build_state(lambda_pred, memory, p_cold)
            action = agent.select_action(state)

            latency, cost, energy, p_cold, memory = simulate_environment(action, lambda_pred)
            reward = compute_reward(latency, cost, energy, W1, W2, W3)

            latency_list.append(latency)
            cost_list.append(cost)
            energy_list.append(energy)
            total_reward += reward

        log_data.append(
            {
                "episode": episode,
                "reward": total_reward,
                "avg_latency": sum(latency_list) / len(latency_list),
                "avg_cost": sum(cost_list) / len(cost_list),
                "avg_energy": sum(energy_list) / len(energy_list),
            }
        )

        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}/{episodes} | Reward: {total_reward:.2f}")

    output_path = results_dir / "training_log.csv"
    pd.DataFrame(log_data).to_csv(output_path, index=False)
    print(f"✅ Training complete. Saved log to: {output_path}")


if __name__ == "__main__":
    main()