import pandas as pd

log_data = []

for episode in range(50):

    total_reward = 0
    latency_list = []
    cost_list = []
    energy_list = []

    for t in range(20, len(data)):

        # (your existing code)

        latency_list.append(latency)
        cost_list.append(cost)
        energy_list.append(energy)

        total_reward += reward

    # Save episode stats
    log_data.append({
        "episode": episode,
        "reward": total_reward,
        "avg_latency": sum(latency_list)/len(latency_list),
        "avg_cost": sum(cost_list)/len(cost_list),
        "avg_energy": sum(energy_list)/len(energy_list)
    })

# ✅ Save to CSV
df = pd.DataFrame(log_data)
df.to_csv("results/training_log.csv", index=False)