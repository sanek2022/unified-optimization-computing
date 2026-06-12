import pandas as pd
import matplotlib.pyplot as plt

# Load your real training log
df = pd.read_csv("results/training_log.csv")

# ✅ 1. Reward Curve
plt.figure()
plt.plot(df["episode"], df["reward"])
plt.title("Training Reward over Episodes")
plt.xlabel("Episodes")
plt.ylabel("Reward")
plt.savefig("results/reward_curve.png")

# ✅ 2. Latency Trend
plt.figure()
plt.plot(df["episode"], df["avg_latency"], label="Avg Latency")
plt.title("Latency Reduction Over Training")
plt.xlabel("Episodes")
plt.ylabel("Latency (ms)")
plt.savefig("results/latency_trend.png")

# ✅ 3. Cost vs Latency
plt.figure()
plt.plot(df["avg_cost"], df["avg_latency"])
plt.title("Cost vs Latency Tradeoff")
plt.xlabel("Cost")
plt.ylabel("Latency")
plt.savefig("results/cost_latency_tradeoff.png")

# ✅ 4. Energy Trend
plt.figure()
plt.plot(df["episode"], df["avg_energy"], color='green')
plt.title("Energy Consumption")
plt.xlabel("Episodes")
plt.ylabel("Energy")
plt.savefig("results/energy_curve.png")

print("✅ All plots generated successfully!")