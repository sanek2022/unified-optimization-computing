import matplotlib.pyplot as plt

def plot_training(rewards):

    plt.figure()
    plt.plot(rewards)
    plt.title("Training Reward over Episodes")
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.savefig("results/reward_curve.png")


def plot_comparison(latency_with, latency_without):

    plt.figure()
    plt.plot(latency_without, label="Without Optimization")
    plt.plot(latency_with, label="With UCSOM + PPO")

    plt.title("Latency Comparison")
    plt.xlabel("Time")
    plt.ylabel("Latency")
    plt.legend()

    plt.savefig("results/latency_comparison.png")
``