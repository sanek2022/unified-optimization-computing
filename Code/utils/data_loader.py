import numpy as np

def get_workload_data(t, window=10):
    """
    Generate synthetic workload (can replace with real logs)
    """
    return np.sin(np.linspace(0, t, window)) + np.random.rand(window) * 0.1