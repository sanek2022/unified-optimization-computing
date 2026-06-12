import math
import random
from config import *

def compute_cold_probability(lambda_i, x_i):
    return (1 - x_i) * math.exp(-lambda_i * TEARDOWN_TIME)

def simulate_environment(action, lambda_pred):
    """
    Simulates serverless system response
    """

    # Interpret action
    if action == "prewarm":
        x_i = 1
    else:
        x_i = 0

    Mi = DEFAULT_MEMORY + random.randint(-32, 32)

    P_cold = compute_cold_probability(lambda_pred, x_i)

    # Latency
    latency = P_cold * L_COLD + (1 - P_cold) * L_WARM

    # Cost
    cost = C_E * lambda_pred + C_P * x_i + C_M * Mi

    # Energy
    utilization = lambda_pred / (lambda_pred + 1)
    energy = DELTA * Mi * utilization

    return latency, cost, energy, P_cold, Mi