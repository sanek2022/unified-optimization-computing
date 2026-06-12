def compute_latency(P_cold, L_cold, L_warm):
    return P_cold * L_cold + (1 - P_cold) * L_warm


def compute_cost(lambda_i, ce, cp, xi, cm, Mi):
    return ce * lambda_i + cp * xi + cm * Mi


def compute_energy(delta, Mi, Ui):
    return delta * Mi * Ui