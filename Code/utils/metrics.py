def compute_reward(latency, cost, energy, w1, w2, w3):
	"""Higher reward is better, so we negate weighted objectives."""
	return -(w1 * latency + w2 * cost + w3 * energy)
