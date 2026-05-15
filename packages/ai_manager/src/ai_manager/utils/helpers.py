import numpy as np

def _compute_entropy(q_values: list[float]) -> float:
    q = np.array(q_values)
    q = q - q.max()
    probs = np.exp(q) / np.exp(q).sum()
    entropy = -np.sum(probs * np.log(probs + 1e-8))
    max_entropy = np.log(len(q_values))
    return float(entropy / max_entropy)