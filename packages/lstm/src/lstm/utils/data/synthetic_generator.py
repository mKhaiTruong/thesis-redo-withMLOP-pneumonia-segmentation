import numpy as np
import pandas as pd

def generate_synthetic_metrics(n_samples: int = 10000) -> pd.DataFrame:
    rng = np.random.default_rng(seed=42)  # reproducible
    t   = np.linspace(0, 6 * np.pi, n_samples)
    
    # RAM: baseline ~400MB, slow oscillation, occasional memory pressure spikes
    ram_base   = 400 + 200 * np.sin(t / 15)
    ram_spikes = rng.choice([0.0, 400.0], size=n_samples, p=[0.97, 0.03])
    ram        = np.clip(ram_base + ram_spikes + rng.normal(0, 30, n_samples), 100, 1200)
    
    # Latency P95: baseline ~0.3s, rises with load, spikes under pressure
    latency_base   = 0.3 + 0.4 * np.sin(t / 8)
    latency_spikes = rng.choice([0.0, 6.0], size=n_samples, p=[0.97, 0.03])
    latency        = np.clip(latency_base + latency_spikes + rng.normal(0, 0.05, n_samples), 0.05, 10.0)
    
    # Drift: gradual accumulation, periodic resets (mimicking model reloads)
    drift_raw    = 15 + 50 * (t % (2 * np.pi)) / (2 * np.pi)
    drift_resets = (t % (2 * np.pi)) < 0.05
    drift        = np.where(drift_resets, 5.0, drift_raw)
    drift        = np.clip(drift + rng.normal(0, 2, n_samples), 0.0, 100.0)
    
    # Requests: correlated with latency (more requests → higher latency)
    requests = np.clip(20 + 80 * np.sin(t / 12) + rng.normal(0, 10, n_samples), 1, 200)
    
    return pd.DataFrame({
        "ram":      ram,
        "latency":  latency,
        "drift":    drift,
        "requests": requests,
    })