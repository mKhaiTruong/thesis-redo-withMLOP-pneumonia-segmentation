import numpy as np
import pandas as pd

def generate_synthetic_metrics(n_samples: int = 10000) -> pd.DataFrame:
    t = np.linspace(0, 1, n_samples)
    
    return pd.DataFrame({
        "cpu_usage": 20 + 30*np.sin(t/10) + np.random.normal(0, 2, n_samples),
        "ram_usage": 40 + 20*np.sin(t/15) + np.random.normal(0, 2, n_samples),
        "latency"  : 0.1 + 0.05*np.sin(t/8) + np.random.normal(0, 0.01, n_samples),
        
        "drift_score": 0.1 + 0.3*(t/100) + np.random.normal(0, 0.02, n_samples),
    })