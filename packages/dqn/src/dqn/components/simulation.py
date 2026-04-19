import numpy as np

"""
    Simulate state of microservices for DQN training
    State: [
        cpu, ram, latency, drift, 
        predicted_cpu*4, predicted_ram*4, predicted_latency*4, predicted_drift*4
    ]
"""

from dqn import Simulation_Config

class SystemSimulation:
    def __init__(self, config: Simulation_Config):
        self.config     = config
        self.state_size = 20
        self.reset()
        
    def reset(self) -> np.ndarray:
        self.cpu     = np.random.uniform(20, 50)
        self.ram     = np.random.uniform(30, 60)
        self.latency = np.random.uniform(0.05, 0.3)
        self.drift   = np.random.uniform(0.0, 0.2)
        self.step_n  = 0
        return self._get_state()

    def _get_state(self) -> np.ndarray:
        current   = [self.cpu, self.ram, self.latency, self.drift]
        predicted = current * 4
        return np.array(current + predicted, dtype=np.float32)
    
    def step(self, action: int) -> tuple[np.ndarray, float, bool]:
        self.step_n += 1
        
        self.cpu     += np.random.normal(0, 2)
        self.ram     += np.random.normal(0, 1)
        self.latency += np.random.normal(0, 0.01)
        self.drift   += np.random.normal(0, 0.01)
        
        if action == 1:
            self.drift   = max(0, self.drift - 0.3)
        elif action == 2:
            self.ram     = max(0, self.ram - 20)
            self.cpu     = max(0, self.cpu - 15)
            self.latency += 0.05
        elif action == 3:
            self.latency = max(0, self.latency - 0.1)
            self.ram     += 10
        elif action == 4:
            self.cpu     = 20.0
            self.latency = 0.1
        
        self.cpu     = np.clip(self.cpu,     0, 100)
        self.ram     = np.clip(self.ram,     0, 100)
        self.latency = np.clip(self.latency, 0, 5)
        self.drift   = np.clip(self.drift,   0, 1)
        
        reward = self._compute_reward(action)
        done   = self.step_n >= 200 or \
                    self.ram > self.config.ram_critical or \
                        self.cpu > self.config.cpu_critical
                        
        return self._get_state(), reward, done
    
    def _compute_reward(self, action: int) -> float:
        reward = 0.0
        
        if self.ram     > self.config.ram_warning:     reward -= 3.0
        if self.cpu     > self.config.cpu_warning:     reward -= 2.0
        if self.latency > self.config.latency_warning: reward -= 2.0
        if self.drift   > self.config.drift_warning:   reward -= 2.0
        
        if action != 0:
            if (self.ram     < self.config.ram_warning  and
                self.cpu     < self.config.cpu_warning  and
                self.latency < self.config.latency_warning and
                self.drift   < self.config.drift_warning):
                reward -= 1.0
        
        if (self.ram     < self.config.ram_warning      and
            self.cpu     < self.config.cpu_warning      and
            self.latency < self.config.latency_warning  and
            self.drift   < self.config.drift_warning):
            reward += 1.0
        
        return reward