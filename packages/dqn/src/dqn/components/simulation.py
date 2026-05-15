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
        self.state_size = 24
        self.reset()
        
    def reset(self) -> np.ndarray:
        scenario = np.random.choice(
            ['healthy', 'high_drift', 'high_cpu', 'high_load'],
            p=[0.2, 0.5, 0.2, 0.1]
        )
        
        if scenario == 'healthy':
            self.cpu     = np.random.uniform(0.2, 0.5)
            self.ram     = np.random.uniform(0.3, 0.6)
            self.latency = np.random.uniform(0.01, 0.06)
            self.drift   = np.random.uniform(0.0, 0.2)
        elif scenario == 'high_drift':
            self.cpu     = np.random.uniform(0.1, 0.4)
            self.ram     = np.random.uniform(0.3, 0.6)
            self.latency = np.random.uniform(0.01, 0.05)
            self.drift   = np.random.uniform(0.35, 0.8)
        elif scenario == 'high_cpu':
            self.cpu     = np.random.uniform(0.75, 0.95)
            self.ram     = np.random.uniform(0.5, 0.8)
            self.latency = np.random.uniform(0.05, 0.15)
            self.drift   = np.random.uniform(0.0, 0.2)
        elif scenario == 'high_load':
            self.cpu     = np.random.uniform(0.7, 0.9)
            self.ram     = np.random.uniform(0.6, 0.85)
            self.latency = np.random.uniform(0.08, 0.2)
            self.drift   = np.random.uniform(0.2, 0.5)
        
        self.step_n = 0
        return self._get_state()

    def _get_state(self) -> np.ndarray:
        cpu, ram, lat, drift = self.cpu, self.ram, self.latency, self.drift
        current  = [cpu, ram, lat, drift]
        predicted = []
        for _ in range(self.config.output_steps):
            cpu   = np.clip(cpu   + np.random.normal(0, 0.02),  0, 1)
            ram   = np.clip(ram   + np.random.normal(0, 0.01),  0, 1)
            lat   = np.clip(lat   + np.random.normal(0, 0.001), 0, 1)
            drift = np.clip(drift + np.random.normal(0, 0.01),  0, 1)
            predicted += [cpu, ram, lat, drift]
        return np.array(current + predicted, dtype=np.float32)
    
    def step(self, action: int):
        self.step_n  += 1
        self.cpu     += np.random.normal(0, 0.02)
        self.ram     += np.random.normal(0, 0.01)
        self.latency += np.random.normal(0, 0.001)
        self.drift   += np.random.normal(0, 0.01)

        if action == 1:
            self.drift   = max(0, self.drift - 0.3)
        elif action == 2:
            self.ram     = max(0, self.ram - 0.2)
            self.cpu     = max(0, self.cpu - 0.15)
            self.latency += 0.05
        elif action == 3:
            self.latency = max(0, self.latency - 0.1)
            self.ram     += 0.1
        elif action == 4:
            self.latency = max(0, self.latency - 0.1)
            self.ram     += 0.05
        elif action == 5:
            self.latency = max(0, self.latency - 0.15)
            self.cpu     = max(0, self.cpu - 0.1)
        elif action == 6:
            self.latency += 0.05
            self.cpu     += 0.05
        elif action == 7:
            self.latency += 0.1
            self.ram     += 0.05

        self.cpu     = np.clip(self.cpu,     0, 1)
        self.ram     = np.clip(self.ram,     0, 1)
        self.latency = np.clip(self.latency, 0, 1)
        self.drift   = np.clip(self.drift,   0, 1)

        reward = self._compute_reward(action)
        done   = self.step_n >= 200 or \
                self.ram  > self.config.ram_critical or \
                self.cpu  > self.config.cpu_critical
        return self._get_state(), reward, done
    
    def _compute_reward(self, action: int) -> float:
        reward = 0.0

        if self.drift > self.config.drift_critical:
            reward -= 2.0
            if action == 0: reward -= 1.0
        elif self.drift > self.config.drift_warning:
            reward -= 0.8
            if action == 0: reward -= 0.5

        if self.cpu > self.config.cpu_critical:
            reward -= 1.5
            if action == 0: reward -= 0.5
        elif self.cpu > self.config.cpu_warning:
            reward -= 0.5

        if self.ram > self.config.ram_critical:
            reward -= 1.5
            if action == 0: reward -= 0.5
        elif self.ram > self.config.ram_warning:
            reward -= 0.5

        if self.latency > self.config.latency_critical:
            reward -= 1.0
            if action == 0: reward -= 0.3
        elif self.latency > self.config.latency_warning:
            reward -= 0.3

        if action != 0:
            if (self.drift   < self.config.drift_warning and
                self.cpu     < self.config.cpu_warning   and
                self.ram     < self.config.ram_warning   and
                self.latency < self.config.latency_warning):
                reward -= 1.0

        if (self.drift   < self.config.drift_warning and
            self.cpu     < self.config.cpu_warning   and
            self.ram     < self.config.ram_warning   and
            self.latency < self.config.latency_warning):
            reward += 0.05

        return reward