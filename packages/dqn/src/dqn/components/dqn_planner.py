ACTIONS = {
    0: "do_nothing",
    1: "trigger_retraining",
    2: "switch_to_lighter_model",
    3: "scale_up_service",      # legacy
    4: "restart_service",
    5: "scale_out_service",     # new
    6: "scale_in_service",      # new
    7: "swap_model_version",    # new
}

import random, math
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

from core.logging import logger
from dqn import Experience
from dqn import DQN_Planner_Config
from dqn import Simulation_Config
from dqn.components.duel_dqn import DuelingDQN
from dqn.components.buffer import ReplayBuffer

class DQNPlanner:
    def __init__(self, config: DQN_Planner_Config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.epsilon = self.config.dqn_params.epsilon
        
        self.online_net = self._get_network().to(self.device)
        self.target_net = self._get_network().to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()
        
        self.optimizer  = torch.optim.Adam(self.online_net.parameters(), lr=self.config.dqn_params.lr)
        self.buffer     = ReplayBuffer()
        self._ready     = False
        self.step_count = 0
    
    def _get_network(self):
        return DuelingDQN(
            state_size  = self.config.duel_dqn_params.state_size,
            action_size = self.config.duel_dqn_params.action_size,
            hidden_size = self.config.duel_dqn_params.hidden_size
        )
        
    # --------------------------- Inference ------------------------------------
    def load(self):
        if not Path(self.config.model_dir).exists():
            logger.warning("DQN model not found — using untrained model")
            return
        
        checkpoints = torch.load(self.config.model_dir, map_location=self.device)
        self.online_net.load_state_dict(checkpoints["online_net"])
        self.target_net.load_state_dict(checkpoints["target_net"])
        self.epsilon    = checkpoints["epsilon"]
        self.step_count = checkpoints["step_count"]
        self._ready     = True
        logger.info("DQN model loaded")
        
    def plan(self, state: dict) -> str:
        state_tensor = self._dict_to_tensor(state)
        
        if random.random() < self.epsilon:
            action_idx = random.randint(0, self.config.duel_dqn_params.action_size - 1)
            q_spread   = 0.0
        else:
            with torch.no_grad():
                q_val       = self.online_net(state_tensor)
                action_idx  = q_val.argmax().item()
                q_spread    = float(q_val.max() - q_val.min())
        
        action = ACTIONS[action_idx] 
        logger.info(f"DQN action: {action} (epsilon={self.epsilon:.3f})")
        return {"action": action, "q_spread": q_spread}
    
    def _dict_to_tensor(self, state: dict) -> torch.Tensor:
        n = self.config.duel_dqn_params.output_steps  # 5
        values = (
            state.get("current_cpu",         [0.0]) +
            state.get("current_ram",         [0.0]) +
            state.get("current_latency",     [0.0]) +
            state.get("current_drift",       [0.0]) +
            state.get("predicted_cpu",       [0.0]*n) +
            state.get("predicted_ram",       [0.0]*n) +
            state.get("predicted_latency",   [0.0]*n) +
            state.get("predicted_drift",     [0.0]*n)
        )
        return torch.tensor(values, dtype=torch.float32).unsqueeze(0).to(self.device)
    # ------------------------------------------------------------------------------------
    
    # ----------------------- Training DQN (simulation) -------------------------------
    def train(self, config: Simulation_Config, n_episodes: int = 1000):
        from dqn.components.simulation import SystemSimulation
        self.simulation = SystemSimulation(config)
        
        for episode in range(n_episodes):
            total_reward = self._learn_and_reward()
            
            # Decay epsilon
            self.epsilon = max(
                self.config.dqn_params.eps_min, 
                self.epsilon * self.config.dqn_params.eps_decay
            )
            
            if episode % 100 == 0:
                logger.info(f"Episode {episode}/{n_episodes} — Reward: {total_reward:.2f} — Epsilon: {self.epsilon:.3f}")

        self._save()
        self._ready = True
    
    def _learn_and_reward(self):
        state = self.simulation.reset()
        total_reward = 0
            
        for _ in range(200):
            action_idx = self._select_action(state)
            next_state, reward, done = self.simulation.step(action_idx)
                
            self.buffer.push(Experience(state, action_idx, reward, next_state, done))
            self._learn()
                
            state = next_state
            total_reward += reward
            self.step_count += 1
                
            if self.step_count % self.config.dqn_params.target_update_freq == 0:
                self.target_net.load_state_dict(self.online_net.state_dict())
                
            if done:
                break
        return total_reward
    
    
    def _select_action(self, state: np.ndarray) -> int:
        if random.random() < self.epsilon:
            return random.randint(0, self.config.duel_dqn_params.action_size - 1)
        
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.online_net(state_tensor).argmax().item()
    
    
    def _learn(self):
        if len(self.buffer) < self.config.dqn_params.batch_size:
            return
        
        batch       = self.buffer.sample(self.config.dqn_params.batch_size)
        states      = torch.tensor(np.array([e.state  for e in batch]), dtype=torch.float32).to(self.device)
        actions     = torch.tensor(np.array([e.action for e in batch]), dtype=torch.long).to(self.device)
        rewards     = torch.tensor(np.array([e.reward for e in batch]), dtype=torch.float32).to(self.device)
        dones       = torch.tensor(np.array([e.done   for e in batch]), dtype=torch.float32).to(self.device)
        next_states = torch.tensor(np.array([e.next_state for e in batch]), dtype=torch.float32).to(self.device)
        
        # Double DQN: online net chooses action
        with torch.no_grad():
            next_actions = self.online_net(next_states).argmax(dim=1)
            next_q       = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze()
            target_q     = rewards + self.config.dqn_params.gamma * next_q * (1 - dones)
        
        current_q = self.online_net(states).gather(1, actions.unsqueeze(1)).squeeze()
        loss      = F.smooth_l1_loss(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), 10)
        self.optimizer.step()
    
    def _save(self):
        torch.save({
            "online_net":  self.online_net.state_dict(),
            "target_net":  self.target_net.state_dict(),
            "epsilon":     self.epsilon,
            "step_count":  self.step_count,
        }, self.config.model_dir)
        
        logger.info(f"DQN saved -> {self.config.model_dir}")