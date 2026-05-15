import json, time, os, sys, httpx, anthropic
from core.logging import logger
from core.exception import CustomException
from core.metrics_queries import QUERIES
from claude_architecture import ClaudeArchitectureConfig

from dotenv import load_dotenv
load_dotenv()

class ClaudeArchitecture:
    def __init__(self, config: ClaudeArchitectureConfig):
        self.config  = config
        self.client  = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.history = self._load_history()
        self._evolution_cooldown = 300
        self._last_evolution_time = {}
    
    def _load_history(self) -> list:
        history_path = self.config.history_path
        
        history_path.parent.mkdir(parents=True, exist_ok=True)
        if history_path.exists():
            return json.loads(history_path.read_text())
        return []
    
    
    def get_claude_decision(self) -> dict:
        try:
            topology  = self._get_topology_snapshot()
            
            response  = self.client.messages.create(
                model = self.config.params.model,
                max_tokens = self.config.params.max_tokens,
                messages = [{
                    "role": "user",
                    "content": self._build_prompt(topology)
                }]
            )       
        
            result = self._parse_response(response.content[0].text)
            self._save_history(result, topology)
            logger.info(f"Claude architect: {result['action']} — {result['reasoning']}")
            
            if result.get("evolution_needed") and result["action"] != "none":
                self._execute_evolution(result)
            
            return result
        except Exception as e:
            raise CustomException(e, sys)
    
    
        
    def _get_topology_snapshot(self) -> dict:
        q = self._query_prometheus
        lstm_prediction = self._get_lstm_prediction()
        
        return {
            "services": {
                "app": {
                    "current": {
                        "ram":      q(QUERIES["app_ram"]),
                        "latency":  q(QUERIES["app_latency"]),
                        "drift":    q(QUERIES["app_drift"]),
                        "requests": q(QUERIES["app_requests"]),
                    },
                    "predicted_next_5min": lstm_prediction,
                },
                "lstm": {"ram": q(QUERIES["lstm_ram"])},
                "dqn":  {"ram": q(QUERIES["dqn_ram"])},
            }
        }
    
    def _query_prometheus(self, query: str) -> float:
        try: 
            r = httpx.get(
                f"{self.config.prometheus_url}/api/v1/query", 
                params={"query": query}
            )
            
            result = r.json()["data"]["result"]
            return float(result[0]["value"][1]) if result else 0.0
        except Exception:
            return 0.0
    
    def _get_lstm_prediction(self) -> dict:
        try: 
            r = httpx.post(
                f"{self.config.lstm_url}/predict",
                timeout=10.0
            )
            
            return r.json()
        except Exception:
            return {"status": "unavailable"}
        
    def _build_prompt(self, topology: dict) -> str:
        history_str = ""
        if self.history:
            history_str = f"\nPrevious architectural decisions:\n{json.dumps(self.history[-3:], indent=2)}\n"
        
    
        return f"""You are an AI system architect managing a medical image segmentation MLOps platform.
            {history_str}
            Current system topology:
            {json.dumps(topology, indent=2)}

            The "predicted_next_5min" field contains LSTM forecasts for the next 5 minutes.
            Use BOTH current metrics AND predictions to decide — act before thresholds are breached, not after.

            Respond ONLY with a JSON object, no explanation, no markdown:
            {{
                "evolution_needed": true/false,
                "action": "none" | "spawn" | "swap" | "rollback",
                "target_service": "app" | "lstm" | "dqn" | "none",
                "parameters": {{}},
                "reasoning": "brief explanation referencing both current and predicted values",
                "confidence": 0.0-1.0
            }}

            Rules — HARD thresholds, no exceptions:
            - If predicted metrics unavailable: reason from current metrics ONLY, still MUST follow threshold rules
            - spawn: ONLY if current latency > 1.0s OR any predicted_latency value > 1.0s
            - swap:  ONLY if current drift > 0.6 OR any predicted_drift value > 0.6
            - rollback: ONLY if latency increased > 2x after a recent spawn/swap action
            - none: if ALL current AND predicted metrics below thresholds

            FORBIDDEN:
            - Do NOT act on "trends" or "gradual increases" unless threshold is actually breached
            - Do NOT use "proactive" as justification for action below threshold
            - predicted_drift=0.237 with threshold=0.6 → action MUST be "none"
            """
    
    def _parse_response(self, raw: str) -> dict:
        cleaned = raw.replace("```json", "").replace("```", "").strip()
        start = cleaned.find("{")
        end = cleaned.rfind("}") + 1
        if start == -1 or end == 0:
            raise ValueError(f"No JSON found in response: {cleaned}")
    
        return json.loads(cleaned[start:end])

    def _save_history(self, result: dict, topology: dict):
        self.history.append({
            "timestamp": time.time(),
            "topology_summary": {
                "app_drift":   topology["services"]["app"]["current"]["drift"],
                "app_latency": topology["services"]["app"]["current"]["latency"],
                "app_ram":     topology["services"]["app"]["current"]["ram"],
            },
            "action":    result["action"],
            "reasoning": result["reasoning"][:150].encode('ascii', 'ignore').decode()
        })
        self.config.history_path.write_text(json.dumps(self.history[-10:], indent=2))
    
    def _execute_evolution(self, decision: dict) -> None:
        action = decision["action"]
        last = self._last_evolution_time.get(action, 0)
        
        if (time.time() - last) < self._evolution_cooldown:
            logger.info(f"Evolution cooldown active, skipping: {action}")
            return

        self._last_evolution_time[action] = time.time()
        
        action_map = {
            "spawn":    "scale_out_service",
            "swap":     "swap_model_version",
            "rollback": "rollback"
        }
        
        orchestrator_action = action_map.get(action)
        if not orchestrator_action:
            return
        
        try: 
            res = httpx.post(
                f"{self.config.orchestrator_url}/execute/{orchestrator_action}",
                timeout=30
            )
            logger.info(f"Evolution executed: {orchestrator_action} → {res.json()}")
        except Exception as e:
            logger.error(f"Evolution execution failed: {e}")