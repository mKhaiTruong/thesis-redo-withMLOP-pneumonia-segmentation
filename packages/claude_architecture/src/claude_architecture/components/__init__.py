import json, time, os, sys, httpx, anthropic
from core.logging import logger
from core.exception import CustomException
from claude_architecture import ClaudeArchitectureConfig

from dotenv import load_dotenv
load_dotenv()

class ClaudeArchitecture:
    def __init__(self, config: ClaudeArchitectureConfig):
        self.config  = config
        self.client  = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.history = self._load_history()
    
    def _load_history(self) -> list:
        history_path = self.config.history_path
        
        history_path.parent.mkdir(parents=True, exist_ok=True)
        if history_path.exists():
            return json.loads(history_path.read_text())
        return []
        
    def _get_topology_snapshot(self) -> dict:
        q = self._query_prometheus
        return {
            "services": {
                "app": {
                    "cpu":     q('rate(process_cpu_seconds_total{job="app"}[5m]) * 100'),
                    "ram":     q('process_resident_memory_bytes{job="app"} / 1024 / 1024'),
                    "latency": q('histogram_quantile(0.95, rate(service_request_latency_seconds_bucket{job="app", endpoint="/predict"}[5m]))'),
                    "drift":   q('inference_drift_score{job="app"}'),
                },
                "lstm": {
                    "cpu": q('rate(process_cpu_seconds_total{job="lstm"}[5m]) * 100'),
                    "ram": q('process_resident_memory_bytes{job="lstm"} / 1024 / 1024'),
                },
                "dqn": {
                    "cpu": q('rate(process_cpu_seconds_total{job="dqn"}[5m]) * 100'),
                    "ram": q('process_resident_memory_bytes{job="dqn"} / 1024 / 1024'),
                }
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
        
    def _build_prompt(self, topology: dict) -> str:
        history_str = ""
        if self.history:
            history_str = f"\nPrevious architectural decisions:\n{json.dumps(self.history[-3:], indent=2)}\n"
        
    
        return f"""You are an AI system architect managing a medical image segmentation MLOps platform.
            {history_str}
            Current system topology:
            {json.dumps(topology, indent=2)}

            Analyze the system state and decide if architectural evolution is needed.

            Respond ONLY with a JSON object, no explanation, no markdown:
            {{
                "evolution_needed": true/false,
                "action": "none" | "spawn" | "swap" | "rollback",
                "target_service": "app" | "lstm" | "dqn" | "none",
                "parameters": {{}},
                "reasoning": "brief explanation",
                "confidence": 0.0-1.0
            }}

            Rules:
            - spawn: if app latency > 1.0s AND cpu > 70%
            - swap: if drift > 60 (on scale 0-100) AND current model is int8
            - rollback: if latency suddenly increased > 2x after recent action
            - none: if system is healthy
            """
    
    def _parse_response(self, raw: str) -> dict:
        cleaned = raw.replace("```json", "").replace("```", "").strip()
        return json.loads(cleaned)

    def _save_history(self, result: dict, topology: dict):
        self.history.append({
            "timestamp": time.time(),
            "topology_summary": {
                "app_drift": topology["services"]["app"]["drift"],
                "app_cpu":   topology["services"]["app"]["cpu"],
                "app_ram":   topology["services"]["app"]["ram"],
            },
            "action":    result["action"],
            "reasoning": result["reasoning"][:150].encode('ascii', 'ignore').decode()
        })
        self.config.history_path.write_text(json.dumps(self.history[-10:], indent=2))
    
    def _execute_evolution(self, decision: dict) -> None:
        action_map = {
            "spawn":    "scale_out_service",
            "swap":     "swap_model_version",
            "rollback": "rollback"
        }
        
        orchestrator_action = action_map.get(decision["action"])
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