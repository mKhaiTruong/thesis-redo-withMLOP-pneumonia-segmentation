import json, os, sys, httpx, anthropic
from core.logging import logger
from core.exception import CustomException
from claude_architecture import ClaudeArchitectureConfig

from dotenv import load_dotenv
load_dotenv()

class ClaudeArchitecture:
    def __init__(self, config: ClaudeArchitectureConfig):
        self.config = config
        self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        
    def _get_topology_snapshot(self) -> dict:
        q = self._query_prometheus
        return {
            "services": {
                "app": {
                    "cpu":     q('rate(process_cpu_seconds_total{job="app"}[5m]) * 100'),
                    "ram":     q('process_resident_memory_bytes{job="app"} / 1024 / 1024'),
                    "latency": q('http_request_duration_seconds{job="app", endpoint="/predict", quantile="0.95"}'),
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
        except Exception as e:
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
            logger.info(f"Claude architect: {result['action']} — {result['reasoning']}")
            return result
        except Exception as e:
            raise CustomException(e, sys)
        
    def _build_prompt(self, topology: dict) -> str:
        return f"""You are an AI system architect managing a medical image segmentation MLOps platform.
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
            - swap: if drift > 0.8 AND current model is int8
            - rollback: if latency suddenly increased > 2x after recent action
            - none: if system is healthy
            """
    
    def _parse_response(self, raw: str) -> dict:
        cleaned = raw.replace("```json", "").replace("```", "").strip()
        return json.loads(cleaned)