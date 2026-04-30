import httpx
from datetime import datetime, timedelta, timezone
from core.logging import logger

def _query_prometheus(query: str, scale: float, input_steps: int, prometheus_url: str) -> list[float]:
        end     = datetime.now(timezone.utc)
        start   = end - timedelta(minutes=input_steps * 15 // 60 + 5)
        
        response = httpx.get(
            f"{prometheus_url}/api/v1/query_range",
            params={
                "query": query,
                "start": start.isoformat(),
                "end":   end.isoformat(),
                "step": "15s"
            }
        )
        
        body = response.json()
        
        # --- debug ---
        if body.get("status") != "success":
            logger.warning(f"Prometheus query failed | query={query} | response={body}")
            return [0.0] * input_steps
        
        result = body["data"]["result"]
        if not result:
            logger.warning(f"Prometheus empty result | query={query}")
            return [0.0] * input_steps
        
        values = [float(v[1]) / scale for v in result[0]["values"]]
        if len(values) < input_steps:
            values = [values[0]] * (input_steps - len(values)) + values
        return values[-input_steps:]