import httpx
import numpy as np
from datetime import datetime, timedelta, timezone
from core.logging import logger

def _query_prometheus(
                        query:          str,
                        input_steps:    int,
                        prometheus_url: str,
                        scale:          float = 1.0,
                        step_seconds:   int   = 15,
                    ) -> list[float]:
        
        window_seconds = input_steps * step_seconds + 60   # +60s buffer
        end     = datetime.now(timezone.utc)
        start   = end - timedelta(seconds=window_seconds)
        
        try:
            response = httpx.get(
                f"{prometheus_url}/api/v1/query_range",
                params={
                    "query": query,
                    "start": start.isoformat(),
                    "end":   end.isoformat(),
                    "step": "15s"
                },
                timeout=10.0,
            )
            response.raise_for_status()
        except httpx.HTTPError as e:
            logger.warning(f"Prometheus request failed | query={query} | error={e}")
            return [0.0] * input_steps
        
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
        values = [0.0 if not np.isfinite(v) else v for v in values]
        
        # Pad left with first value if not enough history yet
        if len(values) < input_steps:
            values = [values[0]] * (input_steps - len(values)) + values
            
        return values[-input_steps:]