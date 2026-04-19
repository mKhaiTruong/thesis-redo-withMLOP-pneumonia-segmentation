from prometheus_client import Counter, Histogram, Gauge, make_asgi_app

REQUEST_COUNT = Counter(
    "service_requests_total",
    "Total requests",
    ["service", "endpoint", "status"],
)

REQUEST_LATENCY = Histogram(
    "service_request_latency_seconds",
    "Request latency in seconds",
    ["service", "endpoint"],
)

DRIFT_SCORE = Gauge(
    "inference_drift_score", 
    "Drift score from the current prediction model"
)

IS_DRIFT = Gauge(
    "inference_is_drift",
    "1 if drift, 0 if no drift"
)

def instrument_app(app, service_name: str):
    from fastapi import Request
    import time
    
    @app.middleware("http")
    async def metrics_middleware(request: Request, call_next):
        start_time = time.time()
        response = await call_next(request)
        duration = time.time() - start_time
        
        REQUEST_COUNT.labels(
            service=service_name, 
            endpoint=request.url.path, 
            status=response.status_code
        ).inc()
        
        REQUEST_LATENCY.labels(
            service=service_name, 
            endpoint=request.url.path
        ).observe(duration)
        
        return response
    
    metric_app = make_asgi_app()
    app.mount('/metrics', metric_app)