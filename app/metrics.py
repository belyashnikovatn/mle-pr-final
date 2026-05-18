"""Метрики для мониторинга веб-сервиса."""

from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response
import time
from functools import wraps


# Определяем метрики
REQUEST_COUNT = Counter(
    'api_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)

PREDICTION_LATENCY = Histogram(
    'prediction_latency_seconds',
    'Prediction latency in seconds',
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0]
)

ACTIVE_REQUESTS = Gauge(
    'active_requests',
    'Number of active requests'
)

MODEL_LOADED = Gauge(
    'model_loaded',
    'Whether model is loaded (1=loaded, 0=not loaded)'
)


def track_metrics(endpoint: str):
    """Декоратор для отслеживания метрик."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            ACTIVE_REQUESTS.inc()
            start_time = time.time()
            status = "success"
            try:
                result = await func(*args, **kwargs)
                return result
            except Exception:
                status = "error"
                raise
            finally:
                latency = time.time() - start_time
                PREDICTION_LATENCY.observe(latency)
                REQUEST_COUNT.labels(
                    method="POST",
                    endpoint=endpoint,
                    status=status
                ).inc()
                ACTIVE_REQUESTS.dec()
        return wrapper
    return decorator


def get_metrics():
    """Возвращает метрики в формате Prometheus."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)