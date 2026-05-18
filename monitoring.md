# Мониторинг модели рекомендаций банковских продуктов

## Метрики для отслеживания

### Технические метрики
- **Latency (p50, p95)** – время ответа API (цель < 200 мс).
- **Throughput** – количество запросов в секунду.
- **Error rate** – доля ответов с HTTP 4xx/5xx.

### Метрики качества модели
- **MAP@7** – основная метрика рекомендаций. Пересчитывается на свежих данных (например, раз в неделю).
- **PSI (Population Stability Index)** – отслеживание дрейфа признаков (`age`, `renta`, `canal_entrada`).
- **Product coverage** – какие продукты чаще всего рекомендуются (чтобы избежать перекоса).

## Отправка метрик
В коде приложения используйте `prometheus_client`:
```python
from prometheus_client import Counter, Histogram, start_http_server

REQUESTS = Counter('api_requests_total', 'Total requests')
LATENCY = Histogram('prediction_latency_seconds', 'Prediction latency')