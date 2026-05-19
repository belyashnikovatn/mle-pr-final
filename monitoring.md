# Мониторинг модели рекомендаций банковских продуктов

Документ описывает мониторинг **веб-сервиса** рекомендаций: метрики собираются в Prometheus и визуализируются в Grafana.

## Архитектура

```text
FastAPI (app/)  --GET /metrics-->  Prometheus  -->  Grafana
     :8000                            :9090            :3000
```

| Компонент | Файл / путь | Назначение |
|-----------|-------------|------------|
| Экспорт метрик | [app/metrics.py](app/metrics.py) | `prometheus_client`, декоратор `@track_metrics` |
| Эндпоинт | `GET /metrics` | Формат Prometheus text |
| Health | `GET /health` | JSON (не дублируется в Prometheus) |
| Scrape | [prometheus/prometheus.yml](prometheus/prometheus.yml) | Опрос `bank-recommender:8000` каждые 10 с |
| Дашборд | [grafana/dashboards/bank_recommender.json](grafana/dashboards/bank_recommender.json) | Bank Recommender Monitoring |

Запуск стека: см. [README.md](README.md) — раздел «Запуск стека: API + Prometheus + Grafana» (`docker compose up -d`).

---

## Метрики в Prometheus (реализовано)

Сбор ведётся **только для `POST /predict`** (декоратор `@track_metrics` в [app/main.py](app/main.py)).

| Метрика | Тип | Labels | Описание |
|---------|-----|--------|----------|
| `api_requests_total` | Counter | `method`, `endpoint`, `status` | Число запросов к `/predict`; `status`: `success` или `error` |
| `prediction_latency_seconds` | Histogram | — | Время обработки `/predict` (секунды) |
| `active_requests` | Gauge | — | Запросы в обработке (обычно близко к 0) |
| `model_loaded` | Gauge | — | `1` — модель загружена, `0` — нет |

### Целевые значения (технические)

| Показатель | Цель | Как смотреть в Grafana |
|------------|------|-------------------------|
| Latency p95 | < 200 ms | панель **Prediction Latency (p95)** |
| Throughput | по нагрузке | **Requests per Second** |
| Ошибки инференса | минимум | `rate(api_requests_total{status="error"}[5m])` |
| Модель загружена | всегда `1` | **Model Status** |

На тестовой нагрузке (`python test_service.py`) p95 latency ~50 ms, `model_loaded = 1`.

### Примеры PromQL

```promql
# Запросов в секунду
sum(rate(api_requests_total[1m]))

# p95 латентности /predict
histogram_quantile(0.95, sum(rate(prediction_latency_seconds_bucket[5m])) by (le))

# Доля ошибок (исключения в predict)
sum(rate(api_requests_total{status="error"}[5m]))
/
sum(rate(api_requests_total[5m]))

# Статус модели
model_loaded
```

---

## Реализация в коде

Метрики объявлены в [app/metrics.py](app/metrics.py):

```python
REQUEST_COUNT = Counter(
    'api_requests_total', 'Total HTTP requests',
    ['method', 'endpoint', 'status'],
)
PREDICTION_LATENCY = Histogram(
    'prediction_latency_seconds', 'Prediction latency in seconds',
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0],
)
ACTIVE_REQUESTS = Gauge('active_requests', 'Number of active requests')
MODEL_LOADED = Gauge('model_loaded', 'Whether model is loaded (1=loaded, 0=not loaded)')
```

При старте сервиса `MODEL_LOADED` устанавливается в `1` в [app/handler.py](app/handler.py) после успешной загрузки `model.bin`.

Эндпоинт метрик:

```python
@app.get("/metrics")
async def metrics():
    return get_metrics()  # Content-Type: text/plain; Prometheus format
```

---

## Дашборд Grafana

**Название:** Bank Recommender Monitoring  
**URL:** http://localhost:3000 (логин `admin` / `admin`)

| Панель | PromQL / источник |
|--------|-------------------|
| Requests per Second | `sum(rate(api_requests_total[1m]))` |
| Prediction Latency (p95) | `histogram_quantile(0.95, sum(rate(prediction_latency_seconds_bucket[5m])) by (le))` |
| Model Status | `model_loaded` |
| Active Requests | `active_requests` |
| Total Predict Requests (last 5 min) | `sum(increase(api_requests_total[5m]))` |

Рекомендуемый интервал: **Last 15 minutes**, автообновление **5s**.  
Скриншот при нагрузке — в [README.md](README.md) (раздел про Grafana).

> **Важно:** `rate()` показывает скорость **в текущий момент**. После окончания тестов графики RPS могут упасть в 0 — это нормально. Для проверки запустите нагрузку во время просмотра: `python test_service.py`.

---

## Проверка вручную

```bash
# Сырой вывод метрик
curl http://localhost:8000/metrics

# Health (отдельно от Prometheus)
curl http://localhost:8000/health

# Prometheus UI
open http://localhost:9090
```

Пример фрагмента `/metrics`:

```text
api_requests_total{endpoint="/predict",method="POST",status="success"} 42
prediction_latency_seconds_bucket{le="0.1"} 35
model_loaded 1.0
```

---

## Метрики качества модели (офлайн)

Не экспортируются в Prometheus в текущей версии; пересчитываются в ноутбуке / batch-джобе.

| Метрика | Назначение | Где считается |
|---------|------------|---------------|
| **MAP@7** | Основная метрика рекомендаций | [final_project.ipynb](final_project.ipynb), MLflow run `catboost_ovr_baseline` |
| **Recall@7** | Полнота топ-7 | тот же ноутбук |
| **PR-AUC** | Качество ранжирования при дисбалансе | тот же ноутбук |
| **PSI** | Дрейф признаков (`age`, `renta`, `canal_entrada`) | планируется: еженедельный batch |
| **Product coverage** | Частота рекомендаций по продуктам (перекос) | планируется: анализ логов / batch |

Актуальные значения MAP@7 (30% клиентов, 2016): validation **0.3157**, test **0.3144** — см. [README.md](README.md).

---

## Рекомендуемые алерты (пример)

| Условие | PromQL / проверка | Действие |
|---------|-------------------|----------|
| Модель не загружена | `model_loaded == 0` | Перезапуск сервиса, проверить `app/model.bin` |
| Высокая латентность | p95 > 0.5 s за 5m | Масштабирование, профилирование |
| Рост ошибок | error rate > 5% за 5m | Логи `bank-recommender`, проверка входных данных |
| Сервис недоступен | `up{job="bank-recommender"} == 0` | Проверить контейнер / порт 8000 |

---

## Ограничения текущей реализации

- Метрики HTTP **только для `/predict`**; `/health` и `/metrics` в счётчики не попадают.
- `status` в counter — `success` / `error` (исключение), а не коды HTTP 400/500 отдельно.
- `active_requests` почти всегда 0: запросы короткие (~20 ms), scrape раз в 10 s.
- MAP@7 и дрейф данных — вне runtime-мониторинга, только офлайн.

---

## Зависимости

- `prometheus-client` — в [requirements.txt](requirements.txt)
- Образы: `prom/prometheus`, `grafana/grafana` — [docker-compose.yml](docker-compose.yml)
