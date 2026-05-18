# Проект. Рекомендательная система в банковской сфере.

## Цель

Предсказать, какие банковские продукты наиболее релевантны для клиента, и предложить их в виде персонализированной рекомендации.

## Основные задачи и артефакты

| Задача | Артефакты | Ссылки на артефакты |
|--------|-----------|---------------------|
| 1. Исследование данных. Проведите первичный анализ данных в Jupyter Notebook и опишите увиденные в них закономерности. | Jupyter Notebook с EDA. | [final_project.ipynb](final_project.ipynb) |
| 2. Подготовка инфраструктуры. Разверните MLflow с хранилищем артефактов. | .sh-скрипт с запуском и настройкой MLflow. | [mlflow_server/run_mlflow_server.sh](mlflow_server/run_mlflow_server.sh) |
| 3. Трансляция. Выберите метрики, на которые вы хотите повлиять, и решите, как будете решать задачу. | Описано в README.md | [Метрики и подход](#метрики-и-подход-к-решению) |
| 4. Моделирование. Проведите эксперименты. Подготовьте пайплайн обработки данных и построения модели. | Jupyter Notebook с проведением экспериментов, bin-файл модели. | [final_project.ipynb](final_project.ipynb), [model.bin](model.bin) |
| 5. Продуктивизация. Оберните модель в веб-сервис, чтобы она отвечала на запросы по API. Также сервис должен подниматься в Docker для удобства выкатки. | Python-проект с описанным Dockerfile и описанной структурой API. | [app.py](app.py), [Dockerfile](Dockerfile) |
| 6. Мониторинг. Проследите, чтобы все сервисы в продакшен-среде контролировались метриками. | .md-файл с описанием метрик. Метрики должны отправляться из кода проекта. | [monitoring.md](monitoring.md) |
| 7. Документация. Самая важная часть — опишите процесс обработки данных, создания модели, её выкатки и сопровождения. | Заполненный README.md | Выполнено |
| 8. Требования и среда. Зафиксируйте случайные состояния и приложите зависимости, с которыми вы работали в рамках проекта. Важно соблюсти воспроизводимость экспериментов. | Сформированный файл requirements.txt | [requirements.txt](requirements.txt) |

## Стек технологий

| Категория | Инструменты |
|-----------|-------------|
| Анализ данных | Python, Pandas, NumPy, Matplotlib, Seaborn |
| Машинное обучение | Scikit-learn, CatBoost |
| Эксперименты | MLflow |
| Веб-сервис | FastAPI, Uvicorn |
| Контейнеризация | Docker |
| Мониторинг | Prometheus (интеграция через monitoring.md) |

---

## Метрики и подход к решению

### Тип задачи

**Multi-label классификация** – клиент может получить несколько новых продуктов в следующем месяце (24 целевых продукта).

### Основная метрика

**MAP@7 (Mean Average Precision at 7)** – доля релевантных продуктов среди топ-7 рекомендаций, усреднённая по всем клиентам. Именно эта метрика использовалась в оригинальном соревновании Santander Product Recommendation.

### Дополнительные метрики

- **Recall@7** – полнота рекомендаций.
- **PR-AUC** – площадь под Precision-Recall кривой (учитывает дисбаланс классов).

### Подход

- **Time-based split** – обучение на данных 2016-01 и 2016-02, валидация на 2016-03, тест на 2016-04.
- **Модель** – `OneVsRestClassifier` с `CatBoostClassifier` (50 итераций, `auto_class_weights='Balanced'`).
- **Целевая переменная** – новые продукты в следующем месяце (вычитание текущих продуктов из будущих).
- **Признаки** – возраст, стаж, доход, пол, сегмент, тип занятости, канал привлечения (текущие продукты не использовались для ускорения).

### Результаты

| Метрика | Validation | Test |
|---------|------------|------|
| MAP@7 | 0.3188 | 0.3143 |

---

## Воспроизводимость экспериментов

### Фиксированные случайные состояния

- `np.random.seed(42)`
- `random_seed=42` в CatBoostClassifier
- `random_state=42` в выборке клиентов (20%)

### Окружение

Все зависимости зафиксированы в `requirements.txt`. Для воспроизведения:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```


## Запуск проекта

### 1. Подготовка данных

```bash
# Скачайте данные с Яндекс.Диска
# Ссылка: https://disk.yandex.com/d/Io0siOESo2RAaA

# Распакуйте архив в папку data/
unzip train_ver2.csv.zip -d data/
```

### 2. Настройка окружения

```bash
# Создайте виртуальное окружение
python -m venv venv

# Активируйте его
source venv/bin/activate  # Linux/Mac
# или
venv\Scripts\activate     # Windows

# Установите зависимости
pip install -r requirements.txt
```

### 3. Запуск MLflow (для логирования экспериментов)

```bash
# Дайте права на выполнение скрипта
chmod +x mlflow_server/run_mlflow_server.sh

# Запустите MLflow сервер
./mlflow_server/run_mlflow_server.sh

# MLflow UI будет доступен по адресу:
# http://localhost:5000
```

### 4. Обучение модели

```bash
# Запустите Jupyter Notebook
jupyter notebook final_project.ipynb

# Выполните все ячейки последовательно
# Обучение занимает ~2-3 минуты
# После завершения модель сохранится как model.bin
```

**Результаты модели:**

- Validation MAP@7: 0.3188
- Test MAP@7: 0.3143

### 5. Запуск веб-сервиса локально

```bash
# Убедитесь, что model.bin находится в корне проекта
# Запустите FastAPI сервер
uvicorn app:app --reload --port 8000
```

Сервис будет доступен по адресу: http://localhost:8000

Документация API: http://localhost:8000/docs (Swagger UI)

### 6. Тестовый запрос к API

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 35,
    "antiguedad": 24,
    "renta": 45000,
    "sexo": "V",
    "segmento": "02 - PARTICULARES",
    "ind_empleado": "N",
    "canal_entrada": "KHE"
  }'
```

**Пример ответа:**

```json
{
  "recommendations": [
    "ind_cco_fin_ult1",
    "ind_recibo_fin_ult1",
    "ind_nomina_ult1"
  ],
  "scores": [0.982, 0.951, 0.923]
}
```

### 7. Запуск через Docker

```bash
# Сборка Docker-образа
docker build -t bank_recommender .

# Запуск контейнера
docker run -p 8000:8000 bank_recommender

# Проверка работоспособности
curl http://localhost:8000/
```

### 8. Проверка мониторинга

```bash
# Метрики Prometheus доступны по эндпоинту
curl http://localhost:8000/metrics
```

**Пример вывода:**

```text
# HELP api_requests_total Total HTTP requests
# TYPE api_requests_total counter
api_requests_total{method="POST",endpoint="/predict",status="200"} 42

# HELP prediction_latency_seconds Prediction latency in seconds
# TYPE prediction_latency_seconds histogram
prediction_latency_seconds_bucket{le="0.1"} 35
prediction_latency_seconds_bucket{le="0.5"} 42
```

---

## Устранение неполадок

### Проблема: модуль не найден при импорте

```bash
# Решение: установите недостающие зависимости
pip install -r requirements.txt
```

### Проблема: порт 8000 уже занят

```bash
# Решение: используйте другой порт
uvicorn app:app --port 8001
```

### Проблема: модель не загружается

```bash
# Проверьте, что model.bin существует в корне проекта
ls -lh model.bin

# Если файла нет, запустите обучение заново
jupyter notebook final_project.ipynb
```

### Проблема: не хватает памяти при обучении

```bash
# Уменьшите размер выборки в ноутбуке
# Измените параметр выборки клиентов с 0.2 на 0.1
selected_clients = np.random.choice(unique_clients, size=int(len(unique_clients)*0.1), replace=False)
```

---

## Структура проекта после запуска

```text
mle-pr-final/
├── data/
│   └── train_ver2.csv          # Исходные данные (2.3 GB)
├── mlflow_server/
│   └── run_mlflow_server.sh    # Скрипт запуска MLflow
├── final_project.ipynb         # Jupyter ноутбук (EDA + обучение)
├── model.bin                   # Обученная модель
├── app.py                      # FastAPI сервис
├── Dockerfile                  # Конфигурация Docker
├── requirements.txt            # Зависимости Python
├── monitoring.md               # Документация по мониторингу
├── README.md                   # Этот файл
├── .gitignore                  # Git исключения
└── venv/                       # Виртуальное окружение
```

---

## Переменные окружения (опционально)

Создайте файл `.env` на основе `.env_template`:

```bash
cp .env_template .env

# Отредактируйте .env при необходимости
```

```env
MLFLOW_TRACKING_URI=http://localhost:5000
MODEL_PATH=./model.bin
LOG_LEVEL=INFO
```

---

## CI/CD (опционально)

Для автоматизации деплоя можно использовать GitHub Actions:

```yaml
# .github/workflows/deploy.yml
name: Deploy Bank Recommender

on:
  push:
    branches: [main]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Build Docker image
        run: docker build -t bank_recommender .
      - name: Push to registry
        run: docker push your-registry/bank_recommender:latest
```
