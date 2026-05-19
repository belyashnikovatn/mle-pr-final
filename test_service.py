"""
Тесты для имитации работы сервиса рекомендации банковских продуктов.
Проверяет эндпоинты /health, /predict, /metrics.
"""

import requests
import time
import json
from typing import List, Dict, Any

BASE_URL = "http://localhost:8000"

# Примеры клиентов для тестирования (реальные данные из датасета)
TEST_CLIENTS = [
    {
        "name": "Молодой клиент с низким доходом",
        "data": {
            "age": 25.0,
            "antiguedad": 6.0,
            "renta": 25000.0,
            "sexo": "V",
            "segmento": "03 - UNIVERSITARIO",
            "ind_empleado": "N",
            "canal_entrada": "KHE"
        }
    },
    {
        "name": "Клиент среднего возраста, высокий доход",
        "data": {
            "age": 45.0,
            "antiguedad": 120.0,
            "renta": 120000.0,
            "sexo": "H",
            "segmento": "02 - PARTICULARES",
            "ind_empleado": "A",
            "canal_entrada": "KAT"
        }
    },
    {
        "name": "Пожилой клиент, пенсионер",
        "data": {
            "age": 68.0,
            "antiguedad": 360.0,
            "renta": 35000.0,
            "sexo": "V",
            "segmento": "01 - TOP",
            "ind_empleado": "N",
            "canal_entrada": "OFF"
        }
    },
    {
        "name": "Клиент с пропущенным доходом",
        "data": {
            "age": 35.0,
            "antiguedad": 24.0,
            "renta": None,
            "sexo": "U",
            "segmento": "02 - PARTICULARES",
            "ind_empleado": "F",
            "canal_entrada": "KHE"
        }
    }
]

# Негативные тестовые кейсы (должны возвращать ошибку 400)
NEGATIVE_TEST_CASES = [
    {
        "name": "Отрицательный возраст",
        "data": {
            "age": -5.0,
            "antiguedad": 24.0,
            "renta": 45000.0,
            "sexo": "V",
            "segmento": "02 - PARTICULARES",
            "ind_empleado": "N",
            "canal_entrada": "KHE"
        }
    },
    {
        "name": "Слишком большой возраст",
        "data": {
            "age": 150.0,
            "antiguedad": 24.0,
            "renta": 45000.0,
            "sexo": "V",
            "segmento": "02 - PARTICULARES",
            "ind_empleado": "N",
            "canal_entrada": "KHE"
        }
    },
    {
        "name": "Недопустимое значение пола",
        "data": {
            "age": 35.0,
            "antiguedad": 24.0,
            "renta": 45000.0,
            "sexo": "X",
            "segmento": "02 - PARTICULARES",
            "ind_empleado": "N",
            "canal_entrada": "KHE"
        }
    },
    {
        "name": "Отрицательный стаж",
        "data": {
            "age": 35.0,
            "antiguedad": -10.0,
            "renta": 45000.0,
            "sexo": "V",
            "segmento": "02 - PARTICULARES",
            "ind_empleado": "N",
            "canal_entrada": "KHE"
        }
    },
    {
        "name": "Отсутствует обязательное поле",
        "data": {
            "age": 35.0,
            "antiguedad": 24.0,
            "renta": 45000.0,
            "sexo": "V",
            "ind_empleado": "N",
            "canal_entrada": "KHE"
        }
    }
]


def print_separator(char: str = "=", length: int = 70) -> None:
    """Печатает разделительную линию."""
    print(char * length)


def print_test_result(test_name: str, passed: bool, message: str = "") -> None:
    """Выводит результат теста с цветом."""
    status = "✅ PASSED" if passed else "❌ FAILED"
    print(f"[{status}] {test_name}")
    if message:
        print(f"    {message}")


def test_health() -> bool:
    """Тест 1: Проверка healthcheck эндпоинта."""
    print("\n--- Test 1: Health Check ---")
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        passed = response.status_code == 200
        
        if passed:
            data = response.json()
            print(f"    Response: {json.dumps(data, indent=4)}")
            passed = data.get("status") == "healthy" and data.get("model_loaded") == True
        else:
            print(f"    Status: {response.status_code}")
        
        print_test_result("Health check", passed)
        return passed
    except Exception as e:
        print_test_result("Health check", False, str(e))
        return False


def test_metrics() -> bool:
    """Тест 2: Проверка эндпоинта метрик Prometheus."""
    print("\n--- Test 2: Metrics Endpoint ---")
    try:
        response = requests.get(f"{BASE_URL}/metrics", timeout=5)
        passed = response.status_code == 200
        
        if passed:
            # Проверяем, что в ответе есть Prometheus метрики
            content = response.text
            has_metrics = any(metric in content for metric in [
                'api_requests_total', 
                'prediction_latency_seconds',
                'model_loaded',
                'active_requests'
            ])
            passed = passed and has_metrics
            print(f"    Metrics found: {has_metrics}")
            print(f"    Response size: {len(content)} bytes")
        
        print_test_result("Metrics endpoint", passed)
        return passed
    except Exception as e:
        print_test_result("Metrics endpoint", False, str(e))
        return False


def test_predict_positive(client_data: Dict[str, Any]) -> tuple[bool, Dict]:
    """Тест позитивного сценария предсказания."""
    response = requests.post(
        f"{BASE_URL}/predict",
        json=client_data,
        headers={"Content-Type": "application/json"},
        timeout=10
    )
    
    if response.status_code != 200:
        return False, {"error": f"Status {response.status_code}"}
    
    data = response.json()
    # Проверяем структуру ответа
    required_fields = ["recommendations", "scores"]
    if not all(field in data for field in required_fields):
        return False, {"error": f"Missing fields: {required_fields}"}
    
    # Проверяем, что recommendations - список из 7 элементов
    if len(data["recommendations"]) != 7:
        return False, {"error": f"Expected 7 recommendations, got {len(data['recommendations'])}"}
    
    # Проверяем, что scores - список из 7 чисел от 0 до 1
    if len(data["scores"]) != 7:
        return False, {"error": f"Expected 7 scores, got {len(data['scores'])}"}
    
    if not all(0 <= s <= 1 for s in data["scores"]):
        return False, {"error": "Scores must be between 0 and 1"}
    
    return True, data


def test_predict_negative(test_case: Dict[str, Any]) -> bool:
    """Тест негативного сценария предсказания (должен вернуть 400)."""
    response = requests.post(
        f"{BASE_URL}/predict",
        json=test_case["data"],
        headers={"Content-Type": "application/json"},
        timeout=10
    )
    
    passed = response.status_code == 400
    if not passed:
        print(f"    Expected 400, got {response.status_code}")
        if response.status_code == 200:
            print(f"    Response: {response.json()}")
    
    return passed


def test_predict_all_clients() -> tuple[int, int]:
    """Тест 3: Предсказания для разных типов клиентов."""
    print("\n--- Test 3: Predict for Different Client Types ---")
    
    passed_count = 0
    total_count = 0
    
    for client in TEST_CLIENTS:
        total_count += 1
        print(f"\n    Client: {client['name']}")
        print(f"    Input: {json.dumps(client['data'], indent=6)}")
        
        passed, result = test_predict_positive(client["data"])
        
        if passed:
            passed_count += 1
            print(f"    Recommendations: {result['recommendations']}")
            print(f"    Top score: {result['scores'][0]:.4f}")
        else:
            print(f"    Error: {result.get('error', 'Unknown error')}")
        
        print_test_result(f"  Predict for {client['name']}", passed)
        time.sleep(0.1)  # Небольшая задержка между запросами
    
    return passed_count, total_count


def test_negative_cases() -> tuple[int, int]:
    """Тест 4: Негативные сценарии (должны возвращать ошибку 400)."""
    print("\n--- Test 4: Negative Test Cases ---")
    
    passed_count = 0
    total_count = 0
    
    for test_case in NEGATIVE_TEST_CASES:
        total_count += 1
        print(f"\n    Test: {test_case['name']}")
        print(f"    Input: {json.dumps(test_case['data'], indent=6)}")
        
        passed = test_predict_negative(test_case)
        if passed:
            passed_count += 1
            print(f"    ✅ Correctly returned 400")
        else:
            print(f"    ❌ Failed - expected 400 error")
        
        print_test_result(f"  Negative: {test_case['name']}", passed)
        time.sleep(0.1)
    
    return passed_count, total_count


def test_concurrent_requests(num_requests: int = 10) -> bool:
    """Тест 5: Несколько последовательных запросов (имитация нагрузки)."""
    print(f"\n--- Test 5: Concurrent Requests Simulation ({num_requests} requests) ---")
    
    test_client = TEST_CLIENTS[0]["data"]
    success_count = 0
    response_times = []
    
    for i in range(num_requests):
        start_time = time.time()
        response = requests.post(
            f"{BASE_URL}/predict",
            json=test_client,
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        elapsed = time.time() - start_time
        response_times.append(elapsed)
        
        if response.status_code == 200:
            success_count += 1
        
        if (i + 1) % 5 == 0:
            print(f"    Processed {i + 1}/{num_requests} requests")
    
    avg_time = sum(response_times) / len(response_times)
    max_time = max(response_times)
    min_time = min(response_times)
    
    print(f"\n    Results:")
    print(f"    - Success rate: {success_count}/{num_requests} ({success_count/num_requests*100:.1f}%)")
    print(f"    - Avg latency: {avg_time*1000:.2f} ms")
    print(f"    - Min latency: {min_time*1000:.2f} ms")
    print(f"    - Max latency: {max_time*1000:.2f} ms")
    
    passed = success_count == num_requests
    print_test_result("Load test", passed, f"{success_count}/{num_requests} successful")
    
    return passed


def run_all_tests() -> Dict[str, Any]:
    """Запускает все тесты и возвращает сводку."""
    print_separator("=")
    print("BANK PRODUCT RECOMMENDER SERVICE TESTS")
    print_separator("=")
    print(f"Base URL: {BASE_URL}")
    
    results = {
        "health": False,
        "metrics": False,
        "positive_tests": (0, 0),
        "negative_tests": (0, 0),
        "load_test": False
    }
    
    # Проверяем доступность сервиса
    try:
        requests.get(BASE_URL, timeout=3)
    except requests.ConnectionError:
        print("\n❌ ERROR: Service is not available!")
        print(f"   Make sure the service is running on {BASE_URL}")
        print("   Run: uvicorn app.main:app --reload --port 8000")
        return results
    
    # Запускаем тесты
    results["health"] = test_health()
    results["metrics"] = test_metrics()
    results["positive_tests"] = test_predict_all_clients()
    results["negative_tests"] = test_negative_cases()
    results["load_test"] = test_concurrent_requests(10)
    
    # Итоговая сводка
    print_separator("=")
    print("TEST SUMMARY")
    print_separator("=")
    
    print(f"Health check:          {'✅' if results['health'] else '❌'}")
    print(f"Metrics endpoint:      {'✅' if results['metrics'] else '❌'}")
    
    pos_passed, pos_total = results["positive_tests"]
    print(f"Positive tests:        {pos_passed}/{pos_total} passed")
    
    neg_passed, neg_total = results["negative_tests"]
    print(f"Negative tests:        {neg_passed}/{neg_total} passed")
    
    print(f"Load test:             {'✅' if results['load_test'] else '❌'}")
    
    total_passed = all([
        results["health"],
        results["metrics"],
        pos_passed == pos_total,
        neg_passed == neg_total,
        results["load_test"]
    ])
    
    print_separator("=")
    print(f"OVERALL: {'✅ ALL TESTS PASSED' if total_passed else '❌ SOME TESTS FAILED'}")
    print_separator("=")
    
    return results


def monitor_metrics(duration_seconds: int = 5) -> None:
    """Дополнительная функция: мониторинг метрик в реальном времени."""
    print(f"\n--- Monitoring Metrics for {duration_seconds} seconds ---")
    
    response = requests.get(f"{BASE_URL}/metrics")
    if response.status_code == 200:
        # Выводим ключевые метрики
        lines = response.text.split('\n')
        for line in lines:
            if line.startswith(('api_requests_total', 'prediction_latency_seconds_count', 'model_loaded')):
                print(f"  {line}")
    
    time.sleep(duration_seconds)
    
    response = requests.get(f"{BASE_URL}/metrics")
    if response.status_code == 200:
        print("\n  Metrics after test series:")
        lines = response.text.split('\n')
        for line in lines:
            if line.startswith(('api_requests_total', 'prediction_latency_seconds_count')):
                print(f"  {line}")


if __name__ == "__main__":
    results = run_all_tests()
    
    # Если нужно посмотреть метрики после тестов
    if results["health"]:
        monitor_metrics(2)