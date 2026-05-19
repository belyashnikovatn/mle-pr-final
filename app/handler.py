"""Хендлер для загрузки модели и предсказаний."""

import logging
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Tuple, List
from dataclasses import dataclass

from .config import config
from .schemas import ClientFeatures
from .metrics import MODEL_LOADED

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Результат валидации входных данных."""
    is_valid: bool
    message: str = "OK"


class FastApiHandler:
    """
    Класс для обработки запросов к модели рекомендации банковских продуктов.
    Загружает модель, валидирует входные данные и возвращает предсказания.
    """
    
    def __init__(self, model_path: str):
        """
        Инициализация хендлера.
        
        Args:
            model_path: Путь к файлу модели (model.bin)
        """
        self.model_path = Path(model_path)
        self.model = None
        self.products: List[str] = config.PRODUCTS
        self.cat_features: List[str] = config.CAT_FEATURES
        self._load_model()
    
    def _load_model(self) -> None:
        """Загружает модель из файла."""
        try:
            if not self.model_path.exists():
                raise FileNotFoundError(f"Model not found at {self.model_path}")
            
            self.model = joblib.load(self.model_path)
            MODEL_LOADED.set(1)
            logger.info(f"Model loaded successfully from {self.model_path}")
            
        except Exception as e:
            MODEL_LOADED.set(0)
            logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Failed to load model: {e}")
    
    def _preprocess_features(self, features: ClientFeatures) -> pd.DataFrame:
        """
        Преобразует входные данные в DataFrame для модели.
        
        Args:
            features: Валидированные входные данные клиента
            
        Returns:
            DataFrame с признаками в нужном порядке
        """
        data = {
            'age': [features.age],
            'antiguedad': [features.antiguedad],
            'renta': [features.renta],
            'sexo': [features.sexo],
            'segmento': [features.segmento],
            'ind_empleado': [features.ind_empleado],
            'canal_entrada': [features.canal_entrada]
        }
        
        df = pd.DataFrame(data)
        
        for col in self.cat_features:
            df[col] = df[col].astype(str)
        
        return df
    
    def _extract_probabilities(self, predict_proba_result, n_samples: int = 1) -> np.ndarray:
        """
        Приводит выход OneVsRestClassifier.predict_proba к вектору (n_products,).
        CatBoost + OneVsRest может вернуть список (n_products,) массивов или матрицу (n_samples, n_products).
        """
        n_products = len(self.products)

        if isinstance(predict_proba_result, list):
            cols = []
            for p in predict_proba_result:
                p = np.asarray(p)
                if p.ndim == 2 and p.shape[1] == 2:
                    cols.append(p[:, 1])
                else:
                    cols.append(p.ravel())
            matrix = np.column_stack(cols)
        else:
            arr = np.asarray(predict_proba_result)
            if arr.ndim == 3:
                matrix = arr[:, :, 1]
            elif arr.ndim == 2:
                if arr.shape[0] == n_samples and arr.shape[1] == n_products:
                    matrix = arr
                elif arr.shape[0] == n_products and arr.shape[1] == n_samples:
                    matrix = arr.T
                else:
                    raise ValueError(f"Unexpected predict_proba shape: {arr.shape}")
            else:
                raise ValueError(f"Unexpected predict_proba shape: {arr.shape}")

        return matrix[0] if n_samples == 1 else matrix
    
    def validate_params(self, features: ClientFeatures) -> ValidationResult:
        """Валидация входных параметров."""
        if features.age < 18 or features.age > 120:
            return ValidationResult(False, "Age must be between 18 and 120")
        
        if features.antiguedad < 0 or features.antiguedad > 1200:
            return ValidationResult(False, "Antiguedad must be between 0 and 1200 months")
        
        allowed_sexo = {'V', 'H', 'U', ''}
        if features.sexo not in allowed_sexo:
            return ValidationResult(False, f"Sexo must be one of {allowed_sexo}")
        
        return ValidationResult(True, "OK")
    
    def predict(self, features: ClientFeatures) -> Tuple[List[str], List[float]]:
        """Возвращает топ-7 рекомендаций продуктов."""
        validation = self.validate_params(features)
        if not validation.is_valid:
            raise ValueError(validation.message)
        
        if self.model is None:
            raise RuntimeError("Model is not loaded")
        
        X = self._preprocess_features(features)
        proba_result = self.model.predict_proba(X)
        probs = self._extract_probabilities(proba_result, n_samples=len(X))
        
        top_indices = np.argsort(probs)[::-1][:7]
        recommendations = [self.products[i] for i in top_indices]
        scores = probs[top_indices].tolist()
        
        return recommendations, scores
    
    def is_ready(self) -> bool:
        """Проверяет, готова ли модель к работе."""
        return self.model is not None
    
    def health_check(self) -> dict:
        """Возвращает статус сервиса."""
        return {
            "status": "healthy" if self.is_ready() else "unhealthy",
            "model_loaded": self.is_ready(),
            "model_path": str(self.model_path)
        }