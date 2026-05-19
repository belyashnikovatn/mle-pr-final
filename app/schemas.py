"""Pydantic схемы для валидации запросов и ответов."""

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ClientFeatures(BaseModel):
    """Входные данные клиента."""

    age: float = Field(..., ge=0, le=120, description="Возраст клиента")
    antiguedad: float = Field(..., ge=0, description="Стаж клиента в месяцах")
    renta: Optional[float] = Field(None, ge=0, description="Доход домохозяйства")
    sexo: str = Field(..., pattern="^(V|H|U|)$", description="Пол: V/H/U")
    segmento: str = Field(..., description="Сегмент клиента")
    ind_empleado: str = Field(..., pattern="^(A|B|F|N|S|)$", description="Статус занятости")
    canal_entrada: str = Field(..., description="Канал привлечения")

    @field_validator("renta", mode="before")
    @classmethod
    def validate_renta(cls, v):
        """Заполняем пропуски дохода значением по умолчанию."""
        return v if v is not None else 30000.0

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "age": 35.0,
                "antiguedad": 24.0,
                "renta": 45000.0,
                "sexo": "V",
                "segmento": "02 - PARTICULARES",
                "ind_empleado": "N",
                "canal_entrada": "KHE",
            }
        }
    )


class PredictionResponse(BaseModel):
    """Ответ сервера с рекомендациями."""

    recommendations: List[str] = Field(..., max_length=7, description="Топ-7 рекомендуемых продуктов")
    scores: List[float] = Field(..., description="Вероятности для рекомендованных продуктов")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "recommendations": ["ind_cco_fin_ult1", "ind_recibo_fin_ult1", "ind_nomina_ult1"],
                "scores": [0.98, 0.95, 0.92],
            }
        }
    )


class ErrorResponse(BaseModel):
    """Ответ с ошибкой."""

    error: str = Field(..., description="Сообщение об ошибке")
