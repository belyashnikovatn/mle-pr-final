"""Конфигурация приложения."""

from pathlib import Path
from dataclasses import dataclass


@dataclass
class Config:
    """Настройки приложения."""
    
    # Пути (относительно корня проекта)
    PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
    MODEL_PATH: Path = PROJECT_ROOT / "app" / "model.bin"
    
    # Категориальные признаки
    CAT_FEATURES: list = None
    
    # Список продуктов (24)
    PRODUCTS: list = None
    
    # Параметры модели
    MODEL_ITERATIONS: int = 50
    MODEL_SEED: int = 42
    
    # API настройки
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    
    # Мониторинг
    METRICS_PORT: int = 9090
    
    def __post_init__(self):
        if self.CAT_FEATURES is None:
            self.CAT_FEATURES = ['sexo', 'segmento', 'ind_empleado', 'canal_entrada']
        
        if self.PRODUCTS is None:
            self.PRODUCTS = [
                'ind_ahor_fin_ult1', 'ind_aval_fin_ult1', 'ind_cco_fin_ult1', 'ind_cder_fin_ult1',
                'ind_cno_fin_ult1', 'ind_ctju_fin_ult1', 'ind_ctma_fin_ult1', 'ind_ctop_fin_ult1',
                'ind_ctpp_fin_ult1', 'ind_deco_fin_ult1', 'ind_deme_fin_ult1', 'ind_dela_fin_ult1',
                'ind_ecue_fin_ult1', 'ind_fond_fin_ult1', 'ind_hip_fin_ult1', 'ind_plan_fin_ult1',
                'ind_pres_fin_ult1', 'ind_reca_fin_ult1', 'ind_tjcr_fin_ult1', 'ind_valo_fin_ult1',
                'ind_viv_fin_ult1', 'ind_nomina_ult1', 'ind_nom_pens_ult1', 'ind_recibo_ult1'
            ]


# Глобальный экземпляр конфигурации
config = Config()