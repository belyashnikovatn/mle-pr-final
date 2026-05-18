"""FastAPI веб-сервис для рекомендации банковских продуктов."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from .config import config
from .handler import FastApiHandler
from .schemas import ClientFeatures, PredictionResponse, ErrorResponse
from .metrics import track_metrics, get_metrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

handler: FastApiHandler = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Управление жизненным циклом приложения."""
    global handler
    
    logger.info("Loading model...")
    handler = FastApiHandler(str(config.MODEL_PATH))
    logger.info("Model loaded successfully")
    
    yield
    
    logger.info("Shutting down...")


app = FastAPI(
    title="Bank Product Recommender API",
    description="Рекомендация банковских продуктов на основе характеристик клиента",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", tags=["Health"])
async def health_check():
    return handler.health_check()


@app.get("/metrics", tags=["Monitoring"])
async def metrics():
    return get_metrics()


@app.post(
    "/predict",
    response_model=PredictionResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Ошибка валидации"},
        500: {"model": ErrorResponse, "description": "Внутренняя ошибка"}
    },
    tags=["Prediction"]
)
@track_metrics(endpoint="/predict")
async def predict(features: ClientFeatures):
    try:
        recommendations, scores = handler.predict(features)
        return PredictionResponse(recommendations=recommendations, scores=scores)
    
    except ValueError as e:
        logger.warning(f"Validation error: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    
    except RuntimeError as e:
        logger.error(f"Runtime error: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
    
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=config.API_HOST,
        port=config.API_PORT,
        reload=True
    )