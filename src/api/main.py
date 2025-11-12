"""
FastAPI REST API for Energy Demand Forecasting

Endpoints:
- POST /predict: Get energy demand predictions
- GET /models: List available models
- GET /health: Health check
- GET /metrics: Prometheus metrics
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, Any
from datetime import datetime, date
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import logging
from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    generate_latest,
    CONTENT_TYPE_LATEST,
)
from fastapi.responses import Response
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Energy Demand Forecasting API",
    description="REST API for energy consumption predictions and trading signals",
    version="1.0.0",
)

# Prometheus metrics
REQUEST_COUNT = Counter(
    "api_requests_total", "Total API requests", ["endpoint", "method"]
)
REQUEST_LATENCY = Histogram(
    "api_request_latency_seconds", "Request latency", ["endpoint"]
)
PREDICTION_COUNT = Counter("predictions_total", "Total predictions made", ["model"])
MODEL_LOAD_TIME = Gauge("model_load_time_seconds", "Time to load model", ["model"])
ACTIVE_MODELS = Gauge("active_models", "Number of loaded models")


# =============================================================================
# Pydantic Models
# =============================================================================


class WeatherInput(BaseModel):
    """Weather features for prediction."""

    temperature_2m_max: float = Field(
        ..., ge=-50, le=50, description="Max temperature in °C"
    )
    temperature_2m_min: float = Field(
        ..., ge=-50, le=50, description="Min temperature in °C"
    )
    precipitation_sum: float = Field(..., ge=0, description="Precipitation sum in mm")
    wind_speed_10m_max: float = Field(
        ..., ge=0, le=200, description="Max wind speed in km/h"
    )
    shortwave_radiation_sum: float = Field(
        ..., ge=0, description="Solar radiation in Wh/m²"
    )

    @field_validator("temperature_2m_min")
    @classmethod
    def validate_temp_min_max(cls, v, info):
        """Ensure min temp <= max temp."""
        if "temperature_2m_max" in info.data and v > info.data["temperature_2m_max"]:
            raise ValueError("temperature_2m_min must be <= temperature_2m_max")
        return v


class PredictionRequest(BaseModel):
    """Request for energy demand prediction."""

    date: date = Field(..., description="Date for prediction")
    weather: WeatherInput
    region: Optional[str] = Field("FR", description="Region code")
    model: Optional[str] = Field("xgboost", description="Model to use")
    include_confidence: bool = Field(False, description="Include prediction intervals")


class EnergyPrediction(BaseModel):
    """Energy demand prediction result."""

    date: date
    electricity_mw: float = Field(
        ..., description="Predicted electricity consumption in MW"
    )
    gas_mw: float = Field(..., description="Predicted gas consumption in MW")
    model: str
    confidence: Optional[Dict[str, Any]] = None


class PredictionResponse(BaseModel):
    """Response containing prediction."""

    prediction: EnergyPrediction
    timestamp: datetime
    request_id: Optional[str] = None


class ModelInfo(BaseModel):
    """Information about an available model."""

    name: str
    type: str
    version: Optional[str] = None
    metrics: Optional[Dict[str, float]] = None
    loaded: bool


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    timestamp: datetime
    models_loaded: int
    uptime_seconds: float


# =============================================================================
# Model Manager
# =============================================================================


class ModelManager:
    """Manages loading and caching of ML models."""

    def __init__(self, models_dir: Path = Path("models")):
        self.models_dir = models_dir
        self.loaded_models: Dict[str, Any] = {}
        self.model_info: Dict[str, Dict] = {}
        self.start_time = time.time()

        logger.info(f"Initialized ModelManager with models_dir: {models_dir}")

    def load_model(self, model_name: str) -> Any:
        """
        Load a model from disk.

        Args:
            model_name: Name of the model

        Returns:
            Loaded model object
        """
        if model_name in self.loaded_models:
            logger.debug(f"Model {model_name} already loaded")
            return self.loaded_models[model_name]

        # Find model file
        model_paths = {
            "xgboost": self.models_dir / "xgboost" / "xgboost_model_daily.pkl",
            "lightgbm": self.models_dir
            / "Quantile"
            / "lightgbm_quantile"
            / "lightgbm_quantile_model_daily.pkl",
            "ridge": self.models_dir / "reg_lin" / "ridge_model_daily.pkl",
            "lasso": self.models_dir / "reg_lin" / "lasso_model_daily.pkl",
        }

        if model_name not in model_paths:
            raise ValueError(f"Unknown model: {model_name}")

        model_path = model_paths[model_name]

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Load model
        start_time = time.time()

        with open(model_path, "rb") as f:
            model = pickle.load(f)

        load_time = time.time() - start_time

        # Cache model
        self.loaded_models[model_name] = model
        self.model_info[model_name] = {
            "type": model_name,
            "path": str(model_path),
            "loaded_at": datetime.now(),
            "load_time": load_time,
        }

        # Update metrics
        MODEL_LOAD_TIME.labels(model=model_name).set(load_time)
        ACTIVE_MODELS.set(len(self.loaded_models))

        logger.info(f"Loaded model {model_name} in {load_time:.3f}s")

        return model

    def get_model(self, model_name: str) -> Any:
        """Get a model (load if not cached)."""
        return self.load_model(model_name)

    def list_models(self) -> List[ModelInfo]:
        """List all available models."""
        models = []

        for name in ["xgboost", "lightgbm", "ridge", "lasso", "tft"]:
            models.append(
                ModelInfo(
                    name=name,
                    type=name,
                    loaded=name in self.loaded_models,
                    metrics=None,  # TODO: Load from metadata
                )
            )

        return models

    def unload_model(self, model_name: str):
        """Unload a model from memory."""
        if model_name in self.loaded_models:
            del self.loaded_models[model_name]
            ACTIVE_MODELS.set(len(self.loaded_models))
            logger.info(f"Unloaded model: {model_name}")


# Initialize model manager
model_manager = ModelManager()


# =============================================================================
# API Endpoints
# =============================================================================


@app.get("/", tags=["General"])
async def root():
    """Root endpoint."""
    return {
        "message": "Energy Demand Forecasting API",
        "version": "1.0.0",
        "endpoints": ["/predict", "/models", "/health", "/metrics"],
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Health check endpoint."""
    REQUEST_COUNT.labels(endpoint="/health", method="GET").inc()

    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(),
        models_loaded=len(model_manager.loaded_models),
        uptime_seconds=time.time() - model_manager.start_time,
    )


@app.get("/models", response_model=List[ModelInfo], tags=["Models"])
async def list_models():
    """List all available models."""
    REQUEST_COUNT.labels(endpoint="/models", method="GET").inc()

    return model_manager.list_models()


@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
async def predict(request: PredictionRequest):
    """
    Make energy demand prediction.

    Args:
        request: Prediction request with weather data

    Returns:
        Energy demand prediction
    """
    start_time = time.time()

    REQUEST_COUNT.labels(endpoint="/predict", method="POST").inc()

    try:
        # Load model
        model = model_manager.get_model(request.model)

        # Prepare features
        features = pd.DataFrame(
            [
                {
                    "temperature_2m_max": request.weather.temperature_2m_max,
                    "temperature_2m_min": request.weather.temperature_2m_min,
                    "precipitation_sum": request.weather.precipitation_sum,
                    "wind_speed_10m_max": request.weather.wind_speed_10m_max,
                    "shortwave_radiation_sum": request.weather.shortwave_radiation_sum,
                }
            ]
        )

        # Add derived features
        features["temperature_2m_mean"] = (
            features["temperature_2m_max"] + features["temperature_2m_min"]
        ) / 2
        features["temp_range"] = (
            features["temperature_2m_max"] - features["temperature_2m_min"]
        )

        # Make prediction
        prediction = model.predict(features.values)

        # Extract electricity and gas predictions
        elec_mw = float(prediction[0][0])
        gas_mw = float(prediction[0][1])

        # Update metrics
        PREDICTION_COUNT.labels(model=request.model).inc()

        # Prepare response
        response = PredictionResponse(
            prediction=EnergyPrediction(
                date=request.date,
                electricity_mw=elec_mw,
                gas_mw=gas_mw,
                model=request.model,
                confidence=(
                    None
                    if not request.include_confidence
                    else {
                        "electricity_lower": elec_mw * 0.9,
                        "electricity_upper": elec_mw * 1.1,
                        "gas_lower": gas_mw * 0.9,
                        "gas_upper": gas_mw * 1.1,
                    }
                ),
            ),
            timestamp=datetime.now(),
            request_id=None,
        )

        # Record latency
        latency = time.time() - start_time
        REQUEST_LATENCY.labels(endpoint="/predict").observe(latency)

        logger.info(
            f"Prediction made in {latency:.3f}s: elec={elec_mw:.2f} MW, gas={gas_mw:.2f} MW"
        )

        return response

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", tags=["Predictions"])
async def predict_batch(requests: List[PredictionRequest]):
    """
    Make batch predictions.

    Args:
        requests: List of prediction requests

    Returns:
        List of predictions
    """
    REQUEST_COUNT.labels(endpoint="/predict/batch", method="POST").inc()

    predictions = []

    for req in requests:
        try:
            pred = await predict(req)
            predictions.append(pred)
        except Exception as e:
            logger.error(f"Batch prediction error for {req.date}: {str(e)}")
            predictions.append({"error": str(e), "date": req.date})

    return predictions


@app.get("/metrics", tags=["Monitoring"])
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.on_event("startup")
async def startup_event():
    """Run on application startup."""
    logger.info("Starting Energy Demand Forecasting API...")

    # Pre-load default model
    try:
        model_manager.load_model("xgboost")
        logger.info("Pre-loaded default model: xgboost")
    except Exception as e:
        logger.warning(f"Could not pre-load default model: {e}")


@app.on_event("shutdown")
async def shutdown_event():
    """Run on application shutdown."""
    logger.info("Shutting down Energy Demand Forecasting API...")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
