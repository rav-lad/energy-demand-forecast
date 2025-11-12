"""
Centralized Configuration Management with Pydantic

All configuration is managed here with validation and type checking.
Environment variables are loaded from .env file.

Usage:
    from src.config.settings import settings

    api_key = settings.data.entsoe_api_key
    capital = settings.trading.initial_capital
"""

from pydantic_settings import BaseSettings
from pydantic import Field, field_validator
from pathlib import Path
from typing import List, Optional
import os


class PathsConfig(BaseSettings):
    """Path configuration."""

    # Project root
    project_root: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent.parent,
        description="Project root directory",
    )

    # Data directories
    data_root: Path = Field(default=None)
    raw_data: Path = Field(default=None)
    modified_data: Path = Field(default=None)
    transformed_data: Path = Field(default=None)

    # Model directories
    models_root: Path = Field(default=None)
    demand_models: Path = Field(default=None)
    price_models: Path = Field(default=None)

    # Output directories
    outputs_root: Path = Field(default=None)
    figures: Path = Field(default=None)
    reports: Path = Field(default=None)
    logs: Path = Field(default=None)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Initialize derived paths
        self.data_root = self.project_root / "data"
        self.raw_data = self.data_root / "raw_data"
        self.modified_data = self.data_root / "modified_data"
        self.transformed_data = self.data_root / "transformed_data"

        self.models_root = self.project_root / "models"
        self.demand_models = self.models_root / "demand_forecast"
        self.price_models = self.models_root / "price_forecast"

        self.outputs_root = self.project_root / "outputs"
        self.figures = self.outputs_root / "figures"
        self.reports = self.outputs_root / "reports"
        self.logs = self.outputs_root / "logs"

    class Config:
        env_prefix = "PATH_"


class DataConfig(BaseSettings):
    """Data collection configuration."""

    # API Keys
    entsoe_api_key: str = Field(..., description="ENTSO-E API key")

    # Date ranges
    start_date: str = Field(default="2020-01-01", description="Data start date")
    end_date: str = Field(default="2024-12-31", description="Data end date")

    # Regions
    countries: List[str] = Field(
        default=["FR", "DE", "ES"], description="Country codes"
    )

    # API endpoints
    entsoe_api_url: str = "https://web-api.tp.entsoe.eu/api"
    odre_api_url: str = "https://odre.opendatasoft.com/api/records/1.0/search/"

    @field_validator("entsoe_api_key")
    @classmethod
    def validate_api_key(cls, v):
        if not v or v == "your_entsoe_api_key_here":
            raise ValueError(
                "ENTSO-E API key not configured. "
                "Get one from https://transparency.entsoe.eu/"
            )
        return v

    class Config:
        env_prefix = "DATA_"


class ModelConfig(BaseSettings):
    """Model training configuration."""

    # General
    frequency: str = Field(default="daily", description="Data frequency")
    test_size: float = Field(default=0.2, ge=0.0, le=1.0)
    random_state: int = Field(default=42)

    # XGBoost
    xgb_n_estimators: int = Field(default=1000, ge=10)
    xgb_learning_rate: float = Field(default=0.01, ge=0.001, le=1.0)
    xgb_max_depth: int = Field(default=7, ge=1, le=20)

    # LightGBM
    lgb_n_estimators: int = Field(default=200, ge=10)
    lgb_learning_rate: float = Field(default=0.01, ge=0.001, le=1.0)
    lgb_quantiles: List[float] = Field(default=[0.05, 0.5, 0.95])

    # TFT
    tft_max_epochs: int = Field(default=30, ge=1, le=1000)
    tft_batch_size: int = Field(default=128, ge=8)
    tft_learning_rate: float = Field(default=1e-3, ge=1e-5, le=1.0)
    tft_hidden_size: int = Field(default=32, ge=8)

    # Linear models
    ridge_alpha: float = Field(default=10.0, ge=0.0)
    lasso_alpha: float = Field(default=1.0, ge=0.0)

    class Config:
        env_prefix = "MODEL_"


class TradingConfig(BaseSettings):
    """Trading configuration."""

    # Capital
    initial_capital: float = Field(default=100000, ge=1000)

    # Costs
    transaction_cost: float = Field(default=0.001, ge=0.0, le=0.1)
    slippage: float = Field(default=0.0005, ge=0.0, le=0.1)

    # Limits
    max_position_size: float = Field(default=10000, ge=100)
    max_positions: int = Field(default=5, ge=1)

    # Risk
    max_daily_loss: float = Field(default=5000, ge=0)
    max_drawdown: float = Field(default=0.15, ge=0.0, le=1.0)

    # Strategy: Demand-Price Arbitrage
    buy_threshold: float = Field(default=0.95, ge=0.0, le=1.0)
    sell_threshold: float = Field(default=0.25, ge=0.0, le=1.0)
    renewable_high: float = Field(default=0.7, ge=0.0, le=1.0)
    renewable_low: float = Field(default=0.3, ge=0.0, le=1.0)

    class Config:
        env_prefix = "TRADING_"


class MLflowConfig(BaseSettings):
    """MLflow configuration."""

    tracking_uri: str = Field(default="./mlruns", description="MLflow tracking URI")
    experiment_name: str = Field(default="energy_demand_forecast")
    artifact_location: Optional[str] = None

    class Config:
        env_prefix = "MLFLOW_"


class APIConfig(BaseSettings):
    """API configuration."""

    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000, ge=1024, le=65535)
    reload: bool = Field(default=False)
    workers: int = Field(default=1, ge=1)

    # CORS
    cors_origins: List[str] = Field(default=["*"])

    class Config:
        env_prefix = "API_"


class MonitoringConfig(BaseSettings):
    """Monitoring configuration."""

    # Prometheus
    prometheus_enabled: bool = Field(default=True)
    prometheus_port: int = Field(default=8001, ge=1024, le=65535)

    # Grafana
    grafana_enabled: bool = Field(default=True)
    grafana_port: int = Field(default=3000, ge=1024, le=65535)

    # Alerts
    alert_email: Optional[str] = None
    alert_on_loss: bool = Field(default=True)
    alert_threshold: float = Field(default=1000, ge=0)

    class Config:
        env_prefix = "MONITOR_"


class Settings(BaseSettings):
    """Global settings."""

    # Sub-configurations
    paths: PathsConfig = Field(default_factory=PathsConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    trading: TradingConfig = Field(default_factory=TradingConfig)
    mlflow: MLflowConfig = Field(default_factory=MLflowConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)

    # General
    debug: bool = Field(default=False)
    log_level: str = Field(default="INFO")
    environment: str = Field(default="development")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global singleton instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get global settings instance (singleton pattern)."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


# Export settings instance
settings = get_settings()


# Helper functions
def create_directories():
    """Create all necessary directories."""
    dirs = [
        settings.paths.data_root,
        settings.paths.raw_data,
        settings.paths.modified_data,
        settings.paths.transformed_data,
        settings.paths.models_root,
        settings.paths.demand_models,
        settings.paths.price_models,
        settings.paths.outputs_root,
        settings.paths.figures,
        settings.paths.reports,
        settings.paths.logs,
    ]

    for directory in dirs:
        directory.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    # Test configuration
    print("Configuration loaded successfully!")
    print(f"\nProject: {settings.paths.project_root}")
    print(f"Data root: {settings.paths.data_root}")
    print(f"Models root: {settings.paths.models_root}")
    print(f"\nTrading capital: {settings.trading.initial_capital:,.2f} EUR")
    print(f"API port: {settings.api.port}")
    print(f"\nEnvironment: {settings.environment}")
    print(f"Debug: {settings.debug}")
