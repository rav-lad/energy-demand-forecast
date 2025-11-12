"""
Configuration Loader Utility

Loads configuration from config.yaml and provides easy access to settings.
"""

import yaml
from pathlib import Path
from typing import Any, Dict
import logging

logger = logging.getLogger(__name__)


class ConfigLoader:
    """Load and manage project configuration from YAML file."""

    def __init__(self, config_path: str = None):
        """
        Initialize configuration loader.

        Args:
            config_path: Path to config.yaml file. If None, searches in project root.
        """
        if config_path is None:
            # Search for config.yaml in project root
            project_root = Path(__file__).parent.parent.parent
            config_path = project_root / "config.yaml"

        self.config_path = Path(config_path)
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {self.config_path}\n"
                "Please ensure config.yaml exists in the project root."
            )

        try:
            with open(self.config_path, "r") as f:
                config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {self.config_path}")
            return config
        except Exception as e:
            raise ValueError(f"Error loading configuration: {str(e)}")

    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.

        Args:
            key_path: Dot-separated path to config value (e.g., 'paths.data_root')
            default: Default value if key not found

        Returns:
            Configuration value or default

        Examples:
            >>> config = ConfigLoader()
            >>> config.get('paths.data_root')
            'data'
            >>> config.get('trading.general.initial_capital')
            100000
        """
        keys = key_path.split(".")
        value = self.config

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default

        return value

    def get_paths(self) -> Dict[str, str]:
        """Get all path configurations."""
        return self.config.get("paths", {})

    def get_trading_config(self) -> Dict[str, Any]:
        """Get trading configuration."""
        return self.config.get("trading", {})

    def get_backtesting_config(self) -> Dict[str, Any]:
        """Get backtesting configuration."""
        return self.config.get("backtesting", {})

    def get_model_config(self, model_type: str) -> Dict[str, Any]:
        """
        Get configuration for a specific model.

        Args:
            model_type: Model type (e.g., 'xgboost', 'tft', 'lightgbm')

        Returns:
            Model configuration dictionary
        """
        demand_models = self.config.get("models", {}).get("demand_forecast", {})
        return demand_models.get(model_type, {})

    def get_data_collection_config(self, data_type: str) -> Dict[str, Any]:
        """
        Get data collection configuration.

        Args:
            data_type: Data type (e.g., 'energy', 'weather', 'market')

        Returns:
            Data collection configuration
        """
        data_config = self.config.get("data_collection", {})
        return data_config.get(data_type, {})

    def __repr__(self) -> str:
        return f"ConfigLoader(config_path='{self.config_path}')"


# Global configuration instance
_global_config = None


def get_config(config_path: str = None) -> ConfigLoader:
    """
    Get global configuration instance (singleton pattern).

    Args:
        config_path: Path to config file (only used on first call)

    Returns:
        ConfigLoader instance
    """
    global _global_config

    if _global_config is None:
        _global_config = ConfigLoader(config_path)

    return _global_config


def reload_config(config_path: str = None):
    """
    Reload configuration from file.

    Args:
        config_path: Path to config file
    """
    global _global_config
    _global_config = ConfigLoader(config_path)


if __name__ == "__main__":
    # Test configuration loader
    config = get_config()

    print("Configuration loaded successfully!")
    print(f"\nProject: {config.get('project.name')}")
    print(f"Version: {config.get('project.version')}")
    print(f"Data root: {config.get('paths.data_root')}")
    print(f"Initial capital: {config.get('trading.general.initial_capital')}")

    print("\nTrading strategies:")
    trading = config.get_trading_config()
    for strategy, settings in trading.items():
        if isinstance(settings, dict) and settings.get("enabled"):
            print(f"  - {strategy}")
