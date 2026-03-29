"""
Module: utils.config_loader

Description:
    Configuration loader with environment variable support

Inputs:
    - config_path: Path to configuration file

Outputs:
    - Configuration dictionary

Dependencies:
    - yaml
    - os
    - src.utils.helpers

Usage:
    >>> from src.utils.config_loader import ConfigLoader
    >>> config = ConfigLoader.load()
    >>> model_name = config.get_model_name()
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from src.utils.exceptions import ConfigurationError
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ConfigLoader:
    """
    Load and manage configuration from YAML files and environment variables.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration loader.

        Args:
            config_path: Path to configuration file (default: config/config.yaml)
        """
        if config_path is None:
            config_path = "config/config.yaml"

        self.config_path = Path(config_path)
        self._config: Optional[Dict[str, Any]] = None

    def load(self) -> Dict[str, Any]:
        """
        Load configuration from file and override with environment variables.

        Returns:
            Configuration dictionary

        Raises:
            ConfigurationError: If configuration cannot be loaded
        """
        if self._config is not None:
            return self._config

        logger.info(f"Loading configuration from {self.config_path}")

        if not self.config_path.exists():
            raise ConfigurationError(
                f"Configuration file not found: {self.config_path}",
                details={"config_path": str(self.config_path)}
            )

        try:
            with open(self.config_path, 'r') as f:
                self._config = yaml.safe_load(f)

            # Override with environment variables
            self._apply_env_overrides()

            logger.info("Configuration loaded successfully")
            return self._config

        except Exception as e:
            raise ConfigurationError(
                f"Failed to load configuration: {str(e)}",
                details={"config_path": str(self.config_path), "error": str(e)}
            )

    def _apply_env_overrides(self) -> None:
        """
        Override configuration values with environment variables.

        Environment variables should be prefixed with RAG_ and use double
        underscores to separate nested keys.

        Example:
            RAG_MODELS__EMBEDDING_MODEL=all-MiniLM-L6-v2

        Returns:
            None
        """
        if self._config is None:
            return

        for key, value in os.environ.items():
            if key.startswith("RAG_"):
                # Remove RAG_ prefix and split by double underscore
                config_key = key[4:].lower()
                keys = config_key.split("__")

                # Navigate to the correct nested level
                current = self._config
                for k in keys[:-1]:
                    if k not in current:
                        current[k] = {}
                    current = current[k]

                # Set the value
                final_key = keys[-1]
                # Try to parse as YAML for correct type conversion
                try:
                    current[final_key] = yaml.safe_load(value)
                except yaml.YAMLError:
                    current[final_key] = value

                logger.debug(f"Override config: {config_key}={value}")

    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value by key path (dot-separated).

        Args:
            key_path: Dot-separated path to configuration value
            default: Default value if key not found

        Returns:
            Configuration value or default

        Example:
            >>> config = ConfigLoader()
            >>> model = config.get("models.embedding_model")
        """
        if self._config is None:
            self.load()

        keys = key_path.split(".")
        current = self._config

        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default

        return current

    def reload(self) -> Dict[str, Any]:
        """
        Reload configuration from file.

        Returns:
            Configuration dictionary
        """
        self._config = None
        return self.load()
