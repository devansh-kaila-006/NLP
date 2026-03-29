"""
Module: utils.helpers

Description:
    Helper functions and utilities for the RAG system

Inputs:
    - Various (depends on function)

Outputs:
    - Various utility functions

Dependencies:
    - yaml
    - os
    - pathlib.Path

Usage:
    >>> from src.utils.helpers import load_config, ensure_dir
    >>> config = load_config()
    >>> ensure_dir("data/output")
"""

import os
from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to configuration file

    Returns:
        Dictionary with configuration parameters

    Raises:
        ConfigurationError: If config file cannot be loaded

    Example:
        >>> config = load_config()
        >>> chunk_size = config['processing']['pdf']['chunk_size']
    """
    from src.utils.exceptions import ConfigurationError

    config_file = Path(config_path)

    if not config_file.exists():
        raise ConfigurationError(
            f"Configuration file not found: {config_path}",
            details={"config_path": config_path}
        )

    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        raise ConfigurationError(
            f"Failed to load configuration: {str(e)}",
            details={"config_path": config_path, "error": str(e)}
        )


def ensure_dir(directory: str) -> Path:
    """
    Ensure directory exists, create if it doesn't.

    Args:
        directory: Directory path to ensure

    Returns:
        Path object for the directory

    Example:
        >>> data_dir = ensure_dir("data/chunks")
        >>> # Directory is guaranteed to exist
    """
    dir_path = Path(directory)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def get_env_var(var_name: str, default: Optional[str] = None) -> str:
    """
    Get environment variable with optional default.

    Args:
        var_name: Name of environment variable
        default: Default value if variable not set

    Returns:
        Environment variable value or default

    Raises:
        ConfigurationError: If variable not set and no default provided

    Example:
        >>> api_key = get_env_var("GOOGLE_API_KEY")
        >>> timeout = get_env_var("TIMEOUT", "30")
    """
    from src.utils.exceptions import ConfigurationError

    value = os.getenv(var_name, default)

    if value is None:
        raise ConfigurationError(
            f"Required environment variable not set: {var_name}",
            details={"var_name": var_name}
        )

    return value


def chunk_list(items: list, chunk_size: int) -> list[list]:
    """
    Split a list into chunks of specified size.

    Args:
        items: List to chunk
        chunk_size: Size of each chunk

    Returns:
        List of chunks

    Example:
        >>> items = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        >>> chunks = chunk_list(items, 3)
        >>> # [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    """
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]


def merge_dicts(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge two dictionaries.

    Args:
        dict1: Base dictionary
        dict2: Dictionary to merge into base

    Returns:
        Merged dictionary

    Example:
        >>> base = {"a": 1, "b": {"x": 10}}
        >>> update = {"b": {"y": 20}, "c": 3}
        >>> merged = merge_dicts(base, update)
        >>> # {"a": 1, "b": {"x": 10, "y": 20}, "c": 3}
    """
    result = dict1.copy()

    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = value

    return result
