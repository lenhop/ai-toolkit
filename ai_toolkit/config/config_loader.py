"""
Simple Configuration Loader

Load configuration from YAML files and environment variables.
Keep it simple - just load and access config values.

Usage:
    >>> from ai_toolkit.config import load_config, load_env
    >>> 
    >>> # Load environment variables
    >>> load_env('.env')
    >>> 
    >>> # Load YAML config
    >>> config = load_config('config.yaml')
    >>> print(config['models']['deepseek']['model'])
    >>> 
    >>> # Access nested values
    >>> model_name = config.get('models', {}).get('deepseek', {}).get('model')

Official Python config best practices:
    - Use .env files for secrets (with python-dotenv)
    - Use YAML/JSON for structured config
    - Keep it simple - don't over-engineer
"""

import os
import yaml
import json
from pathlib import Path
from typing import Any, Dict, Optional, Union
from dotenv import load_dotenv


def load_env(env_file: Union[str, Path] = '.env', override: bool = False) -> bool:
    """
    Load environment variables from .env file.
    
    Args:
        env_file: Path to .env file (default: '.env')
        override: Override existing environment variables
    
    Returns:
        True if file was loaded successfully
    
    Example:
        >>> load_env('.env')
        >>> api_key = os.getenv('DEEPSEEK_API_KEY')
    """
    env_path = Path(env_file)
    
    if not env_path.exists():
        return False
    
    load_dotenv(env_path, override=override)
    return True


def load_config(config_file: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from YAML or JSON file.
    
    Args:
        config_file: Path to config file (.yaml, .yml, or .json)
    
    Returns:
        Configuration dictionary
    
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If file format is not supported
    
    Example:
        >>> config = load_config('config.yaml')
        >>> model_name = config['models']['deepseek']['model']
    """
    config_path = Path(config_file)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    suffix = config_path.suffix.lower()
    
    with open(config_path, 'r', encoding='utf-8') as f:
        if suffix in ['.yaml', '.yml']:
            return yaml.safe_load(f) or {}
        elif suffix == '.json':
            return json.load(f)
        else:
            raise ValueError(f"Unsupported format: {suffix}. Use .yaml, .yml, or .json")


def save_config(config: Dict[str, Any], config_file: Union[str, Path]) -> None:
    """
    Save configuration to YAML or JSON file.
    
    Args:
        config: Configuration dictionary
        config_file: Path to save config file
    
    Example:
        >>> config = {'model': 'deepseek-chat', 'temperature': 0.7}
        >>> save_config(config, 'my_config.yaml')
    """
    config_path = Path(config_file)
    suffix = config_path.suffix.lower()
    
    with open(config_path, 'w', encoding='utf-8') as f:
        if suffix in ['.yaml', '.yml']:
            yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)
        elif suffix == '.json':
            json.dump(config, f, indent=2)
        else:
            raise ValueError(f"Unsupported format: {suffix}. Use .yaml, .yml, or .json")


def get_nested(config: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    """
    Get nested configuration value safely.
    
    Args:
        config: Configuration dictionary
        *keys: Nested keys to access
        default: Default value if key not found
    
    Returns:
        Configuration value or default
    
    Example:
        >>> config = {'models': {'deepseek': {'model': 'deepseek-chat'}}}
        >>> model = get_nested(config, 'models', 'deepseek', 'model')
        >>> # Returns: 'deepseek-chat'
        >>> 
        >>> missing = get_nested(config, 'models', 'gpt4', 'model', default='not-found')
        >>> # Returns: 'not-found'
    """
    value = config
    for key in keys:
        if isinstance(value, dict):
            value = value.get(key)
            if value is None:
                return default
        else:
            return default
    return value
