"""
Configuration Module

Simple configuration loading for AI Toolkit.

Key Functions:
    - load_env(): Load environment variables from .env file
    - load_config(): Load YAML/JSON configuration files
    - save_config(): Save configuration to file
    - get_nested(): Safely access nested config values

Philosophy:
    Keep it simple. Use standard Python tools:
    - python-dotenv for .env files
    - PyYAML for config files
    - Standard dict access for values

Example:
    >>> from ai_toolkit.config import load_env, load_config
    >>> 
    >>> # Load environment
    >>> load_env('.env')
    >>> api_key = os.getenv('DEEPSEEK_API_KEY')
    >>> 
    >>> # Load config
    >>> config = load_config('config.yaml')
    >>> model = config['models']['deepseek']['model']
"""

from .config_loader import (
    load_env,
    load_config,
    save_config,
    get_nested
)

__all__ = [
    'load_env',
    'load_config',
    'save_config',
    'get_nested',
]
