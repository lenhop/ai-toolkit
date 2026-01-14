"""
Environment variable loader for loading configuration from environment.

This module provides functionality to load configuration from environment
variables with type conversion and validation.
"""

import os
import logging
from typing import Any, Dict, Optional, Union, List, Callable, TypeVar, Type
from pathlib import Path
from dotenv import load_dotenv


T = TypeVar('T')


class EnvLoader:
    """
    Environment variable loader for loading configuration.
    
    Provides type-safe loading of environment variables with
    default values and validation.
    """
    
    def __init__(self, 
                 env_file: Optional[Union[str, Path]] = None,
                 auto_load: bool = True,
                 override: bool = False,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize the environment loader.
        
        Args:
            env_file: Path to .env file
            auto_load: Whether to automatically load .env file
            override: Whether to override existing environment variables
            logger: Logger for debugging
        """
        self.env_file = Path(env_file) if env_file else None
        self.logger = logger or logging.getLogger(__name__)
        self.override = override
        
        if auto_load:
            self.load_env_file(self.env_file, override=override)
    
    def load_env_file(self, 
                     env_file: Optional[Union[str, Path]] = None,
                     override: bool = False) -> bool:
        """
        Load environment variables from .env file.
        
        Args:
            env_file: Path to .env file
            override: Whether to override existing variables
            
        Returns:
            True if file was loaded, False otherwise
        """
        env_file = Path(env_file) if env_file else self.env_file
        
        if not env_file:
            # Try to find .env in current directory
            env_file = Path('.env')
        
        if not env_file.exists():
            self.logger.warning(f"Environment file not found: {env_file}")
            return False
        
        try:
            load_dotenv(env_file, override=override)
            self.env_file = env_file
            self.logger.info(f"Loaded environment from: {env_file}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to load environment file: {e}")
            return False
    
    def get(self, 
            key: str, 
            default: Optional[T] = None,
            cast: Optional[Type[T]] = None,
            required: bool = False) -> Optional[T]:
        """
        Get environment variable with type casting.
        
        Args:
            key: Environment variable name
            default: Default value if not found
            cast: Type to cast value to
            required: Whether variable is required
            
        Returns:
            Environment variable value
            
        Raises:
            ValueError: If required variable is not found
        """
        value = os.environ.get(key)
        
        if value is None:
            if required:
                raise ValueError(f"Required environment variable not found: {key}")
            return default
        
        # Cast to specified type
        if cast:
            try:
                return self._cast_value(value, cast)
            except Exception as e:
                self.logger.warning(f"Failed to cast {key} to {cast.__name__}: {e}")
                return default
        
        return value
    
    def _cast_value(self, value: str, cast_type: Type[T]) -> T:
        """
        Cast string value to specified type.
        
        Args:
            value: String value to cast
            cast_type: Type to cast to
            
        Returns:
            Casted value
        """
        if cast_type == bool:
            return value.lower() in ('true', '1', 'yes', 'on')
        elif cast_type == int:
            return int(value)
        elif cast_type == float:
            return float(value)
        elif cast_type == list:
            # Split by comma
            return [item.strip() for item in value.split(',')]
        elif cast_type == dict:
            # Parse as JSON
            import json
            return json.loads(value)
        else:
            return cast_type(value)
    
    def get_str(self, key: str, default: Optional[str] = None, 
                required: bool = False) -> Optional[str]:
        """Get string environment variable."""
        return self.get(key, default=default, cast=str, required=required)
    
    def get_int(self, key: str, default: Optional[int] = None,
                required: bool = False) -> Optional[int]:
        """Get integer environment variable."""
        return self.get(key, default=default, cast=int, required=required)
    
    def get_float(self, key: str, default: Optional[float] = None,
                  required: bool = False) -> Optional[float]:
        """Get float environment variable."""
        return self.get(key, default=default, cast=float, required=required)
    
    def get_bool(self, key: str, default: Optional[bool] = None,
                 required: bool = False) -> Optional[bool]:
        """Get boolean environment variable."""
        return self.get(key, default=default, cast=bool, required=required)
    
    def get_list(self, key: str, default: Optional[List[str]] = None,
                 required: bool = False) -> Optional[List[str]]:
        """Get list environment variable (comma-separated)."""
        return self.get(key, default=default, cast=list, required=required)
    
    def get_api_key(self, 
                   provider: str, 
                   key_name: Optional[str] = None,
                   required: bool = True) -> Optional[str]:
        """
        Get API key for a provider.
        
        Tries multiple naming conventions:
        - {PROVIDER}_API_KEY
        - {PROVIDER}_KEY
        - API_KEY_{PROVIDER}
        
        Args:
            provider: Provider name (e.g., 'deepseek', 'openai')
            key_name: Specific key name to use
            required: Whether key is required
            
        Returns:
            API key or None
        """
        provider_upper = provider.upper()
        
        # Try different naming conventions
        possible_keys = [
            key_name,
            f"{provider_upper}_API_KEY",
            f"{provider_upper}_KEY",
            f"API_KEY_{provider_upper}",
        ]
        
        for key in possible_keys:
            if key:
                value = os.environ.get(key)
                if value:
                    self.logger.debug(f"Found API key for {provider} in {key}")
                    return value
        
        if required:
            raise ValueError(f"API key for {provider} not found in environment")
        
        return None
    
    def set(self, key: str, value: Any) -> None:
        """
        Set environment variable.
        
        Args:
            key: Environment variable name
            value: Value to set
        """
        os.environ[key] = str(value)
        self.logger.debug(f"Set environment variable: {key}")
    
    def delete(self, key: str) -> bool:
        """
        Delete environment variable.
        
        Args:
            key: Environment variable name
            
        Returns:
            True if deleted, False if not found
        """
        if key in os.environ:
            del os.environ[key]
            self.logger.debug(f"Deleted environment variable: {key}")
            return True
        return False
    
    def has(self, key: str) -> bool:
        """
        Check if environment variable exists.
        
        Args:
            key: Environment variable name
            
        Returns:
            True if exists, False otherwise
        """
        return key in os.environ
    
    def get_all(self, prefix: Optional[str] = None) -> Dict[str, str]:
        """
        Get all environment variables.
        
        Args:
            prefix: Optional prefix to filter variables
            
        Returns:
            Dictionary of environment variables
        """
        if prefix:
            return {k: v for k, v in os.environ.items() if k.startswith(prefix)}
        return dict(os.environ)
    
    def load_from_env(self, 
                     mapping: Dict[str, str],
                     required: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Load multiple values from environment using a mapping.
        
        Args:
            mapping: Dictionary mapping config keys to env var names
            required: List of required keys
            
        Returns:
            Dictionary of loaded values
            
        Raises:
            ValueError: If required variable is not found
        """
        result = {}
        required = required or []
        
        for config_key, env_key in mapping.items():
            value = os.environ.get(env_key)
            
            if value is None and config_key in required:
                raise ValueError(f"Required environment variable not found: {env_key}")
            
            if value is not None:
                result[config_key] = value
        
        return result
    
    def create_env_file(self, 
                       variables: Dict[str, str],
                       env_file: Optional[Union[str, Path]] = None,
                       overwrite: bool = False) -> None:
        """
        Create .env file with variables.
        
        Args:
            variables: Dictionary of variables to write
            env_file: Path to .env file
            overwrite: Whether to overwrite existing file
            
        Raises:
            FileExistsError: If file exists and overwrite is False
        """
        env_file = Path(env_file) if env_file else Path('.env')
        
        if env_file.exists() and not overwrite:
            raise FileExistsError(f"Environment file already exists: {env_file}")
        
        try:
            with open(env_file, 'w', encoding='utf-8') as f:
                for key, value in variables.items():
                    # Escape quotes in value
                    if '"' in value or ' ' in value:
                        value = f'"{value}"'
                    f.write(f"{key}={value}\n")
            
            self.logger.info(f"Created environment file: {env_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to create environment file: {e}")
            raise
    
    def validate_required_vars(self, required_vars: List[str]) -> bool:
        """
        Validate that required environment variables are set.
        
        Args:
            required_vars: List of required variable names
            
        Returns:
            True if all required vars are set, False otherwise
        """
        missing = []
        
        for var in required_vars:
            if var not in os.environ:
                missing.append(var)
        
        if missing:
            self.logger.error(f"Missing required environment variables: {missing}")
            return False
        
        return True
    
    def get_with_fallback(self, 
                         keys: List[str], 
                         default: Optional[T] = None,
                         cast: Optional[Type[T]] = None) -> Optional[T]:
        """
        Get environment variable with fallback keys.
        
        Tries each key in order until one is found.
        
        Args:
            keys: List of keys to try
            default: Default value if none found
            cast: Type to cast value to
            
        Returns:
            Environment variable value
        """
        for key in keys:
            value = os.environ.get(key)
            if value is not None:
                if cast:
                    try:
                        return self._cast_value(value, cast)
                    except Exception:
                        continue
                return value
        
        return default
    
    def __repr__(self) -> str:
        """String representation of EnvLoader."""
        return f"EnvLoader(env_file={self.env_file})"
