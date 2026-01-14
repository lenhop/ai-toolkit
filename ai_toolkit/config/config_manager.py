"""
Configuration manager for loading and managing application configurations.

This module provides functionality to load, validate, and manage
configuration files in YAML and JSON formats.

Classes:
    ConfigManager: Manager for application configurations
        - Loads YAML and JSON configuration files
        - Supports environment variable substitution
        - Provides nested key access with dot notation
        - Merges multiple configuration files
        
        Methods:
            __init__(config_path, auto_load, logger): Initialize manager
            load_config(config_path): Load configuration from file
            save_config(config_path, config): Save configuration to file
            get_config(key, default): Get configuration value
            set_config(key, value): Set configuration value
            delete_config(key): Delete configuration key
            has_config(key): Check if key exists
            update_config(updates, merge): Update configuration
            reset_config(): Reset to original configuration
            clear_config(): Clear all configuration
            get_all_keys(prefix, separator): Get all configuration keys
            to_dict(): Convert configuration to dictionary
            from_dict(config): Load configuration from dictionary
            merge_configs(*config_paths): Merge multiple configuration files
"""

import os
import yaml
import json
import logging
from typing import Any, Dict, Optional, Union, List
from pathlib import Path
from copy import deepcopy


class ConfigManager:
    """
    Configuration manager for loading and managing configurations.
    
    Supports YAML and JSON configuration files with environment variable
    substitution and nested configuration access.
    """
    
    def __init__(self, 
                 config_path: Optional[Union[str, Path]] = None,
                 auto_load: bool = True,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Path to configuration file
            auto_load: Whether to automatically load config on init
            logger: Logger for debugging
        """
        self.config_path = Path(config_path) if config_path else None
        self.logger = logger or logging.getLogger(__name__)
        self._config: Dict[str, Any] = {}
        self._original_config: Dict[str, Any] = {}
        
        if auto_load and self.config_path:
            self.load_config(self.config_path)
    
    def load_config(self, config_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load configuration from a file.
        
        Args:
            config_path: Path to configuration file (YAML or JSON)
            
        Returns:
            Loaded configuration dictionary
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If file format is not supported
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        self.config_path = config_path
        
        # Determine file format
        suffix = config_path.suffix.lower()
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                if suffix in ['.yaml', '.yml']:
                    self._config = yaml.safe_load(f) or {}
                elif suffix == '.json':
                    self._config = json.load(f)
                else:
                    raise ValueError(f"Unsupported config format: {suffix}")
            
            # Store original config for reset
            self._original_config = deepcopy(self._config)
            
            # Substitute environment variables
            self._config = self._substitute_env_vars(self._config)
            
            self.logger.info(f"Loaded configuration from: {config_path}")
            return self._config
            
        except Exception as e:
            self.logger.error(f"Failed to load config from {config_path}: {e}")
            raise
    
    def _substitute_env_vars(self, config: Any) -> Any:
        """
        Recursively substitute environment variables in config.
        
        Supports ${VAR_NAME} and ${VAR_NAME:default_value} syntax.
        
        Args:
            config: Configuration value (can be dict, list, or string)
            
        Returns:
            Configuration with substituted values
        """
        if isinstance(config, dict):
            return {key: self._substitute_env_vars(value) 
                   for key, value in config.items()}
        
        elif isinstance(config, list):
            return [self._substitute_env_vars(item) for item in config]
        
        elif isinstance(config, str):
            # Handle ${VAR_NAME} or ${VAR_NAME:default}
            import re
            pattern = r'\$\{([^}:]+)(?::([^}]*))?\}'
            
            def replace_env(match):
                var_name = match.group(1)
                default_value = match.group(2) if match.group(2) is not None else ''
                return os.environ.get(var_name, default_value)
            
            return re.sub(pattern, replace_env, config)
        
        else:
            return config
    
    def save_config(self, 
                   config_path: Optional[Union[str, Path]] = None,
                   config: Optional[Dict[str, Any]] = None) -> None:
        """
        Save configuration to a file.
        
        Args:
            config_path: Path to save config (defaults to loaded path)
            config: Configuration to save (defaults to current config)
            
        Raises:
            ValueError: If no config path is specified
        """
        config_path = Path(config_path) if config_path else self.config_path
        
        if not config_path:
            raise ValueError("No configuration path specified")
        
        config_to_save = config if config is not None else self._config
        
        # Determine file format
        suffix = config_path.suffix.lower()
        
        try:
            # Ensure directory exists
            config_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(config_path, 'w', encoding='utf-8') as f:
                if suffix in ['.yaml', '.yml']:
                    yaml.safe_dump(config_to_save, f, default_flow_style=False, 
                                 allow_unicode=True, sort_keys=False)
                elif suffix == '.json':
                    json.dump(config_to_save, f, indent=2, ensure_ascii=False)
                else:
                    raise ValueError(f"Unsupported config format: {suffix}")
            
            self.logger.info(f"Saved configuration to: {config_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save config to {config_path}: {e}")
            raise
    
    def get_config(self, key: Optional[str] = None, default: Any = None) -> Any:
        """
        Get configuration value by key.
        
        Supports nested keys using dot notation (e.g., 'models.deepseek.api_key').
        
        Args:
            key: Configuration key (None returns entire config)
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        if key is None:
            return self._config
        
        # Handle nested keys
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set_config(self, key: str, value: Any) -> None:
        """
        Set configuration value by key.
        
        Supports nested keys using dot notation (e.g., 'models.deepseek.api_key').
        Creates intermediate dictionaries as needed.
        
        Args:
            key: Configuration key
            value: Value to set
        """
        keys = key.split('.')
        config = self._config
        
        # Navigate to the parent of the target key
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            elif not isinstance(config[k], dict):
                # Convert non-dict to dict
                config[k] = {}
            config = config[k]
        
        # Set the value
        config[keys[-1]] = value
        self.logger.debug(f"Set config: {key} = {value}")
    
    def delete_config(self, key: str) -> bool:
        """
        Delete configuration value by key.
        
        Args:
            key: Configuration key to delete
            
        Returns:
            True if key was deleted, False if not found
        """
        keys = key.split('.')
        config = self._config
        
        # Navigate to the parent of the target key
        for k in keys[:-1]:
            if k not in config or not isinstance(config[k], dict):
                return False
            config = config[k]
        
        # Delete the key
        if keys[-1] in config:
            del config[keys[-1]]
            self.logger.debug(f"Deleted config key: {key}")
            return True
        
        return False
    
    def has_config(self, key: str) -> bool:
        """
        Check if configuration key exists.
        
        Args:
            key: Configuration key to check
            
        Returns:
            True if key exists, False otherwise
        """
        return self.get_config(key) is not None
    
    def update_config(self, updates: Dict[str, Any], merge: bool = True) -> None:
        """
        Update configuration with new values.
        
        Args:
            updates: Dictionary of updates
            merge: If True, merge with existing config; if False, replace
        """
        if merge:
            self._merge_dict(self._config, updates)
        else:
            self._config = updates
        
        self.logger.info(f"Updated configuration with {len(updates)} items")
    
    def _merge_dict(self, target: Dict[str, Any], source: Dict[str, Any]) -> None:
        """
        Recursively merge source dict into target dict.
        
        Args:
            target: Target dictionary to merge into
            source: Source dictionary to merge from
        """
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._merge_dict(target[key], value)
            else:
                target[key] = value
    
    def reset_config(self) -> None:
        """Reset configuration to original loaded state."""
        self._config = deepcopy(self._original_config)
        self.logger.info("Reset configuration to original state")
    
    def clear_config(self) -> None:
        """Clear all configuration."""
        self._config = {}
        self.logger.info("Cleared all configuration")
    
    def get_all_keys(self, prefix: str = '', separator: str = '.') -> List[str]:
        """
        Get all configuration keys.
        
        Args:
            prefix: Prefix to filter keys
            separator: Separator for nested keys
            
        Returns:
            List of all configuration keys
        """
        def _get_keys(config: Dict[str, Any], current_prefix: str = '') -> List[str]:
            keys = []
            for key, value in config.items():
                full_key = f"{current_prefix}{separator}{key}" if current_prefix else key
                keys.append(full_key)
                
                if isinstance(value, dict):
                    keys.extend(_get_keys(value, full_key))
            
            return keys
        
        all_keys = _get_keys(self._config)
        
        if prefix:
            all_keys = [k for k in all_keys if k.startswith(prefix)]
        
        return all_keys
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Get configuration as dictionary.
        
        Returns:
            Configuration dictionary (deep copy)
        """
        return deepcopy(self._config)
    
    def from_dict(self, config: Dict[str, Any]) -> None:
        """
        Load configuration from dictionary.
        
        Args:
            config: Configuration dictionary
        """
        self._config = deepcopy(config)
        self._original_config = deepcopy(config)
        self.logger.info("Loaded configuration from dictionary")
    
    def merge_configs(self, *config_paths: Union[str, Path]) -> Dict[str, Any]:
        """
        Merge multiple configuration files.
        
        Later configs override earlier ones.
        
        Args:
            config_paths: Paths to configuration files
            
        Returns:
            Merged configuration
        """
        merged = {}
        
        for path in config_paths:
            temp_manager = ConfigManager(config_path=path, auto_load=True)
            self._merge_dict(merged, temp_manager.to_dict())
        
        self._config = merged
        self._original_config = deepcopy(merged)
        
        self.logger.info(f"Merged {len(config_paths)} configuration files")
        return merged
    
    def __repr__(self) -> str:
        """String representation of ConfigManager."""
        return f"ConfigManager(config_path={self.config_path}, keys={len(self.get_all_keys())})"
    
    def __str__(self) -> str:
        """String representation of configuration."""
        return yaml.dump(self._config, default_flow_style=False, allow_unicode=True)
