"""
Model manager for handling multiple AI models and providers.

This module provides centralized management for AI models and providers,
including model creation, caching, and configuration management.

Classes:
    ModelManager: Central manager for AI models and providers
        - Handles model creation, caching, and configuration
        - Supports multiple providers (DeepSeek, Qwen, GLM)
        - Loads configuration from YAML files or environment variables
        
        Public Methods:
            __init__(config_path): Initialize model manager with optional config
            load_config(config_path): Load configuration from YAML file
            create_model(provider_name, model_name, **kwargs): Create a model instance
            get_model(provider_name, model_name): Get cached model instance
            list_models(): List all available models
            list_providers(): List all available providers
            remove_model(provider_name, model_name): Remove cached model
            clear_cache(): Clear all cached models
            get_model_info(provider_name, model_name): Get detailed model information
        
        Private Methods:
            _load_default_config(): Load default configuration from config files
            _load_model_config(provider_name, config_data): Load model configuration
            _load_provider_config(provider_name, config_data): Load provider configuration
            _expand_env_vars(config_data): Expand environment variables in config
"""

from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import yaml
import logging

from langchain_core.language_models.base import BaseLanguageModel

from .model_config import ModelConfig, ProviderConfig, load_config_from_env
from .model_providers import BaseModelProvider, create_provider, get_provider_class


logger = logging.getLogger(__name__)


class ModelManager:
    """
    Central manager for AI models and providers.
    
    Handles model creation, caching, and configuration management.
    """
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize model manager.
        
        Args:
            config_path: Path to configuration file (optional)
        """
        self._models: Dict[str, BaseLanguageModel] = {}
        self._providers: Dict[str, BaseModelProvider] = {}
        self._configs: Dict[str, ModelConfig] = {}
        self._provider_configs: Dict[str, ProviderConfig] = {}
        
        # Load configuration
        if config_path:
            self.load_config(config_path)
        else:
            self._load_default_config()
    
    def _load_default_config(self) -> None:
        """Load default configuration from config files."""
        try:
            # Try to load from default config paths
            config_paths = [
                Path("config/config.yaml"),
                Path("config/models.yaml"),
                Path("../config/config.yaml"),
                Path("../config/models.yaml"),
            ]
            
            for config_path in config_paths:
                if config_path.exists():
                    self.load_config(config_path)
                    break
            else:
                logger.warning("No configuration file found, using environment variables only")
                
        except Exception as e:
            logger.warning(f"Failed to load default configuration: {e}")
    
    def load_config(self, config_path: Union[str, Path]) -> None:
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to configuration file
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
            
            # Load model configurations
            if 'models' in config_data:
                for provider_name, model_config in config_data['models'].items():
                    self._load_model_config(provider_name, model_config)
            
            # Load provider configurations
            if 'providers' in config_data:
                for provider_name, provider_config in config_data['providers'].items():
                    self._load_provider_config(provider_name, provider_config)
                    
            logger.info(f"Configuration loaded from {config_path}")
            
        except Exception as e:
            raise ValueError(f"Failed to load configuration from {config_path}: {e}")
    
    def _load_model_config(self, provider_name: str, config_data: Dict[str, Any]) -> None:
        """Load model configuration for a provider."""
        try:
            # Expand environment variables in config
            expanded_config = self._expand_env_vars(config_data)
            
            config = ModelConfig(**expanded_config)
            self._configs[provider_name] = config
            
        except Exception as e:
            logger.error(f"Failed to load model config for {provider_name}: {e}")
    
    def _load_provider_config(self, provider_name: str, config_data: Dict[str, Any]) -> None:
        """Load provider configuration."""
        try:
            config = ProviderConfig(name=provider_name, **config_data)
            self._provider_configs[provider_name] = config
            
        except Exception as e:
            logger.error(f"Failed to load provider config for {provider_name}: {e}")
    
    def _expand_env_vars(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """Expand environment variables in configuration."""
        import os
        import re
        
        expanded = {}
        for key, value in config_data.items():
            if isinstance(value, str) and value.startswith('${') and value.endswith('}'):
                # Extract environment variable name
                env_var = value[2:-1]
                expanded[key] = os.getenv(env_var, value)
            else:
                expanded[key] = value
        
        return expanded
    
    def create_model(self, provider_name: str, model_name: Optional[str] = None, **kwargs) -> BaseLanguageModel:
        """
        Create a model instance.
        
        Args:
            provider_name: Name of the provider ('deepseek', 'qwen', 'glm')
            model_name: Specific model name (optional, uses default if not provided)
            **kwargs: Additional configuration parameters
            
        Returns:
            Model instance
            
        Raises:
            ValueError: If provider is not supported or configuration is invalid
        """
        provider_name = provider_name.lower()
        
        # Get or create configuration
        if provider_name in self._configs:
            config = self._configs[provider_name]
            # Override model name if provided
            if model_name:
                config = config.copy(update={'model': model_name})
        else:
            # Try to load from environment variables
            try:
                config = load_config_from_env(provider_name)
                if model_name:
                    config = config.copy(update={'model': model_name})
                self._configs[provider_name] = config
            except Exception as e:
                raise ValueError(f"Failed to create configuration for {provider_name}: {e}")
        
        # Apply additional parameters
        if kwargs:
            config = config.copy(update=kwargs)
        
        # Create provider and model
        try:
            provider = create_provider(provider_name, config)
            model = provider.create_model()
            
            # Cache the model
            cache_key = f"{provider_name}:{config.model}"
            self._models[cache_key] = model
            self._providers[cache_key] = provider
            
            logger.info(f"Created model: {cache_key}")
            return model
            
        except Exception as e:
            raise ValueError(f"Failed to create model for {provider_name}: {e}")
    
    def get_model(self, provider_name: str, model_name: Optional[str] = None) -> Optional[BaseLanguageModel]:
        """
        Get cached model instance.
        
        Args:
            provider_name: Name of the provider
            model_name: Specific model name (optional)
            
        Returns:
            Model instance if cached, None otherwise
        """
        if model_name:
            cache_key = f"{provider_name.lower()}:{model_name}"
        else:
            # Find first model for this provider
            for key in self._models.keys():
                if key.startswith(f"{provider_name.lower()}:"):
                    cache_key = key
                    break
            else:
                return None
        
        return self._models.get(cache_key)
    
    def list_models(self) -> List[Dict[str, str]]:
        """
        List all available models.
        
        Returns:
            List of model information dictionaries
        """
        models = []
        
        # Add cached models
        for cache_key, model in self._models.items():
            provider_name, model_name = cache_key.split(':', 1)
            models.append({
                'provider': provider_name,
                'model': model_name,
                'status': 'cached',
                'type': model._llm_type if hasattr(model, '_llm_type') else 'unknown'
            })
        
        # Add configured but not cached models
        for provider_name, config in self._configs.items():
            cache_key = f"{provider_name}:{config.model}"
            if cache_key not in self._models:
                models.append({
                    'provider': provider_name,
                    'model': config.model,
                    'status': 'configured',
                    'type': 'chat'
                })
        
        # Add supported models from provider configs
        for provider_name, provider_config in self._provider_configs.items():
            for model_name in provider_config.supported_models:
                cache_key = f"{provider_name}:{model_name}"
                if not any(m['provider'] == provider_name and m['model'] == model_name for m in models):
                    models.append({
                        'provider': provider_name,
                        'model': model_name,
                        'status': 'available',
                        'type': 'chat'
                    })
        
        return models
    
    def list_providers(self) -> List[str]:
        """
        List all available providers.
        
        Returns:
            List of provider names
        """
        providers = set()
        
        # Add configured providers
        providers.update(self._configs.keys())
        providers.update(self._provider_configs.keys())
        
        # Add hardcoded supported providers
        providers.update(['deepseek', 'qwen', 'glm'])
        
        return sorted(list(providers))
    
    def remove_model(self, provider_name: str, model_name: Optional[str] = None) -> bool:
        """
        Remove cached model instance.
        
        Args:
            provider_name: Name of the provider
            model_name: Specific model name (optional)
            
        Returns:
            True if model was removed, False if not found
        """
        if model_name:
            cache_key = f"{provider_name.lower()}:{model_name}"
        else:
            # Find first model for this provider
            for key in list(self._models.keys()):
                if key.startswith(f"{provider_name.lower()}:"):
                    cache_key = key
                    break
            else:
                return False
        
        if cache_key in self._models:
            del self._models[cache_key]
            if cache_key in self._providers:
                del self._providers[cache_key]
            logger.info(f"Removed model: {cache_key}")
            return True
        
        return False
    
    def clear_cache(self) -> None:
        """Clear all cached models."""
        self._models.clear()
        self._providers.clear()
        logger.info("Cleared model cache")
    
    def get_model_info(self, provider_name: str, model_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a model.
        
        Args:
            provider_name: Name of the provider
            model_name: Specific model name (optional)
            
        Returns:
            Model information dictionary
        """
        cache_key = f"{provider_name.lower()}:{model_name}" if model_name else None
        
        if not cache_key:
            # Find first model for this provider
            for key in self._models.keys():
                if key.startswith(f"{provider_name.lower()}:"):
                    cache_key = key
                    break
        
        if not cache_key:
            return None
        
        info = {
            'provider': provider_name.lower(),
            'model': model_name or cache_key.split(':', 1)[1],
            'cached': cache_key in self._models,
        }
        
        # Add configuration info
        if provider_name.lower() in self._configs:
            config = self._configs[provider_name.lower()]
            info.update({
                'temperature': config.temperature,
                'max_tokens': config.max_tokens,
                'base_url': config.base_url,
            })
        
        # Add provider info
        if provider_name.lower() in self._provider_configs:
            provider_config = self._provider_configs[provider_name.lower()]
            info.update({
                'supported_models': provider_config.supported_models,
                'pricing': provider_config.pricing,
            })
        
        return info