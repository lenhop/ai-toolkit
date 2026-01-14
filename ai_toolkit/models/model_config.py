"""
Model configuration data classes using Pydantic.

This module provides Pydantic-based configuration classes for AI models,
including validation, environment variable loading, and provider configurations.

Classes:
    ModelConfig: Model configuration data class
        - Validates API keys, URLs, and model parameters
        - Supports temperature, max_tokens, penalties, timeouts
        - Includes provider-specific extra parameters
        
        Fields:
            api_key: API key for the model provider
            base_url: Base URL for the API endpoint
            model: Model name/identifier
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum tokens in response
            top_p: Top-p sampling parameter (0.0-1.0)
            frequency_penalty: Frequency penalty (-2.0-2.0)
            presence_penalty: Presence penalty (-2.0-2.0)
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
            extra_params: Additional provider-specific parameters
    
    ProviderConfig: Provider configuration data class
        - Defines provider name and supported models
        - Includes default settings and pricing information
        
        Fields:
            name: Provider name
            supported_models: List of supported model names
            default_model: Default model to use
            pricing: Pricing information dictionary
            features: Supported features list

Functions:
    load_config_from_env(provider_name): Load configuration from environment variables
    validate_config(config): Validate model configuration
"""

from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, validator
import os


class ModelConfig(BaseModel):
    """Model configuration data class."""
    
    api_key: str = Field(..., description="API key for the model provider")
    base_url: str = Field(..., description="Base URL for the API endpoint")
    model: str = Field(..., description="Model name/identifier")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: int = Field(default=4096, gt=0, description="Maximum tokens in response")
    top_p: float = Field(default=0.9, ge=0.0, le=1.0, description="Top-p sampling parameter")
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0, description="Frequency penalty")
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0, description="Presence penalty")
    timeout: int = Field(default=60, gt=0, description="Request timeout in seconds")
    max_retries: int = Field(default=3, ge=0, description="Maximum retry attempts")
    
    # Optional provider-specific settings
    extra_params: Dict[str, Any] = Field(default_factory=dict, description="Additional provider-specific parameters")
    
    class Config:
        """Pydantic configuration."""
        extra = "forbid"  # Don't allow extra fields
        validate_assignment = True  # Validate on assignment
        
    @validator('api_key')
    def validate_api_key(cls, v):
        """Validate API key is not empty or placeholder."""
        if not v or v.startswith('your_') or v == 'placeholder':
            raise ValueError("API key must be a valid key, not a placeholder")
        return v
    
    @validator('base_url')
    def validate_base_url(cls, v):
        """Validate base URL format."""
        if not v.startswith(('http://', 'https://')):
            raise ValueError("Base URL must start with http:// or https://")
        return v.rstrip('/')  # Remove trailing slash for consistency
    
    @validator('model')
    def validate_model(cls, v):
        """Validate model name is not empty."""
        if not v or not v.strip():
            raise ValueError("Model name cannot be empty")
        return v.strip()


class ProviderConfig(BaseModel):
    """Provider-specific configuration."""
    
    name: str = Field(..., description="Provider name")
    class_path: str = Field(..., description="Full class path for the provider")
    supported_models: List[str] = Field(..., description="List of supported model names")
    default_model: str = Field(..., description="Default model for this provider")
    pricing: Optional[Dict[str, float]] = Field(default=None, description="Pricing information")
    
    @validator('supported_models')
    def validate_supported_models(cls, v):
        """Validate supported models list is not empty."""
        if not v:
            raise ValueError("Supported models list cannot be empty")
        return v
    
    @validator('default_model')
    def validate_default_model(cls, v, values):
        """Validate default model is in supported models list."""
        if 'supported_models' in values and v not in values['supported_models']:
            raise ValueError(f"Default model '{v}' must be in supported models list")
        return v


def load_config_from_env(provider: str) -> ModelConfig:
    """
    Load model configuration from environment variables.
    
    Args:
        provider: Provider name (e.g., 'deepseek', 'qwen', 'glm')
        
    Returns:
        ModelConfig instance
        
    Raises:
        ValueError: If required environment variables are missing
    """
    provider_upper = provider.upper()
    
    # Required environment variables
    api_key = os.getenv(f"{provider_upper}_API_KEY")
    if not api_key:
        raise ValueError(f"Environment variable {provider_upper}_API_KEY is required")
    
    # Provider-specific base URLs
    base_url_map = {
        'deepseek': 'https://api.deepseek.com',
        'qwen': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
        'glm': 'https://open.bigmodel.cn/api/paas/v4'
    }
    
    # Provider-specific default models
    default_model_map = {
        'deepseek': 'deepseek-chat',
        'qwen': 'qwen-turbo',
        'glm': 'glm-4.6'
    }
    
    base_url = os.getenv(f"{provider_upper}_BASE_URL", base_url_map.get(provider, ''))
    model = os.getenv(f"{provider_upper}_MODEL", default_model_map.get(provider, ''))
    
    if not base_url:
        raise ValueError(f"Base URL for provider '{provider}' is not configured")
    
    if not model:
        raise ValueError(f"Model for provider '{provider}' is not configured")
    
    # Optional parameters with defaults
    temperature = float(os.getenv(f"{provider_upper}_TEMPERATURE", "0.7"))
    max_tokens = int(os.getenv(f"{provider_upper}_MAX_TOKENS", "4096"))
    top_p = float(os.getenv(f"{provider_upper}_TOP_P", "0.9"))
    frequency_penalty = float(os.getenv(f"{provider_upper}_FREQUENCY_PENALTY", "0.0"))
    presence_penalty = float(os.getenv(f"{provider_upper}_PRESENCE_PENALTY", "0.0"))
    timeout = int(os.getenv(f"{provider_upper}_TIMEOUT", "60"))
    max_retries = int(os.getenv(f"{provider_upper}_MAX_RETRIES", "3"))
    
    return ModelConfig(
        api_key=api_key,
        base_url=base_url,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        timeout=timeout,
        max_retries=max_retries
    )


def validate_config(config: ModelConfig) -> bool:
    """
    Validate model configuration.
    
    Args:
        config: ModelConfig instance to validate
        
    Returns:
        True if valid
        
    Raises:
        ValueError: If configuration is invalid
    """
    try:
        # Pydantic validation happens automatically
        # Additional custom validation can be added here
        return True
    except Exception as e:
        raise ValueError(f"Configuration validation failed: {str(e)}")