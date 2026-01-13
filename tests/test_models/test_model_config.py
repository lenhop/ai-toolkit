"""
Tests for model configuration classes.
"""

import pytest
import os
from unittest.mock import patch

from ai_toolkit.models.model_config import ModelConfig, ProviderConfig, load_config_from_env, validate_config


class TestModelConfig:
    """Test ModelConfig class."""
    
    def test_valid_config(self):
        """Test creating a valid model configuration."""
        config = ModelConfig(
            api_key="sk-test123",
            base_url="https://api.example.com",
            model="test-model",
            temperature=0.7,
            max_tokens=2048
        )
        
        assert config.api_key == "sk-test123"
        assert config.base_url == "https://api.example.com"
        assert config.model == "test-model"
        assert config.temperature == 0.7
        assert config.max_tokens == 2048
        assert config.top_p == 0.9  # default value
    
    def test_invalid_api_key(self):
        """Test validation of invalid API keys."""
        with pytest.raises(ValueError, match="API key must be a valid key"):
            ModelConfig(
                api_key="your_api_key_here",
                base_url="https://api.example.com",
                model="test-model"
            )
    
    def test_invalid_base_url(self):
        """Test validation of invalid base URLs."""
        with pytest.raises(ValueError, match="Base URL must start with http"):
            ModelConfig(
                api_key="sk-test123",
                base_url="ftp://api.example.com",
                model="test-model"
            )
    
    def test_empty_model_name(self):
        """Test validation of empty model name."""
        with pytest.raises(ValueError, match="Model name cannot be empty"):
            ModelConfig(
                api_key="sk-test123",
                base_url="https://api.example.com",
                model=""
            )
    
    def test_temperature_bounds(self):
        """Test temperature parameter bounds."""
        # Valid temperature
        config = ModelConfig(
            api_key="sk-test123",
            base_url="https://api.example.com",
            model="test-model",
            temperature=1.5
        )
        assert config.temperature == 1.5
        
        # Invalid temperature (too high)
        with pytest.raises(ValueError):
            ModelConfig(
                api_key="sk-test123",
                base_url="https://api.example.com",
                model="test-model",
                temperature=3.0
            )


class TestProviderConfig:
    """Test ProviderConfig class."""
    
    def test_valid_provider_config(self):
        """Test creating a valid provider configuration."""
        config = ProviderConfig(
            name="test-provider",
            class_path="test.providers.TestProvider",
            supported_models=["model1", "model2"],
            default_model="model1"
        )
        
        assert config.name == "test-provider"
        assert config.supported_models == ["model1", "model2"]
        assert config.default_model == "model1"
    
    def test_empty_supported_models(self):
        """Test validation of empty supported models list."""
        with pytest.raises(ValueError, match="Supported models list cannot be empty"):
            ProviderConfig(
                name="test-provider",
                class_path="test.providers.TestProvider",
                supported_models=[],
                default_model="model1"
            )
    
    def test_invalid_default_model(self):
        """Test validation of default model not in supported models."""
        with pytest.raises(ValueError, match="Default model .* must be in supported models"):
            ProviderConfig(
                name="test-provider",
                class_path="test.providers.TestProvider",
                supported_models=["model1", "model2"],
                default_model="model3"
            )


class TestLoadConfigFromEnv:
    """Test loading configuration from environment variables."""
    
    @patch.dict(os.environ, {
        'DEEPSEEK_API_KEY': 'sk-deepseek123',
        'DEEPSEEK_MODEL': 'deepseek-chat',
        'DEEPSEEK_TEMPERATURE': '0.8'
    })
    def test_load_deepseek_config(self):
        """Test loading DeepSeek configuration from environment."""
        config = load_config_from_env('deepseek')
        
        assert config.api_key == 'sk-deepseek123'
        assert config.model == 'deepseek-chat'
        assert config.temperature == 0.8
        assert config.base_url == 'https://api.deepseek.com'
    
    @patch.dict(os.environ, {
        'QWEN_API_KEY': 'sk-qwen123'
    })
    def test_load_qwen_config_defaults(self):
        """Test loading Qwen configuration with defaults."""
        config = load_config_from_env('qwen')
        
        assert config.api_key == 'sk-qwen123'
        assert config.model == 'qwen-turbo'
        assert config.temperature == 0.7  # default
        assert config.base_url == 'https://dashscope.aliyuncs.com/compatible-mode/v1'
    
    def test_missing_api_key(self):
        """Test error when API key is missing."""
        with pytest.raises(ValueError, match="Environment variable .* is required"):
            load_config_from_env('nonexistent')
    
    @patch.dict(os.environ, {
        'GLM_API_KEY': 'glm-test123'
    })
    def test_load_glm_config(self):
        """Test loading GLM configuration."""
        config = load_config_from_env('glm')
        
        assert config.api_key == 'glm-test123'
        assert config.model == 'glm-4.6'
        assert config.base_url == 'https://open.bigmodel.cn/api/paas/v4'


class TestValidateConfig:
    """Test configuration validation."""
    
    def test_validate_valid_config(self):
        """Test validating a valid configuration."""
        config = ModelConfig(
            api_key="sk-test123",
            base_url="https://api.example.com",
            model="test-model"
        )
        
        assert validate_config(config) is True
    
    def test_validate_invalid_config(self):
        """Test validating an invalid configuration."""
        # This should be caught by Pydantic validation
        with pytest.raises(ValueError):
            config = ModelConfig(
                api_key="",  # Invalid empty API key
                base_url="https://api.example.com",
                model="test-model"
            )
            validate_config(config)