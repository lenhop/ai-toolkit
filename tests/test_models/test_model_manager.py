"""
Tests for ModelManager class.
"""

import pytest
from unittest.mock import patch, MagicMock, mock_open
import tempfile
import yaml
from pathlib import Path

from ai_toolkit.models.model_manager import ModelManager
from ai_toolkit.models.model_config import ModelConfig


class TestModelManager:
    """Test ModelManager class."""
    
    def test_init_without_config(self):
        """Test initializing ModelManager without configuration file."""
        manager = ModelManager()
        assert isinstance(manager, ModelManager)
        assert len(manager._models) == 0
        assert len(manager._providers) == 0
    
    def test_load_config_from_file(self):
        """Test loading configuration from YAML file."""
        config_data = {
            'models': {
                'deepseek': {
                    'api_key': 'sk-test123',
                    'base_url': 'https://api.deepseek.com',
                    'model': 'deepseek-chat',
                    'temperature': 0.7
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name
        
        try:
            manager = ModelManager(config_path)
            assert 'deepseek' in manager._configs
            assert manager._configs['deepseek'].api_key == 'sk-test123'
        finally:
            Path(config_path).unlink()
    
    def test_load_config_file_not_found(self):
        """Test error when configuration file is not found."""
        with pytest.raises(FileNotFoundError):
            ModelManager('nonexistent.yaml')
    
    @patch.dict('os.environ', {
        'DEEPSEEK_API_KEY': 'sk-deepseek123'
    })
    def test_create_model_from_env(self):
        """Test creating model from environment variables."""
        manager = ModelManager()
        
        # Mock the provider creation to avoid actual API calls
        with patch('ai_toolkit.models.model_manager.create_provider') as mock_create:
            mock_provider = MagicMock()
            mock_model = MagicMock()
            mock_provider.create_model.return_value = mock_model
            mock_create.return_value = mock_provider
            
            model = manager.create_model('deepseek')
            
            assert model is mock_model
            assert 'deepseek:deepseek-chat' in manager._models
            mock_create.assert_called_once()
    
    def test_create_model_unsupported_provider(self):
        """Test error when creating model with unsupported provider."""
        manager = ModelManager()
        
        with pytest.raises(ValueError, match="Failed to create configuration"):
            manager.create_model('unsupported_provider')
    
    @patch.dict('os.environ', {
        'DEEPSEEK_API_KEY': 'sk-deepseek123'
    })
    def test_get_model(self):
        """Test getting cached model."""
        manager = ModelManager()
        
        with patch('ai_toolkit.models.model_manager.create_provider') as mock_create:
            mock_provider = MagicMock()
            mock_model = MagicMock()
            mock_provider.create_model.return_value = mock_model
            mock_create.return_value = mock_provider
            
            # Create model first
            created_model = manager.create_model('deepseek')
            
            # Get cached model
            cached_model = manager.get_model('deepseek')
            
            assert cached_model is created_model
            assert cached_model is mock_model
    
    def test_get_model_not_cached(self):
        """Test getting model that is not cached."""
        manager = ModelManager()
        
        model = manager.get_model('deepseek')
        assert model is None
    
    @patch.dict('os.environ', {
        'DEEPSEEK_API_KEY': 'sk-deepseek123',
        'QWEN_API_KEY': 'sk-qwen123'
    })
    def test_list_models(self):
        """Test listing available models."""
        manager = ModelManager()
        
        with patch('ai_toolkit.models.model_manager.create_provider') as mock_create:
            mock_provider = MagicMock()
            mock_model = MagicMock()
            mock_model._llm_type = 'chat'
            mock_provider.create_model.return_value = mock_model
            mock_create.return_value = mock_provider
            
            # Create some models
            manager.create_model('deepseek')
            manager.create_model('qwen')
            
            models = manager.list_models()
            
            assert len(models) >= 2
            provider_names = [m['provider'] for m in models]
            assert 'deepseek' in provider_names
            assert 'qwen' in provider_names
    
    def test_list_providers(self):
        """Test listing available providers."""
        manager = ModelManager()
        
        providers = manager.list_providers()
        
        assert 'deepseek' in providers
        assert 'qwen' in providers
        assert 'glm' in providers
    
    @patch.dict('os.environ', {
        'DEEPSEEK_API_KEY': 'sk-deepseek123'
    })
    def test_remove_model(self):
        """Test removing cached model."""
        manager = ModelManager()
        
        with patch('ai_toolkit.models.model_manager.create_provider') as mock_create:
            mock_provider = MagicMock()
            mock_model = MagicMock()
            mock_provider.create_model.return_value = mock_model
            mock_create.return_value = mock_provider
            
            # Create model
            manager.create_model('deepseek')
            assert 'deepseek:deepseek-chat' in manager._models
            
            # Remove model
            removed = manager.remove_model('deepseek')
            assert removed is True
            assert 'deepseek:deepseek-chat' not in manager._models
    
    def test_remove_model_not_found(self):
        """Test removing model that doesn't exist."""
        manager = ModelManager()
        
        removed = manager.remove_model('nonexistent')
        assert removed is False
    
    @patch.dict('os.environ', {
        'DEEPSEEK_API_KEY': 'sk-deepseek123'
    })
    def test_clear_cache(self):
        """Test clearing model cache."""
        manager = ModelManager()
        
        with patch('ai_toolkit.models.model_manager.create_provider') as mock_create:
            mock_provider = MagicMock()
            mock_model = MagicMock()
            mock_provider.create_model.return_value = mock_model
            mock_create.return_value = mock_provider
            
            # Create model
            manager.create_model('deepseek')
            assert len(manager._models) > 0
            
            # Clear cache
            manager.clear_cache()
            assert len(manager._models) == 0
            assert len(manager._providers) == 0
    
    @patch.dict('os.environ', {
        'DEEPSEEK_API_KEY': 'sk-deepseek123'
    })
    def test_get_model_info(self):
        """Test getting model information."""
        manager = ModelManager()
        
        with patch('ai_toolkit.models.model_manager.create_provider') as mock_create:
            mock_provider = MagicMock()
            mock_model = MagicMock()
            mock_provider.create_model.return_value = mock_model
            mock_create.return_value = mock_provider
            
            # Create model
            manager.create_model('deepseek')
            
            # Get model info
            info = manager.get_model_info('deepseek')
            
            assert info is not None
            assert info['provider'] == 'deepseek'
            assert info['cached'] is True
            assert 'temperature' in info
            assert 'max_tokens' in info
    
    def test_get_model_info_not_found(self):
        """Test getting info for non-existent model."""
        manager = ModelManager()
        
        info = manager.get_model_info('nonexistent')
        assert info is None