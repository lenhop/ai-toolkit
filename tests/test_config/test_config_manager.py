"""
Unit tests for ConfigManager.
"""

import pytest
import tempfile
import yaml
import json
from pathlib import Path

from ai_toolkit.config import ConfigManager


class TestConfigManager:
    """Test cases for ConfigManager."""
    
    @pytest.fixture
    def sample_config(self):
        """Sample configuration dictionary."""
        return {
            'app': {
                'name': 'test_app',
                'version': '1.0.0',
                'debug': True
            },
            'database': {
                'host': 'localhost',
                'port': 5432,
                'name': 'testdb'
            },
            'api_keys': {
                'service1': 'key123',
                'service2': 'key456'
            }
        }
    
    @pytest.fixture
    def yaml_config_file(self, sample_config):
        """Create temporary YAML config file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(sample_config, f)
            return Path(f.name)
    
    @pytest.fixture
    def json_config_file(self, sample_config):
        """Create temporary JSON config file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(sample_config, f)
            return Path(f.name)
    
    def test_init_without_config(self):
        """Test initialization without config file."""
        manager = ConfigManager(auto_load=False)
        assert manager.config_path is None
        assert manager._config == {}
    
    def test_load_yaml_config(self, yaml_config_file, sample_config):
        """Test loading YAML configuration."""
        manager = ConfigManager(config_path=yaml_config_file)
        assert manager._config == sample_config
        yaml_config_file.unlink()
    
    def test_load_json_config(self, json_config_file, sample_config):
        """Test loading JSON configuration."""
        manager = ConfigManager(config_path=json_config_file)
        assert manager._config == sample_config
        json_config_file.unlink()
    
    def test_load_nonexistent_file(self):
        """Test loading non-existent file raises error."""
        with pytest.raises(FileNotFoundError):
            ConfigManager(config_path='nonexistent.yaml')
    
    def test_load_unsupported_format(self):
        """Test loading unsupported format raises error."""
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
            f.write(b'test')
            temp_path = Path(f.name)
        
        with pytest.raises(ValueError):
            ConfigManager(config_path=temp_path)
        
        temp_path.unlink()
    
    def test_get_config_all(self, yaml_config_file, sample_config):
        """Test getting entire configuration."""
        manager = ConfigManager(config_path=yaml_config_file)
        assert manager.get_config() == sample_config
        yaml_config_file.unlink()
    
    def test_get_config_simple_key(self, yaml_config_file):
        """Test getting simple key."""
        manager = ConfigManager(config_path=yaml_config_file)
        assert manager.get_config('app') == {
            'name': 'test_app',
            'version': '1.0.0',
            'debug': True
        }
        yaml_config_file.unlink()
    
    def test_get_config_nested_key(self, yaml_config_file):
        """Test getting nested key with dot notation."""
        manager = ConfigManager(config_path=yaml_config_file)
        assert manager.get_config('app.name') == 'test_app'
        assert manager.get_config('database.port') == 5432
        yaml_config_file.unlink()
    
    def test_get_config_nonexistent_key(self, yaml_config_file):
        """Test getting non-existent key returns default."""
        manager = ConfigManager(config_path=yaml_config_file)
        assert manager.get_config('nonexistent') is None
        assert manager.get_config('nonexistent', default='default') == 'default'
        yaml_config_file.unlink()
    
    def test_set_config_simple(self):
        """Test setting simple configuration value."""
        manager = ConfigManager(auto_load=False)
        manager.set_config('key', 'value')
        assert manager.get_config('key') == 'value'
    
    def test_set_config_nested(self):
        """Test setting nested configuration value."""
        manager = ConfigManager(auto_load=False)
        manager.set_config('app.name', 'test')
        assert manager.get_config('app.name') == 'test'
        assert manager.get_config('app') == {'name': 'test'}
    
    def test_set_config_deep_nested(self):
        """Test setting deeply nested value."""
        manager = ConfigManager(auto_load=False)
        manager.set_config('a.b.c.d', 'value')
        assert manager.get_config('a.b.c.d') == 'value'
    
    def test_delete_config(self):
        """Test deleting configuration key."""
        manager = ConfigManager(auto_load=False)
        manager.set_config('key', 'value')
        assert manager.delete_config('key') is True
        assert manager.get_config('key') is None
    
    def test_delete_config_nested(self):
        """Test deleting nested configuration key."""
        manager = ConfigManager(auto_load=False)
        manager.set_config('app.name', 'test')
        assert manager.delete_config('app.name') is True
        assert manager.get_config('app.name') is None
    
    def test_delete_nonexistent_key(self):
        """Test deleting non-existent key returns False."""
        manager = ConfigManager(auto_load=False)
        assert manager.delete_config('nonexistent') is False
    
    def test_has_config(self):
        """Test checking if configuration key exists."""
        manager = ConfigManager(auto_load=False)
        manager.set_config('key', 'value')
        assert manager.has_config('key') is True
        assert manager.has_config('nonexistent') is False
    
    def test_update_config_merge(self, yaml_config_file):
        """Test updating configuration with merge."""
        manager = ConfigManager(config_path=yaml_config_file)
        updates = {
            'app': {'new_field': 'new_value'},
            'new_section': {'field': 'value'}
        }
        manager.update_config(updates, merge=True)
        
        assert manager.get_config('app.name') == 'test_app'  # Original preserved
        assert manager.get_config('app.new_field') == 'new_value'  # New added
        assert manager.get_config('new_section.field') == 'value'
        yaml_config_file.unlink()
    
    def test_update_config_replace(self, yaml_config_file):
        """Test updating configuration with replace."""
        manager = ConfigManager(config_path=yaml_config_file)
        new_config = {'new': 'config'}
        manager.update_config(new_config, merge=False)
        
        assert manager.get_config() == new_config
        yaml_config_file.unlink()
    
    def test_reset_config(self, yaml_config_file, sample_config):
        """Test resetting configuration to original."""
        manager = ConfigManager(config_path=yaml_config_file)
        manager.set_config('new_key', 'new_value')
        manager.reset_config()
        
        assert manager.get_config() == sample_config
        assert manager.get_config('new_key') is None
        yaml_config_file.unlink()
    
    def test_clear_config(self, yaml_config_file):
        """Test clearing all configuration."""
        manager = ConfigManager(config_path=yaml_config_file)
        manager.clear_config()
        assert manager.get_config() == {}
        yaml_config_file.unlink()
    
    def test_get_all_keys(self, yaml_config_file):
        """Test getting all configuration keys."""
        manager = ConfigManager(config_path=yaml_config_file)
        keys = manager.get_all_keys()
        
        assert 'app' in keys
        assert 'app.name' in keys
        assert 'database.host' in keys
        yaml_config_file.unlink()
    
    def test_get_all_keys_with_prefix(self, yaml_config_file):
        """Test getting keys with prefix filter."""
        manager = ConfigManager(config_path=yaml_config_file)
        keys = manager.get_all_keys(prefix='app')
        
        assert all(k.startswith('app') for k in keys)
        assert 'database.host' not in keys
        yaml_config_file.unlink()
    
    def test_to_dict(self, yaml_config_file, sample_config):
        """Test converting configuration to dictionary."""
        manager = ConfigManager(config_path=yaml_config_file)
        config_dict = manager.to_dict()
        
        assert config_dict == sample_config
        # Ensure it's a copy
        config_dict['new'] = 'value'
        assert manager.get_config('new') is None
        yaml_config_file.unlink()
    
    def test_from_dict(self, sample_config):
        """Test loading configuration from dictionary."""
        manager = ConfigManager(auto_load=False)
        manager.from_dict(sample_config)
        
        assert manager.get_config() == sample_config
    
    def test_save_yaml_config(self, sample_config):
        """Test saving configuration to YAML file."""
        manager = ConfigManager(auto_load=False)
        manager.from_dict(sample_config)
        
        with tempfile.NamedTemporaryFile(suffix='.yaml', delete=False) as f:
            temp_path = Path(f.name)
        
        manager.save_config(temp_path)
        
        # Load and verify
        with open(temp_path, 'r') as f:
            loaded = yaml.safe_load(f)
        
        assert loaded == sample_config
        temp_path.unlink()
    
    def test_save_json_config(self, sample_config):
        """Test saving configuration to JSON file."""
        manager = ConfigManager(auto_load=False)
        manager.from_dict(sample_config)
        
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            temp_path = Path(f.name)
        
        manager.save_config(temp_path)
        
        # Load and verify
        with open(temp_path, 'r') as f:
            loaded = json.load(f)
        
        assert loaded == sample_config
        temp_path.unlink()
    
    def test_save_without_path(self):
        """Test saving without path raises error."""
        manager = ConfigManager(auto_load=False)
        with pytest.raises(ValueError):
            manager.save_config()
    
    def test_env_var_substitution(self, monkeypatch):
        """Test environment variable substitution."""
        monkeypatch.setenv('TEST_VAR', 'test_value')
        monkeypatch.setenv('PORT', '8080')
        
        config = {
            'api_key': '${TEST_VAR}',
            'port': '${PORT}',
            'url': 'http://localhost:${PORT}'
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config, f)
            temp_path = Path(f.name)
        
        manager = ConfigManager(config_path=temp_path)
        
        assert manager.get_config('api_key') == 'test_value'
        assert manager.get_config('port') == '8080'
        assert manager.get_config('url') == 'http://localhost:8080'
        
        temp_path.unlink()
    
    def test_env_var_with_default(self):
        """Test environment variable substitution with default."""
        config = {
            'value': '${NONEXISTENT_VAR:default_value}'
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config, f)
            temp_path = Path(f.name)
        
        manager = ConfigManager(config_path=temp_path)
        assert manager.get_config('value') == 'default_value'
        
        temp_path.unlink()
    
    def test_merge_configs(self, sample_config):
        """Test merging multiple configuration files."""
        # Create first config
        config1 = {'app': {'name': 'app1'}, 'db': {'host': 'host1'}}
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config1, f)
            path1 = Path(f.name)
        
        # Create second config
        config2 = {'app': {'version': '2.0'}, 'db': {'port': 5432}}
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config2, f)
            path2 = Path(f.name)
        
        manager = ConfigManager(auto_load=False)
        merged = manager.merge_configs(path1, path2)
        
        assert merged['app']['name'] == 'app1'
        assert merged['app']['version'] == '2.0'
        assert merged['db']['host'] == 'host1'
        assert merged['db']['port'] == 5432
        
        path1.unlink()
        path2.unlink()
    
    def test_repr(self, yaml_config_file):
        """Test string representation."""
        manager = ConfigManager(config_path=yaml_config_file)
        repr_str = repr(manager)
        
        assert 'ConfigManager' in repr_str
        assert str(yaml_config_file) in repr_str
        yaml_config_file.unlink()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
