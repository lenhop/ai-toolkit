"""
Unit tests for EnvLoader.
"""

import pytest
import os
import tempfile
from pathlib import Path

from ai_toolkit.config import EnvLoader


class TestEnvLoader:
    """Test cases for EnvLoader."""
    
    @pytest.fixture
    def env_file(self):
        """Create temporary .env file."""
        content = """
# Test environment file
TEST_STRING=test_value
TEST_INT=42
TEST_FLOAT=3.14
TEST_BOOL_TRUE=true
TEST_BOOL_FALSE=false
TEST_LIST=item1,item2,item3
TEST_EMPTY=
API_KEY_TEST=sk-test123
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
            f.write(content)
            return Path(f.name)
    
    def test_init_without_file(self):
        """Test initialization without env file."""
        loader = EnvLoader(auto_load=False)
        assert loader.env_file is None
    
    def test_init_with_file(self, env_file):
        """Test initialization with env file."""
        loader = EnvLoader(env_file=env_file)
        assert loader.env_file == env_file
        env_file.unlink()
    
    def test_load_env_file(self, env_file):
        """Test loading environment file."""
        loader = EnvLoader(auto_load=False)
        result = loader.load_env_file(env_file)
        
        assert result is True
        assert os.environ.get('TEST_STRING') == 'test_value'
        env_file.unlink()
    
    def test_load_nonexistent_file(self):
        """Test loading non-existent file."""
        loader = EnvLoader(auto_load=False)
        result = loader.load_env_file('/nonexistent/.env')
        assert result is False
    
    def test_get_string(self, env_file):
        """Test getting string value."""
        loader = EnvLoader(env_file=env_file)
        value = loader.get_str('TEST_STRING')
        assert value == 'test_value'
        env_file.unlink()
    
    def test_get_int(self, env_file):
        """Test getting integer value."""
        loader = EnvLoader(env_file=env_file)
        value = loader.get_int('TEST_INT')
        assert value == 42
        assert isinstance(value, int)
        env_file.unlink()
    
    def test_get_float(self, env_file):
        """Test getting float value."""
        loader = EnvLoader(env_file=env_file)
        value = loader.get_float('TEST_FLOAT')
        assert value == 3.14
        assert isinstance(value, float)
        env_file.unlink()
    
    def test_get_bool_true(self, env_file):
        """Test getting boolean true value."""
        loader = EnvLoader(env_file=env_file)
        value = loader.get_bool('TEST_BOOL_TRUE')
        assert value is True
        env_file.unlink()
    
    def test_get_bool_false(self, env_file):
        """Test getting boolean false value."""
        loader = EnvLoader(env_file=env_file)
        value = loader.get_bool('TEST_BOOL_FALSE')
        assert value is False
        env_file.unlink()
    
    def test_get_list(self, env_file):
        """Test getting list value."""
        loader = EnvLoader(env_file=env_file)
        value = loader.get_list('TEST_LIST')
        assert value == ['item1', 'item2', 'item3']
        assert isinstance(value, list)
        env_file.unlink()
    
    def test_get_with_default(self):
        """Test getting value with default."""
        loader = EnvLoader(auto_load=False)
        value = loader.get('NONEXISTENT', default='default_value')
        assert value == 'default_value'
    
    def test_get_required_missing(self):
        """Test getting required value that's missing."""
        loader = EnvLoader(auto_load=False)
        with pytest.raises(ValueError):
            loader.get('NONEXISTENT', required=True)
    
    def test_get_with_cast(self, env_file):
        """Test getting value with type casting."""
        loader = EnvLoader(env_file=env_file)
        value = loader.get('TEST_INT', cast=int)
        assert value == 42
        assert isinstance(value, int)
        env_file.unlink()
    
    def test_get_cast_failure(self, env_file):
        """Test getting value with failed cast returns default."""
        loader = EnvLoader(env_file=env_file)
        value = loader.get('TEST_STRING', cast=int, default=0)
        assert value == 0
        env_file.unlink()
    
    def test_get_api_key_standard(self, monkeypatch):
        """Test getting API key with standard naming."""
        monkeypatch.setenv('OPENAI_API_KEY', 'sk-test123')
        loader = EnvLoader(auto_load=False)
        
        key = loader.get_api_key('openai')
        assert key == 'sk-test123'
    
    def test_get_api_key_alternative(self, monkeypatch):
        """Test getting API key with alternative naming."""
        monkeypatch.setenv('OPENAI_KEY', 'sk-test456')
        loader = EnvLoader(auto_load=False)
        
        key = loader.get_api_key('openai')
        assert key == 'sk-test456'
    
    def test_get_api_key_custom_name(self, monkeypatch):
        """Test getting API key with custom name."""
        monkeypatch.setenv('CUSTOM_KEY_NAME', 'sk-custom')
        loader = EnvLoader(auto_load=False)
        
        key = loader.get_api_key('provider', key_name='CUSTOM_KEY_NAME')
        assert key == 'sk-custom'
    
    def test_get_api_key_missing_required(self):
        """Test getting required API key that's missing."""
        loader = EnvLoader(auto_load=False)
        with pytest.raises(ValueError):
            loader.get_api_key('nonexistent', required=True)
    
    def test_get_api_key_missing_optional(self):
        """Test getting optional API key that's missing."""
        loader = EnvLoader(auto_load=False)
        key = loader.get_api_key('nonexistent', required=False)
        assert key is None
    
    def test_set(self):
        """Test setting environment variable."""
        loader = EnvLoader(auto_load=False)
        loader.set('TEST_SET', 'value')
        
        assert os.environ.get('TEST_SET') == 'value'
        del os.environ['TEST_SET']
    
    def test_delete_existing(self, monkeypatch):
        """Test deleting existing environment variable."""
        monkeypatch.setenv('TEST_DELETE', 'value')
        loader = EnvLoader(auto_load=False)
        
        result = loader.delete('TEST_DELETE')
        assert result is True
        assert 'TEST_DELETE' not in os.environ
    
    def test_delete_nonexistent(self):
        """Test deleting non-existent environment variable."""
        loader = EnvLoader(auto_load=False)
        result = loader.delete('NONEXISTENT')
        assert result is False
    
    def test_has_existing(self, monkeypatch):
        """Test checking if environment variable exists."""
        monkeypatch.setenv('TEST_HAS', 'value')
        loader = EnvLoader(auto_load=False)
        
        assert loader.has('TEST_HAS') is True
    
    def test_has_nonexistent(self):
        """Test checking if non-existent variable exists."""
        loader = EnvLoader(auto_load=False)
        assert loader.has('NONEXISTENT') is False
    
    def test_get_all(self, monkeypatch):
        """Test getting all environment variables."""
        monkeypatch.setenv('TEST_ALL_1', 'value1')
        monkeypatch.setenv('TEST_ALL_2', 'value2')
        loader = EnvLoader(auto_load=False)
        
        all_vars = loader.get_all()
        assert 'TEST_ALL_1' in all_vars
        assert 'TEST_ALL_2' in all_vars
    
    def test_get_all_with_prefix(self, monkeypatch):
        """Test getting environment variables with prefix."""
        monkeypatch.setenv('PREFIX_VAR1', 'value1')
        monkeypatch.setenv('PREFIX_VAR2', 'value2')
        monkeypatch.setenv('OTHER_VAR', 'value3')
        loader = EnvLoader(auto_load=False)
        
        prefixed = loader.get_all(prefix='PREFIX_')
        assert 'PREFIX_VAR1' in prefixed
        assert 'PREFIX_VAR2' in prefixed
        assert 'OTHER_VAR' not in prefixed
    
    def test_load_from_env(self, monkeypatch):
        """Test loading multiple values from environment."""
        monkeypatch.setenv('DB_HOST', 'localhost')
        monkeypatch.setenv('DB_PORT', '5432')
        monkeypatch.setenv('DB_NAME', 'testdb')
        
        loader = EnvLoader(auto_load=False)
        mapping = {
            'host': 'DB_HOST',
            'port': 'DB_PORT',
            'name': 'DB_NAME'
        }
        
        result = loader.load_from_env(mapping)
        assert result == {
            'host': 'localhost',
            'port': '5432',
            'name': 'testdb'
        }
    
    def test_load_from_env_missing_required(self, monkeypatch):
        """Test loading with missing required variable."""
        monkeypatch.setenv('VAR1', 'value1')
        
        loader = EnvLoader(auto_load=False)
        mapping = {'var1': 'VAR1', 'var2': 'VAR2'}
        
        with pytest.raises(ValueError):
            loader.load_from_env(mapping, required=['var2'])
    
    def test_load_from_env_missing_optional(self, monkeypatch):
        """Test loading with missing optional variable."""
        monkeypatch.setenv('VAR1', 'value1')
        
        loader = EnvLoader(auto_load=False)
        mapping = {'var1': 'VAR1', 'var2': 'VAR2'}
        
        result = loader.load_from_env(mapping)
        assert 'var1' in result
        assert 'var2' not in result
    
    def test_create_env_file(self):
        """Test creating environment file."""
        loader = EnvLoader(auto_load=False)
        variables = {
            'VAR1': 'value1',
            'VAR2': 'value2',
            'VAR3': 'value with spaces'
        }
        
        with tempfile.NamedTemporaryFile(suffix='.env', delete=False) as f:
            temp_path = Path(f.name)
        temp_path.unlink()  # Delete so we can test creation
        
        loader.create_env_file(variables, temp_path)
        
        assert temp_path.exists()
        content = temp_path.read_text()
        assert 'VAR1=value1' in content
        assert 'VAR2=value2' in content
        assert 'VAR3="value with spaces"' in content
        
        temp_path.unlink()
    
    def test_create_env_file_overwrite(self):
        """Test creating environment file with overwrite."""
        loader = EnvLoader(auto_load=False)
        
        with tempfile.NamedTemporaryFile(suffix='.env', delete=False) as f:
            temp_path = Path(f.name)
        
        # Should raise error without overwrite
        with pytest.raises(FileExistsError):
            loader.create_env_file({'VAR': 'value'}, temp_path, overwrite=False)
        
        # Should succeed with overwrite
        loader.create_env_file({'VAR': 'value'}, temp_path, overwrite=True)
        
        temp_path.unlink()
    
    def test_validate_required_vars_all_present(self, monkeypatch):
        """Test validating when all required vars are present."""
        monkeypatch.setenv('REQ1', 'value1')
        monkeypatch.setenv('REQ2', 'value2')
        
        loader = EnvLoader(auto_load=False)
        result = loader.validate_required_vars(['REQ1', 'REQ2'])
        assert result is True
    
    def test_validate_required_vars_missing(self, monkeypatch):
        """Test validating when required vars are missing."""
        monkeypatch.setenv('REQ1', 'value1')
        
        loader = EnvLoader(auto_load=False)
        result = loader.validate_required_vars(['REQ1', 'REQ2'])
        assert result is False
    
    def test_get_with_fallback(self, monkeypatch):
        """Test getting value with fallback keys."""
        monkeypatch.setenv('FALLBACK_KEY2', 'value2')
        
        loader = EnvLoader(auto_load=False)
        value = loader.get_with_fallback(['FALLBACK_KEY1', 'FALLBACK_KEY2', 'FALLBACK_KEY3'])
        assert value == 'value2'
    
    def test_get_with_fallback_none_found(self):
        """Test getting value with fallback when none found."""
        loader = EnvLoader(auto_load=False)
        value = loader.get_with_fallback(['KEY1', 'KEY2'], default='default')
        assert value == 'default'
    
    def test_get_with_fallback_cast(self, monkeypatch):
        """Test getting value with fallback and casting."""
        monkeypatch.setenv('NUM_KEY', '42')
        
        loader = EnvLoader(auto_load=False)
        value = loader.get_with_fallback(['KEY1', 'NUM_KEY'], cast=int)
        assert value == 42
        assert isinstance(value, int)
    
    def test_repr(self, env_file):
        """Test string representation."""
        loader = EnvLoader(env_file=env_file)
        repr_str = repr(loader)
        
        assert 'EnvLoader' in repr_str
        assert str(env_file) in repr_str
        env_file.unlink()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
