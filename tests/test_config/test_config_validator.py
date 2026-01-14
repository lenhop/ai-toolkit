"""
Unit tests for ConfigValidator.
"""

import pytest
import tempfile
from pathlib import Path
from pydantic import BaseModel, Field

from ai_toolkit.config import (
    ConfigValidator,
    RequiredRule,
    TypeRule,
    RangeRule,
    PatternRule,
    ChoiceRule,
    CustomRule
)


class TestValidationRules:
    """Test cases for validation rules."""
    
    def test_required_rule_valid(self):
        """Test RequiredRule with valid values."""
        rule = RequiredRule()
        assert rule.validate('value') is True
        assert rule.validate(123) is True
        assert rule.validate(['item']) is True
        assert rule.validate({'key': 'value'}) is True
    
    def test_required_rule_invalid(self):
        """Test RequiredRule with invalid values."""
        rule = RequiredRule()
        assert rule.validate(None) is False
        assert rule.validate('') is False
        assert rule.validate([]) is False
        assert rule.validate({}) is False
    
    def test_type_rule_valid(self):
        """Test TypeRule with valid types."""
        str_rule = TypeRule(str)
        assert str_rule.validate('text') is True
        
        int_rule = TypeRule(int)
        assert int_rule.validate(123) is True
        
        list_rule = TypeRule(list)
        assert list_rule.validate([1, 2, 3]) is True
    
    def test_type_rule_invalid(self):
        """Test TypeRule with invalid types."""
        str_rule = TypeRule(str)
        assert str_rule.validate(123) is False
        
        int_rule = TypeRule(int)
        assert int_rule.validate('text') is False
    
    def test_range_rule_valid(self):
        """Test RangeRule with valid values."""
        rule = RangeRule(min_value=0, max_value=100)
        assert rule.validate(50) is True
        assert rule.validate(0) is True
        assert rule.validate(100) is True
    
    def test_range_rule_invalid(self):
        """Test RangeRule with invalid values."""
        rule = RangeRule(min_value=0, max_value=100)
        assert rule.validate(-1) is False
        assert rule.validate(101) is False
        assert rule.validate('text') is False
    
    def test_range_rule_min_only(self):
        """Test RangeRule with only minimum."""
        rule = RangeRule(min_value=0)
        assert rule.validate(0) is True
        assert rule.validate(100) is True
        assert rule.validate(-1) is False
    
    def test_range_rule_max_only(self):
        """Test RangeRule with only maximum."""
        rule = RangeRule(max_value=100)
        assert rule.validate(100) is True
        assert rule.validate(-100) is True
        assert rule.validate(101) is False
    
    def test_pattern_rule_valid(self):
        """Test PatternRule with valid patterns."""
        email_rule = PatternRule(r'^[\w\.-]+@[\w\.-]+\.\w+$')
        assert email_rule.validate('test@example.com') is True
        assert email_rule.validate('user.name@domain.co.uk') is True
    
    def test_pattern_rule_invalid(self):
        """Test PatternRule with invalid patterns."""
        email_rule = PatternRule(r'^[\w\.-]+@[\w\.-]+\.\w+$')
        assert email_rule.validate('invalid-email') is False
        assert email_rule.validate('test@') is False
        assert email_rule.validate(123) is False
    
    def test_choice_rule_valid(self):
        """Test ChoiceRule with valid choices."""
        rule = ChoiceRule(['option1', 'option2', 'option3'])
        assert rule.validate('option1') is True
        assert rule.validate('option2') is True
    
    def test_choice_rule_invalid(self):
        """Test ChoiceRule with invalid choices."""
        rule = ChoiceRule(['option1', 'option2', 'option3'])
        assert rule.validate('option4') is False
        assert rule.validate('') is False
    
    def test_custom_rule(self):
        """Test CustomRule with custom validator."""
        # Even numbers only
        rule = CustomRule(lambda x: isinstance(x, int) and x % 2 == 0)
        assert rule.validate(2) is True
        assert rule.validate(4) is True
        assert rule.validate(3) is False
        assert rule.validate('text') is False


class TestConfigValidator:
    """Test cases for ConfigValidator."""
    
    @pytest.fixture
    def validator(self):
        """Create ConfigValidator instance."""
        return ConfigValidator()
    
    @pytest.fixture
    def sample_config(self):
        """Sample configuration."""
        return {
            'app': {
                'name': 'test_app',
                'version': '1.0.0',
                'port': 8080
            },
            'database': {
                'host': 'localhost',
                'port': 5432
            }
        }
    
    def test_add_rule(self, validator):
        """Test adding validation rule."""
        rule = RequiredRule()
        validator.add_rule('key', rule)
        
        assert 'key' in validator.rules
        assert rule in validator.rules['key']
    
    def test_add_multiple_rules(self, validator):
        """Test adding multiple rules to same key."""
        rule1 = RequiredRule()
        rule2 = TypeRule(str)
        
        validator.add_rule('key', rule1)
        validator.add_rule('key', rule2)
        
        assert len(validator.rules['key']) == 2
    
    def test_add_rules_batch(self, validator):
        """Test adding rules in batch."""
        rules = [RequiredRule(), TypeRule(str)]
        validator.add_rules('key', rules)
        
        assert len(validator.rules['key']) == 2
    
    def test_validate_success(self, validator, sample_config):
        """Test successful validation."""
        validator.add_rule('app.name', RequiredRule())
        validator.add_rule('app.name', TypeRule(str))
        validator.add_rule('app.port', RangeRule(1, 65535))
        
        assert validator.validate(sample_config) is True
        assert len(validator.get_errors()) == 0
    
    def test_validate_failure(self, validator, sample_config):
        """Test validation failure."""
        validator.add_rule('app.name', TypeRule(int))  # Should be string
        
        assert validator.validate(sample_config) is False
        assert len(validator.get_errors()) > 0
    
    def test_validate_missing_key(self, validator, sample_config):
        """Test validation of missing key."""
        validator.add_rule('nonexistent', RequiredRule())
        
        assert validator.validate(sample_config) is False
    
    def test_validate_nested_key(self, validator, sample_config):
        """Test validation of nested key."""
        validator.add_rule('app.port', RangeRule(1, 65535))
        
        assert validator.validate(sample_config) is True
    
    def test_get_errors(self, validator, sample_config):
        """Test getting validation errors."""
        validator.add_rule('app.name', TypeRule(int))
        validator.validate(sample_config)
        
        errors = validator.get_errors()
        assert len(errors) > 0
        assert 'app.name' in errors[0]
    
    def test_clear_rules(self, validator):
        """Test clearing validation rules."""
        validator.add_rule('key', RequiredRule())
        validator.clear_rules()
        
        assert len(validator.rules) == 0
        assert len(validator.errors) == 0
    
    def test_validate_model_config_valid(self, validator):
        """Test validating valid model configuration."""
        config = {
            'api_key': 'sk-test123',
            'model': 'gpt-4',
            'temperature': 0.7,
            'max_tokens': 1000
        }
        
        assert validator.validate_model_config(config) is True
    
    def test_validate_model_config_missing_key(self, validator):
        """Test validating model config with missing API key."""
        config = {
            'model': 'gpt-4'
        }
        
        assert validator.validate_model_config(config) is False
        errors = validator.get_errors()
        assert any('api_key' in error for error in errors)
    
    def test_validate_model_config_invalid_temperature(self, validator):
        """Test validating model config with invalid temperature."""
        config = {
            'api_key': 'sk-test123',
            'temperature': 3.0  # Out of range
        }
        
        assert validator.validate_model_config(config) is False
    
    def test_validate_api_keys_valid(self, validator):
        """Test validating valid API keys."""
        api_keys = {
            'openai': 'sk-1234567890abcdef',
            'anthropic': 'sk-ant-1234567890'
        }
        
        assert validator.validate_api_keys(api_keys) is True
    
    def test_validate_api_keys_empty(self, validator):
        """Test validating empty API keys."""
        api_keys = {
            'openai': '',
            'anthropic': None
        }
        
        assert validator.validate_api_keys(api_keys) is False
    
    def test_validate_api_keys_too_short(self, validator):
        """Test validating too short API keys."""
        api_keys = {
            'openai': 'short'
        }
        
        assert validator.validate_api_keys(api_keys) is False
    
    def test_validate_api_keys_placeholder(self, validator):
        """Test validating placeholder API keys."""
        api_keys = {
            'openai': 'your_api_key_here',
            'anthropic': 'replace_with_key'
        }
        
        assert validator.validate_api_keys(api_keys) is False
    
    def test_validate_file_path_exists(self, validator):
        """Test validating existing file path."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            temp_path = Path(f.name)
        
        assert validator.validate_file_path(temp_path, must_exist=True) is True
        temp_path.unlink()
    
    def test_validate_file_path_not_exists(self, validator):
        """Test validating non-existent file path."""
        assert validator.validate_file_path('/nonexistent/file.txt', must_exist=True) is False
    
    def test_validate_file_path_not_required(self, validator):
        """Test validating file path when existence not required."""
        assert validator.validate_file_path('/nonexistent/file.txt', must_exist=False) is True
    
    def test_validate_url_valid(self, validator):
        """Test validating valid URLs."""
        valid_urls = [
            'http://example.com',
            'https://example.com',
            'https://api.example.com/v1',
            'http://localhost:8080',
            'https://192.168.1.1'
        ]
        
        for url in valid_urls:
            assert validator.validate_url(url) is True, f"Failed for {url}"
    
    def test_validate_url_invalid(self, validator):
        """Test validating invalid URLs."""
        invalid_urls = [
            'not-a-url',
            'ftp://example.com',
            'example.com',
            'http://',
            ''
        ]
        
        for url in invalid_urls:
            assert validator.validate_url(url) is False, f"Should fail for {url}"
    
    def test_validate_pydantic_model_valid(self, validator):
        """Test validating data against Pydantic model."""
        class UserModel(BaseModel):
            name: str
            age: int
            email: str
        
        data = {
            'name': 'John',
            'age': 30,
            'email': 'john@example.com'
        }
        
        assert validator.validate_pydantic_model(data, UserModel) is True
    
    def test_validate_pydantic_model_invalid(self, validator):
        """Test validating invalid data against Pydantic model."""
        class UserModel(BaseModel):
            name: str
            age: int
            email: str
        
        data = {
            'name': 'John',
            'age': 'thirty',  # Should be int
            'email': 'john@example.com'
        }
        
        assert validator.validate_pydantic_model(data, UserModel) is False
        errors = validator.get_errors()
        assert len(errors) > 0
    
    def test_validate_pydantic_model_missing_field(self, validator):
        """Test validating data with missing required field."""
        class UserModel(BaseModel):
            name: str
            age: int
        
        data = {
            'name': 'John'
            # Missing 'age'
        }
        
        assert validator.validate_pydantic_model(data, UserModel) is False
    
    def test_custom_error_message(self):
        """Test custom error messages in rules."""
        rule = RequiredRule(error_message="Custom error message")
        validator = ConfigValidator()
        validator.add_rule('key', rule)
        
        validator.validate({})
        errors = validator.get_errors()
        
        assert 'Custom error message' in errors[0]
    
    def test_repr(self, validator):
        """Test string representation."""
        validator.add_rule('key', RequiredRule())
        repr_str = repr(validator)
        
        assert 'ConfigValidator' in repr_str
        assert 'rules=1' in repr_str


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
