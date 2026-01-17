"""
Configuration validator for validating configuration values.

This module provides functionality to validate configuration files,
API keys, and other configuration parameters.

Classes:
    ValidationRule: Base class for validation rules
        - Defines interface for validation rules
        
        Methods:
            __init__(error_message): Initialize rule
            validate(value): Validate value (abstract)
            get_error_message(key, value): Get error message
    
    RequiredRule: Rule to check if value is required
        
        Methods:
            validate(value): Check if value is not None or empty
    
    TypeRule: Rule to check value type
        
        Methods:
            __init__(expected_type, error_message): Initialize with expected type
            validate(value): Check if value matches type
    
    RangeRule: Rule to check numeric value range
        
        Methods:
            __init__(min_value, max_value, error_message): Initialize with range
            validate(value): Check if value is within range
    
    PatternRule: Rule to check string pattern
        
        Methods:
            __init__(pattern, error_message): Initialize with regex pattern
            validate(value): Check if value matches pattern
    
    ChoiceRule: Rule to check if value is in allowed choices
        
        Methods:
            __init__(choices, error_message): Initialize with choices
            validate(value): Check if value is in choices
    
    CustomRule: Rule with custom validation function
        
        Methods:
            __init__(validator, error_message): Initialize with validator function
            validate(value): Check using custom validator
    
    ConfigValidator: Validator for configuration values
        - Validates configurations against rules
        - Provides specialized validation methods
        
        Methods:
            __init__(logger): Initialize validator
            add_rule(key, rule): Add validation rule
            add_rules(key, rules): Add multiple rules
            validate(config): Validate configuration
            get_errors(): Get validation errors
            clear_rules(): Clear all rules
            validate_model_config(config): Validate model configuration
            validate_api_keys(api_keys): Validate API keys
            validate_file_path(path, must_exist): Validate file path
            validate_url(url): Validate URL format
            validate_pydantic_model(data, model_class): Validate against Pydantic model
"""

import os
import re
import logging
from typing import Any, Dict, List, Optional, Callable, Union
from pathlib import Path
from pydantic import BaseModel, ValidationError, Field


class ValidationRule:
    """Base class for validation rules."""
    
    def __init__(self, error_message: Optional[str] = None):
        """
        Initialize validation rule.
        
        Args:
            error_message: Custom error message
        """
        self.error_message = error_message
    
    def validate(self, value: Any) -> bool:
        """
        Validate a value.
        
        Args:
            value: Value to validate
            
        Returns:
            True if valid, False otherwise
        """
        raise NotImplementedError
    
    def get_error_message(self, key: str, value: Any) -> str:
        """
        Get error message for validation failure.
        
        Args:
            key: Configuration key
            value: Invalid value
            
        Returns:
            Error message
        """
        if self.error_message:
            return self.error_message
        return f"Validation failed for {key}: {value}"


class RequiredRule(ValidationRule):
    """Rule to check if value is required (not None or empty)."""
    
    def validate(self, value: Any) -> bool:
        """Check if value is not None or empty."""
        if value is None:
            return False
        if isinstance(value, (str, list, dict)) and len(value) == 0:
            return False
        return True
    
    def get_error_message(self, key: str, value: Any) -> str:
        """Get error message."""
        return self.error_message or f"Required field '{key}' is missing or empty"


class TypeRule(ValidationRule):
    """Rule to check value type."""
    
    def __init__(self, expected_type: type, error_message: Optional[str] = None):
        """
        Initialize type rule.
        
        Args:
            expected_type: Expected type
            error_message: Custom error message
        """
        super().__init__(error_message)
        self.expected_type = expected_type
    
    def validate(self, value: Any) -> bool:
        """Check if value is of expected type."""
        return isinstance(value, self.expected_type)
    
    def get_error_message(self, key: str, value: Any) -> str:
        """Get error message."""
        return self.error_message or f"Field '{key}' must be of type {self.expected_type.__name__}, got {type(value).__name__}"


class RangeRule(ValidationRule):
    """Rule to check if numeric value is within range."""
    
    def __init__(self, min_value: Optional[float] = None, 
                 max_value: Optional[float] = None,
                 error_message: Optional[str] = None):
        """
        Initialize range rule.
        
        Args:
            min_value: Minimum value (inclusive)
            max_value: Maximum value (inclusive)
            error_message: Custom error message
        """
        super().__init__(error_message)
        self.min_value = min_value
        self.max_value = max_value
    
    def validate(self, value: Any) -> bool:
        """Check if value is within range."""
        if not isinstance(value, (int, float)):
            return False
        
        if self.min_value is not None and value < self.min_value:
            return False
        
        if self.max_value is not None and value > self.max_value:
            return False
        
        return True
    
    def get_error_message(self, key: str, value: Any) -> str:
        """Get error message."""
        if self.error_message:
            return self.error_message
        
        if self.min_value is not None and self.max_value is not None:
            return f"Field '{key}' must be between {self.min_value} and {self.max_value}, got {value}"
        elif self.min_value is not None:
            return f"Field '{key}' must be >= {self.min_value}, got {value}"
        else:
            return f"Field '{key}' must be <= {self.max_value}, got {value}"


class PatternRule(ValidationRule):
    """Rule to check if string matches a pattern."""
    
    def __init__(self, pattern: str, error_message: Optional[str] = None):
        """
        Initialize pattern rule.
        
        Args:
            pattern: Regular expression pattern
            error_message: Custom error message
        """
        super().__init__(error_message)
        self.pattern = re.compile(pattern)
    
    def validate(self, value: Any) -> bool:
        """Check if value matches pattern."""
        if not isinstance(value, str):
            return False
        return bool(self.pattern.match(value))
    
    def get_error_message(self, key: str, value: Any) -> str:
        """Get error message."""
        return self.error_message or f"Field '{key}' does not match required pattern"


class ChoiceRule(ValidationRule):
    """Rule to check if value is in allowed choices."""
    
    def __init__(self, choices: List[Any], error_message: Optional[str] = None):
        """
        Initialize choice rule.
        
        Args:
            choices: List of allowed values
            error_message: Custom error message
        """
        super().__init__(error_message)
        self.choices = choices
    
    def validate(self, value: Any) -> bool:
        """Check if value is in choices."""
        return value in self.choices
    
    def get_error_message(self, key: str, value: Any) -> str:
        """Get error message."""
        return self.error_message or f"Field '{key}' must be one of {self.choices}, got {value}"


class CustomRule(ValidationRule):
    """Rule with custom validation function."""
    
    def __init__(self, validator: Callable[[Any], bool], 
                 error_message: Optional[str] = None):
        """
        Initialize custom rule.
        
        Args:
            validator: Custom validation function
            error_message: Custom error message
        """
        super().__init__(error_message)
        self.validator = validator
    
    def validate(self, value: Any) -> bool:
        """Check if value passes custom validation."""
        return self.validator(value)


class ConfigValidator:
    """
    Configuration validator for validating configuration values.
    
    Provides flexible validation rules and error reporting.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the configuration validator.
        
        Args:
            logger: Logger for debugging
        """
        self.logger = logger or logging.getLogger(__name__)
        self.rules: Dict[str, List[ValidationRule]] = {}
        self.errors: List[str] = []
    
    def add_rule(self, key: str, rule: ValidationRule) -> None:
        """
        Add validation rule for a configuration key.
        
        Args:
            key: Configuration key (supports dot notation)
            rule: Validation rule
        """
        if key not in self.rules:
            self.rules[key] = []
        self.rules[key].append(rule)
    
    def add_rules(self, key: str, rules: List[ValidationRule]) -> None:
        """
        Add multiple validation rules for a configuration key.
        
        Args:
            key: Configuration key
            rules: List of validation rules
        """
        for rule in rules:
            self.add_rule(key, rule)
    
    def validate(self, config: Dict[str, Any]) -> bool:
        """
        Validate configuration against all rules.
        
        Args:
            config: Configuration dictionary to validate
            
        Returns:
            True if all validations pass, False otherwise
        """
        self.errors = []
        
        for key, rules in self.rules.items():
            value = self._get_nested_value(config, key)
            
            for rule in rules:
                if not rule.validate(value):
                    error_msg = rule.get_error_message(key, value)
                    self.errors.append(error_msg)
                    self.logger.warning(error_msg)
        
        return len(self.errors) == 0
    
    def _get_nested_value(self, config: Dict[str, Any], key: str) -> Any:
        """
        Get nested value from config using dot notation.
        
        Args:
            config: Configuration dictionary
            key: Key with dot notation
            
        Returns:
            Value or None if not found
        """
        keys = key.split('.')
        value = config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return None
        
        return value
    
    def get_errors(self) -> List[str]:
        """
        Get validation errors.
        
        Returns:
            List of error messages
        """
        return self.errors
    
    def clear_rules(self) -> None:
        """Clear all validation rules."""
        self.rules = {}
        self.errors = []
    
    def validate_model_config(self, config: Dict[str, Any]) -> bool:
        """
        Validate model configuration.
        
        Args:
            config: Model configuration dictionary
            
        Returns:
            True if valid, False otherwise
        """
        # Add standard model validation rules
        self.add_rule('api_key', RequiredRule())
        self.add_rule('api_key', TypeRule(str))
        
        if 'model' in config:
            self.add_rule('model', RequiredRule())
            self.add_rule('model', TypeRule(str))
        
        if 'temperature' in config:
            self.add_rule('temperature', TypeRule((int, float)))
            self.add_rule('temperature', RangeRule(0.0, 2.0))
        
        if 'max_tokens' in config:
            self.add_rule('max_tokens', TypeRule(int))
            self.add_rule('max_tokens', RangeRule(1, 100000))
        
        return self.validate(config)
    
    def validate_api_keys(self, api_keys: Dict[str, str]) -> bool:
        """
        Validate API keys.
        
        Args:
            api_keys: Dictionary of API keys
            
        Returns:
            True if all keys are valid, False otherwise
        """
        self.errors = []
        
        for key_name, key_value in api_keys.items():
            # Check if key is not empty
            if not key_value or not isinstance(key_value, str):
                error_msg = f"API key '{key_name}' is missing or invalid"
                self.errors.append(error_msg)
                self.logger.warning(error_msg)
                continue
            
            # Check minimum length
            if len(key_value) < 10:
                error_msg = f"API key '{key_name}' is too short (minimum 10 characters)"
                self.errors.append(error_msg)
                self.logger.warning(error_msg)
            
            # Check for placeholder values
            placeholder_patterns = ['your_', 'replace_', 'example_', 'test_']
            if any(pattern in key_value.lower() for pattern in placeholder_patterns):
                error_msg = f"API key '{key_name}' appears to be a placeholder"
                self.errors.append(error_msg)
                self.logger.warning(error_msg)
        
        return len(self.errors) == 0
    
    def validate_file_path(self, path: Union[str, Path], 
                          must_exist: bool = True) -> bool:
        """
        Validate file path.
        
        Args:
            path: File path to validate
            must_exist: Whether file must exist
            
        Returns:
            True if valid, False otherwise
        """
        self.errors = []
        
        try:
            path = Path(path)
            
            if must_exist and not path.exists():
                error_msg = f"File does not exist: {path}"
                self.errors.append(error_msg)
                self.logger.warning(error_msg)
                return False
            
            if must_exist and not path.is_file():
                error_msg = f"Path is not a file: {path}"
                self.errors.append(error_msg)
                self.logger.warning(error_msg)
                return False
            
            return True
            
        except Exception as e:
            error_msg = f"Invalid file path: {e}"
            self.errors.append(error_msg)
            self.logger.warning(error_msg)
            return False
    
    def validate_url(self, url: str) -> bool:
        """
        Validate URL format.
        
        Args:
            url: URL to validate
            
        Returns:
            True if valid, False otherwise
        """
        self.errors = []
        
        # Simple URL validation
        url_pattern = re.compile(
            r'^https?://'  # http:// or https://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain
            r'localhost|'  # localhost
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # IP
            r'(?::\d+)?'  # optional port
            r'(?:/?|[/?]\S+)$', re.IGNORECASE)
        
        if not url_pattern.match(url):
            error_msg = f"Invalid URL format: {url}"
            self.errors.append(error_msg)
            self.logger.warning(error_msg)
            return False
        
        return True
    
    def validate_pydantic_model(self, data: Dict[str, Any], 
                               model_class: type) -> bool:
        """
        Validate data against Pydantic model.
        
        Args:
            data: Data to validate
            model_class: Pydantic model class
            
        Returns:
            True if valid, False otherwise
        """
        self.errors = []
        
        try:
            model_class(**data)
            return True
        except ValidationError as e:
            for error in e.errors():
                field = '.'.join(str(loc) for loc in error['loc'])
                error_msg = f"Validation error in '{field}': {error['msg']}"
                self.errors.append(error_msg)
                self.logger.warning(error_msg)
            return False
        except Exception as e:
            error_msg = f"Validation error: {e}"
            self.errors.append(error_msg)
            self.logger.warning(error_msg)
            return False
    
    def __repr__(self) -> str:
        """String representation of ConfigValidator."""
        return f"ConfigValidator(rules={len(self.rules)}, errors={len(self.errors)})"
