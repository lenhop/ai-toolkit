"""
General utilities for AI Toolkit.

This module provides various utility functions for common tasks.

Functions:
    format_messages(messages): Format messages for display
    validate_input(value, value_type, min_value, max_value, allowed_values): Validate input value
    sanitize_text(text, remove_extra_spaces, remove_newlines, lowercase): Sanitize text content
    format_response(response, format_type, indent): Format response content
    truncate_text(text, max_length, suffix): Truncate text to maximum length
    chunk_text(text, chunk_size, overlap): Split text into chunks
    merge_dicts(*dicts, deep): Merge multiple dictionaries
    flatten_dict(d, parent_key, sep): Flatten nested dictionary
    unflatten_dict(d, sep): Unflatten dictionary with dot notation keys
    get_timestamp(format): Get current timestamp as string
    parse_timestamp(timestamp, format): Parse timestamp string to datetime
    calculate_hash(text, algorithm): Calculate hash of text
    retry_on_failure(func, max_attempts, delay, backoff, exceptions): Retry function on failure
    batch_process(items, batch_size, processor): Process items in batches
    filter_dict(d, keys, exclude_keys, predicate): Filter dictionary by keys or predicate
    safe_get(d, key, default, sep): Safely get nested dictionary value
    safe_set(d, key, value, sep): Safely set nested dictionary value
    is_empty(value): Check if value is empty
    coalesce(*values): Return first non-None value
"""

import re
import hashlib
from typing import Any, Dict, List, Optional, Union, Callable
from datetime import datetime


def format_messages(messages: List[Dict[str, Any]]) -> str:
    """
    Format messages for display.
    
    Args:
        messages: List of message dictionaries
        
    Returns:
        Formatted message string
    """
    formatted = []
    
    for msg in messages:
        role = msg.get('role', 'unknown')
        content = msg.get('content', '')
        formatted.append(f"{role.upper()}: {content}")
    
    return '\n'.join(formatted)


def validate_input(value: Any, 
                  value_type: type,
                  min_value: Optional[Any] = None,
                  max_value: Optional[Any] = None,
                  allowed_values: Optional[List[Any]] = None) -> bool:
    """
    Validate input value.
    
    Args:
        value: Value to validate
        value_type: Expected type
        min_value: Minimum value (for numbers)
        max_value: Maximum value (for numbers)
        allowed_values: List of allowed values
        
    Returns:
        True if valid, False otherwise
    """
    # Type check
    if not isinstance(value, value_type):
        return False
    
    # Range check for numbers
    if isinstance(value, (int, float)):
        if min_value is not None and value < min_value:
            return False
        if max_value is not None and value > max_value:
            return False
    
    # Allowed values check
    if allowed_values is not None and value not in allowed_values:
        return False
    
    return True


def sanitize_text(text: str, 
                 remove_extra_spaces: bool = True,
                 remove_newlines: bool = False,
                 lowercase: bool = False) -> str:
    """
    Sanitize text content.
    
    Args:
        text: Text to sanitize
        remove_extra_spaces: Whether to remove extra spaces
        remove_newlines: Whether to remove newlines
        lowercase: Whether to convert to lowercase
        
    Returns:
        Sanitized text
    """
    if remove_extra_spaces:
        text = re.sub(r'\s+', ' ', text)
    
    if remove_newlines:
        text = text.replace('\n', ' ').replace('\r', '')
    
    if lowercase:
        text = text.lower()
    
    return text.strip()


def format_response(response: Any, 
                   format_type: str = 'text',
                   indent: int = 2) -> str:
    """
    Format response content.
    
    Args:
        response: Response to format
        format_type: Format type ('text', 'json', 'pretty')
        indent: Indentation for JSON
        
    Returns:
        Formatted response string
    """
    if format_type == 'json':
        import json
        return json.dumps(response, indent=indent, ensure_ascii=False)
    
    elif format_type == 'pretty':
        import pprint
        return pprint.pformat(response, indent=indent)
    
    else:  # text
        return str(response)


def truncate_text(text: str, max_length: int, suffix: str = '...') -> str:
    """
    Truncate text to maximum length.
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add when truncated
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix


def chunk_text(text: str, chunk_size: int, overlap: int = 0) -> List[str]:
    """
    Split text into chunks.
    
    Args:
        text: Text to chunk
        chunk_size: Size of each chunk
        overlap: Overlap between chunks
        
    Returns:
        List of text chunks
    """
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    
    return chunks


def merge_dicts(*dicts: Dict[str, Any], deep: bool = True) -> Dict[str, Any]:
    """
    Merge multiple dictionaries.
    
    Args:
        dicts: Dictionaries to merge
        deep: Whether to perform deep merge
        
    Returns:
        Merged dictionary
    """
    result = {}
    
    for d in dicts:
        if deep:
            _deep_merge(result, d)
        else:
            result.update(d)
    
    return result


def _deep_merge(target: Dict[str, Any], source: Dict[str, Any]) -> None:
    """
    Deep merge source dict into target dict.
    
    Args:
        target: Target dictionary
        source: Source dictionary
    """
    for key, value in source.items():
        if key in target and isinstance(target[key], dict) and isinstance(value, dict):
            _deep_merge(target[key], value)
        else:
            target[key] = value


def flatten_dict(d: Dict[str, Any], 
                parent_key: str = '', 
                sep: str = '.') -> Dict[str, Any]:
    """
    Flatten nested dictionary.
    
    Args:
        d: Dictionary to flatten
        parent_key: Parent key prefix
        sep: Separator for keys
        
    Returns:
        Flattened dictionary
    """
    items = []
    
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    
    return dict(items)


def unflatten_dict(d: Dict[str, Any], sep: str = '.') -> Dict[str, Any]:
    """
    Unflatten dictionary with dot notation keys.
    
    Args:
        d: Dictionary to unflatten
        sep: Separator in keys
        
    Returns:
        Unflattened dictionary
    """
    result = {}
    
    for key, value in d.items():
        parts = key.split(sep)
        current = result
        
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        
        current[parts[-1]] = value
    
    return result


def get_timestamp(format: str = '%Y-%m-%d %H:%M:%S') -> str:
    """
    Get current timestamp as string.
    
    Args:
        format: Timestamp format
        
    Returns:
        Formatted timestamp
    """
    return datetime.now().strftime(format)


def parse_timestamp(timestamp: str, format: str = '%Y-%m-%d %H:%M:%S') -> datetime:
    """
    Parse timestamp string to datetime.
    
    Args:
        timestamp: Timestamp string
        format: Timestamp format
        
    Returns:
        Datetime object
    """
    return datetime.strptime(timestamp, format)


def calculate_hash(text: str, algorithm: str = 'md5') -> str:
    """
    Calculate hash of text.
    
    Args:
        text: Text to hash
        algorithm: Hash algorithm ('md5', 'sha1', 'sha256')
        
    Returns:
        Hash string
    """
    if algorithm == 'md5':
        return hashlib.md5(text.encode()).hexdigest()
    elif algorithm == 'sha1':
        return hashlib.sha1(text.encode()).hexdigest()
    elif algorithm == 'sha256':
        return hashlib.sha256(text.encode()).hexdigest()
    else:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}")


def retry_on_failure(func: Callable, 
                    max_attempts: int = 3,
                    delay: float = 1.0,
                    backoff: float = 2.0,
                    exceptions: tuple = (Exception,)) -> Any:
    """
    Retry function on failure.
    
    Args:
        func: Function to retry
        max_attempts: Maximum number of attempts
        delay: Initial delay between attempts
        backoff: Backoff multiplier
        exceptions: Exceptions to catch
        
    Returns:
        Function result
        
    Raises:
        Last exception if all attempts fail
    """
    import time
    
    last_exception = None
    current_delay = delay
    
    for attempt in range(max_attempts):
        try:
            return func()
        except exceptions as e:
            last_exception = e
            if attempt < max_attempts - 1:
                time.sleep(current_delay)
                current_delay *= backoff
    
    raise last_exception


def batch_process(items: List[Any], 
                 batch_size: int,
                 processor: Callable[[List[Any]], Any]) -> List[Any]:
    """
    Process items in batches.
    
    Args:
        items: Items to process
        batch_size: Size of each batch
        processor: Function to process each batch
        
    Returns:
        List of results
    """
    results = []
    
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        result = processor(batch)
        results.append(result)
    
    return results


def filter_dict(d: Dict[str, Any], 
               keys: Optional[List[str]] = None,
               exclude_keys: Optional[List[str]] = None,
               predicate: Optional[Callable[[str, Any], bool]] = None) -> Dict[str, Any]:
    """
    Filter dictionary by keys or predicate.
    
    Args:
        d: Dictionary to filter
        keys: Keys to include
        exclude_keys: Keys to exclude
        predicate: Custom filter function
        
    Returns:
        Filtered dictionary
    """
    if keys is not None:
        return {k: v for k, v in d.items() if k in keys}
    
    if exclude_keys is not None:
        return {k: v for k, v in d.items() if k not in exclude_keys}
    
    if predicate is not None:
        return {k: v for k, v in d.items() if predicate(k, v)}
    
    return d


def safe_get(d: Dict[str, Any], 
            key: str, 
            default: Any = None,
            sep: str = '.') -> Any:
    """
    Safely get nested dictionary value.
    
    Args:
        d: Dictionary
        key: Key (supports dot notation)
        default: Default value if not found
        sep: Separator for nested keys
        
    Returns:
        Value or default
    """
    keys = key.split(sep)
    value = d
    
    for k in keys:
        if isinstance(value, dict) and k in value:
            value = value[k]
        else:
            return default
    
    return value


def safe_set(d: Dict[str, Any], 
            key: str, 
            value: Any,
            sep: str = '.') -> None:
    """
    Safely set nested dictionary value.
    
    Args:
        d: Dictionary
        key: Key (supports dot notation)
        value: Value to set
        sep: Separator for nested keys
    """
    keys = key.split(sep)
    current = d
    
    for k in keys[:-1]:
        if k not in current:
            current[k] = {}
        current = current[k]
    
    current[keys[-1]] = value


def is_empty(value: Any) -> bool:
    """
    Check if value is empty.
    
    Args:
        value: Value to check
        
    Returns:
        True if empty, False otherwise
    """
    if value is None:
        return True
    
    if isinstance(value, (str, list, dict, tuple, set)):
        return len(value) == 0
    
    return False


def coalesce(*values: Any) -> Any:
    """
    Return first non-None value.
    
    Args:
        values: Values to check
        
    Returns:
        First non-None value or None
    """
    for value in values:
        if value is not None:
            return value
    return None
