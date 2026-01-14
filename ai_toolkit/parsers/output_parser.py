"""
Output parser classes for different types of structured outputs.

This module provides various parser classes for converting AI model outputs
into structured data formats including JSON, Pydantic models, lists, and more.

Classes:
    BaseOutputParser: Abstract base class for output parsers
        - Defines interface for all parsers
        - Provides LangChain compatibility
        
        Methods:
            __init__(**kwargs): Initialize parser with configuration
            parse(text): Parse output text into structured data (abstract)
            get_format_instructions(): Get format instructions (abstract)
            to_langchain(): Convert to LangChain output parser
    
    StrOutputParser: String output parser
        - Parses output as plain string
        - Optionally strips whitespace
        
        Methods:
            parse(text): Return text as string
            get_format_instructions(): Get string format instructions
            to_langchain(): Convert to LangChain StrOutputParser
    
    JsonOutputParser: JSON output parser
        - Parses JSON formatted output
        - Validates JSON structure
        - Supports schema validation
        
        Methods:
            parse(text): Parse JSON text to dictionary
            get_format_instructions(): Get JSON format instructions
            to_langchain(): Convert to LangChain JsonOutputParser
    
    PydanticOutputParser: Pydantic model output parser
        - Parses output into Pydantic models
        - Validates against model schema
        - Provides detailed error messages
        
        Methods:
            __init__(pydantic_model): Initialize with Pydantic model class
            parse(text): Parse text into Pydantic model instance
            get_format_instructions(): Get Pydantic format instructions
            to_langchain(): Convert to LangChain PydanticOutputParser
    
    ListOutputParser: List output parser
        - Parses comma or newline separated lists
        - Supports custom separators
        - Optionally strips whitespace
        
        Methods:
            __init__(separator, strip): Initialize with separator
            parse(text): Parse text into list
            get_format_instructions(): Get list format instructions
    
    RegexOutputParser: Regex-based output parser
        - Extracts data using regular expressions
        - Supports named capture groups
        - Returns matched groups as dictionary
        
        Methods:
            __init__(pattern, group_names): Initialize with regex pattern
            parse(text): Parse text using regex
            get_format_instructions(): Get regex format instructions
"""

import json
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type, Union, get_origin, get_args
from pydantic import BaseModel, ValidationError
from langchain_core.output_parsers import BaseOutputParser as LangChainBaseOutputParser
from langchain_core.output_parsers import PydanticOutputParser as LangChainPydanticOutputParser
from langchain_core.output_parsers import JsonOutputParser as LangChainJsonOutputParser
from langchain_core.output_parsers import StrOutputParser as LangChainStrOutputParser
import logging

logger = logging.getLogger(__name__)


class BaseOutputParser(ABC):
    """Abstract base class for output parsers."""
    
    def __init__(self, **kwargs):
        """Initialize parser with optional configuration."""
        self.config = kwargs
    
    @abstractmethod
    def parse(self, text: str) -> Any:
        """
        Parse the output text into structured data.
        
        Args:
            text: Raw output text to parse
            
        Returns:
            Parsed structured data
            
        Raises:
            ValueError: If parsing fails
        """
        pass
    
    @abstractmethod
    def get_format_instructions(self) -> str:
        """
        Get instructions for the expected output format.
        
        Returns:
            Format instructions string
        """
        pass
    
    def to_langchain(self) -> LangChainBaseOutputParser:
        """
        Convert to LangChain output parser.
        
        Returns:
            LangChain compatible parser
        """
        raise NotImplementedError("Subclasses must implement to_langchain method")
    
    def validate_output(self, output: Any) -> bool:
        """
        Validate parsed output.
        
        Args:
            output: Parsed output to validate
            
        Returns:
            True if valid
            
        Raises:
            ValueError: If output is invalid
        """
        return True


class StrOutputParser(BaseOutputParser):
    """String output parser that returns text as-is with optional cleaning."""
    
    def __init__(self, strip_whitespace: bool = True, remove_empty_lines: bool = False, **kwargs):
        """
        Initialize string parser.
        
        Args:
            strip_whitespace: Whether to strip leading/trailing whitespace
            remove_empty_lines: Whether to remove empty lines
        """
        super().__init__(**kwargs)
        self.strip_whitespace = strip_whitespace
        self.remove_empty_lines = remove_empty_lines
    
    def parse(self, text: str) -> str:
        """Parse text with optional cleaning."""
        if not isinstance(text, str):
            text = str(text)
        
        if self.strip_whitespace:
            text = text.strip()
        
        if self.remove_empty_lines:
            lines = [line for line in text.split('\n') if line.strip()]
            text = '\n'.join(lines)
        
        return text
    
    def get_format_instructions(self) -> str:
        """Get format instructions for string output."""
        return "Provide your response as plain text."
    
    def to_langchain(self) -> LangChainStrOutputParser:
        """Convert to LangChain StrOutputParser."""
        return LangChainStrOutputParser()


class JsonOutputParser(BaseOutputParser):
    """JSON output parser with error recovery capabilities."""
    
    def __init__(self, schema: Optional[Dict[str, Any]] = None, strict: bool = False, **kwargs):
        """
        Initialize JSON parser.
        
        Args:
            schema: Optional JSON schema for validation
            strict: Whether to use strict JSON parsing
        """
        super().__init__(**kwargs)
        self.schema = schema
        self.strict = strict
    
    def parse(self, text: str) -> Dict[str, Any]:
        """
        Parse JSON from text with error recovery.
        
        Args:
            text: Text containing JSON
            
        Returns:
            Parsed JSON object
            
        Raises:
            ValueError: If JSON cannot be parsed
        """
        if not isinstance(text, str):
            text = str(text)
        
        # Try direct JSON parsing first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        
        # Try to extract JSON from text
        json_text = self._extract_json(text)
        if json_text:
            try:
                result = json.loads(json_text)
                if self.schema:
                    self._validate_schema(result)
                return result
            except json.JSONDecodeError:
                pass
        
        # Try to fix common JSON issues
        if not self.strict:
            fixed_text = self._fix_json_issues(text)
            try:
                result = json.loads(fixed_text)
                if self.schema:
                    self._validate_schema(result)
                return result
            except json.JSONDecodeError:
                pass
        
        raise ValueError(f"Could not parse JSON from text: {text[:100]}...")
    
    def _extract_json(self, text: str) -> Optional[str]:
        """Extract JSON from text using regex patterns."""
        # Look for JSON objects
        json_patterns = [
            r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}',  # Simple nested objects
            r'\{.*\}',  # Any content between braces
            r'\[.*\]',  # Arrays
        ]
        
        for pattern in json_patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            for match in matches:
                try:
                    json.loads(match)
                    return match
                except json.JSONDecodeError:
                    continue
        
        return None
    
    def _fix_json_issues(self, text: str) -> str:
        """Fix common JSON formatting issues."""
        # Remove leading/trailing whitespace
        text = text.strip()
        
        # Fix trailing commas
        text = re.sub(r',(\s*[}\]])', r'\1', text)
        
        # Fix missing quotes around keys
        text = re.sub(r'(\w+):', r'"\1":', text)
        
        # Fix single quotes to double quotes
        text = text.replace("'", '"')
        
        # Fix boolean values
        text = re.sub(r'\bTrue\b', 'true', text)
        text = re.sub(r'\bFalse\b', 'false', text)
        text = re.sub(r'\bNone\b', 'null', text)
        
        return text
    
    def _validate_schema(self, data: Dict[str, Any]) -> None:
        """Validate data against schema."""
        if not self.schema:
            return
        
        # Simple schema validation (can be extended)
        for key, expected_type in self.schema.items():
            if key in data:
                if not isinstance(data[key], expected_type):
                    raise ValueError(f"Key '{key}' should be of type {expected_type.__name__}")
    
    def get_format_instructions(self) -> str:
        """Get format instructions for JSON output."""
        if self.schema:
            schema_desc = ", ".join([f'"{k}": {v.__name__}' for k, v in self.schema.items()])
            return f"Provide your response as a JSON object with the following structure: {{{schema_desc}}}"
        return "Provide your response as a valid JSON object."
    
    def to_langchain(self) -> LangChainJsonOutputParser:
        """Convert to LangChain JsonOutputParser."""
        return LangChainJsonOutputParser()


class PydanticOutputParser(BaseOutputParser):
    """Pydantic model output parser with validation."""
    
    def __init__(self, pydantic_object: Type[BaseModel], **kwargs):
        """
        Initialize Pydantic parser.
        
        Args:
            pydantic_object: Pydantic model class to parse into
        """
        super().__init__(**kwargs)
        self.pydantic_object = pydantic_object
        self._json_parser = JsonOutputParser(strict=False)
    
    def parse(self, text: str) -> BaseModel:
        """
        Parse text into Pydantic model.
        
        Args:
            text: Text to parse
            
        Returns:
            Pydantic model instance
            
        Raises:
            ValueError: If parsing or validation fails
        """
        try:
            # First parse as JSON
            json_data = self._json_parser.parse(text)
            
            # Then validate with Pydantic
            return self.pydantic_object(**json_data)
            
        except ValidationError as e:
            raise ValueError(f"Pydantic validation failed: {e}")
        except Exception as e:
            raise ValueError(f"Failed to parse into {self.pydantic_object.__name__}: {e}")
    
    def get_format_instructions(self) -> str:
        """Get format instructions for Pydantic model."""
        schema = self.pydantic_object.model_json_schema()
        
        # Extract field information
        fields = []
        if 'properties' in schema:
            for field_name, field_info in schema['properties'].items():
                field_type = field_info.get('type', 'any')
                field_desc = field_info.get('description', '')
                fields.append(f'"{field_name}": {field_type}' + (f' // {field_desc}' if field_desc else ''))
        
        fields_str = ', '.join(fields)
        return f"Provide your response as a JSON object matching this schema: {{{fields_str}}}"
    
    def to_langchain(self) -> LangChainPydanticOutputParser:
        """Convert to LangChain PydanticOutputParser."""
        return LangChainPydanticOutputParser(pydantic_object=self.pydantic_object)


class ListOutputParser(BaseOutputParser):
    """Parser for list outputs with various formats."""
    
    def __init__(self, item_parser: Optional[BaseOutputParser] = None, 
                 separator: str = '\n', numbered: bool = False, **kwargs):
        """
        Initialize list parser.
        
        Args:
            item_parser: Parser for individual list items
            separator: Separator between list items
            numbered: Whether to expect numbered lists
        """
        super().__init__(**kwargs)
        self.item_parser = item_parser or StrOutputParser()
        self.separator = separator
        self.numbered = numbered
    
    def parse(self, text: str) -> List[Any]:
        """
        Parse text into list.
        
        Args:
            text: Text containing list
            
        Returns:
            Parsed list
        """
        if not isinstance(text, str):
            text = str(text)
        
        # Split by separator
        items = text.split(self.separator)
        
        # Clean up items
        cleaned_items = []
        for item in items:
            item = item.strip()
            if not item:
                continue
            
            # Remove numbering if expected
            if self.numbered:
                item = re.sub(r'^\d+\.\s*', '', item)
                item = re.sub(r'^[-*]\s*', '', item)
            
            # Parse individual item
            try:
                parsed_item = self.item_parser.parse(item)
                cleaned_items.append(parsed_item)
            except Exception as e:
                logger.warning(f"Failed to parse list item '{item}': {e}")
                cleaned_items.append(item)  # Fallback to raw text
        
        return cleaned_items
    
    def get_format_instructions(self) -> str:
        """Get format instructions for list output."""
        if self.numbered:
            return f"Provide your response as a numbered list, with each item on a new line (separated by '{self.separator}')."
        return f"Provide your response as a list, with each item on a new line (separated by '{self.separator}')."
    
    def to_langchain(self) -> LangChainBaseOutputParser:
        """Convert to LangChain parser (custom implementation)."""
        # Create a custom LangChain parser for lists
        class LangChainListParser(LangChainBaseOutputParser):
            def __init__(self, list_parser):
                self.list_parser = list_parser
            
            def parse(self, text: str) -> List[Any]:
                return self.list_parser.parse(text)
            
            def get_format_instructions(self) -> str:
                return self.list_parser.get_format_instructions()
        
        return LangChainListParser(self)


class RegexOutputParser(BaseOutputParser):
    """Regex-based output parser for custom patterns."""
    
    def __init__(self, regex_pattern: str, group_names: Optional[List[str]] = None, **kwargs):
        """
        Initialize regex parser.
        
        Args:
            regex_pattern: Regex pattern to match
            group_names: Names for regex groups
        """
        super().__init__(**kwargs)
        self.regex_pattern = regex_pattern
        self.group_names = group_names or []
        self.compiled_pattern = re.compile(regex_pattern, re.DOTALL | re.MULTILINE)
    
    def parse(self, text: str) -> Union[str, Dict[str, str], List[str]]:
        """
        Parse text using regex pattern.
        
        Args:
            text: Text to parse
            
        Returns:
            Matched content (string, dict, or list depending on pattern)
            
        Raises:
            ValueError: If pattern doesn't match
        """
        if not isinstance(text, str):
            text = str(text)
        
        match = self.compiled_pattern.search(text)
        if not match:
            raise ValueError(f"Regex pattern '{self.regex_pattern}' did not match text")
        
        groups = match.groups()
        
        if not groups:
            # No groups, return full match
            return match.group(0)
        elif len(groups) == 1:
            # Single group, return as string
            return groups[0]
        else:
            # Multiple groups
            if self.group_names and len(self.group_names) == len(groups):
                # Return as dictionary with named groups
                return dict(zip(self.group_names, groups))
            else:
                # Return as list
                return list(groups)
    
    def get_format_instructions(self) -> str:
        """Get format instructions for regex pattern."""
        return f"Provide your response in a format that matches this pattern: {self.regex_pattern}"
    
    def to_langchain(self) -> LangChainBaseOutputParser:
        """Convert to LangChain parser (custom implementation)."""
        class LangChainRegexParser(LangChainBaseOutputParser):
            def __init__(self, regex_parser):
                self.regex_parser = regex_parser
            
            def parse(self, text: str) -> Any:
                return self.regex_parser.parse(text)
            
            def get_format_instructions(self) -> str:
                return self.regex_parser.get_format_instructions()
        
        return LangChainRegexParser(self)


def create_parser(parser_type: str, **kwargs) -> BaseOutputParser:
    """
    Factory function to create output parsers.
    
    Args:
        parser_type: Type of parser ('str', 'json', 'pydantic', 'list', 'regex')
        **kwargs: Parser-specific configuration
        
    Returns:
        Output parser instance
        
    Raises:
        ValueError: If parser type is not supported
    """
    parser_classes = {
        'str': StrOutputParser,
        'string': StrOutputParser,
        'json': JsonOutputParser,
        'pydantic': PydanticOutputParser,
        'list': ListOutputParser,
        'regex': RegexOutputParser,
    }
    
    if parser_type.lower() not in parser_classes:
        raise ValueError(f"Unsupported parser type: {parser_type}. "
                        f"Supported types: {list(parser_classes.keys())}")
    
    parser_class = parser_classes[parser_type.lower()]
    return parser_class(**kwargs)