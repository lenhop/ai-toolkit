"""
Tests for output parser classes.
"""

import pytest
import json
from pydantic import BaseModel, Field
from typing import List

from ai_toolkit.parsers.output_parser import (
    StrOutputParser,
    JsonOutputParser,
    PydanticOutputParser,
    ListOutputParser,
    RegexOutputParser,
    create_parser
)


class TestModel(BaseModel):
    """Test Pydantic model for testing."""
    name: str = Field(..., description="Person's name")
    age: int = Field(..., description="Person's age")
    email: str = Field(..., description="Person's email")


class TestStrOutputParser:
    """Test StrOutputParser class."""
    
    def test_basic_parsing(self):
        """Test basic string parsing."""
        parser = StrOutputParser()
        
        result = parser.parse("Hello, world!")
        assert result == "Hello, world!"
    
    def test_whitespace_stripping(self):
        """Test whitespace stripping."""
        parser = StrOutputParser(strip_whitespace=True)
        
        result = parser.parse("  Hello, world!  \n")
        assert result == "Hello, world!"
    
    def test_empty_line_removal(self):
        """Test empty line removal."""
        parser = StrOutputParser(remove_empty_lines=True)
        
        text = "Line 1\n\nLine 2\n\n\nLine 3"
        result = parser.parse(text)
        assert result == "Line 1\nLine 2\nLine 3"
    
    def test_format_instructions(self):
        """Test format instructions."""
        parser = StrOutputParser()
        instructions = parser.get_format_instructions()
        assert "plain text" in instructions.lower()
    
    def test_to_langchain(self):
        """Test LangChain conversion."""
        parser = StrOutputParser()
        lc_parser = parser.to_langchain()
        assert lc_parser is not None


class TestJsonOutputParser:
    """Test JsonOutputParser class."""
    
    def test_valid_json_parsing(self):
        """Test parsing valid JSON."""
        parser = JsonOutputParser()
        
        json_text = '{"name": "Alice", "age": 30}'
        result = parser.parse(json_text)
        
        assert result == {"name": "Alice", "age": 30}
    
    def test_json_with_schema(self):
        """Test JSON parsing with schema validation."""
        schema = {"name": str, "age": int}
        parser = JsonOutputParser(schema=schema)
        
        json_text = '{"name": "Alice", "age": 30}'
        result = parser.parse(json_text)
        
        assert result == {"name": "Alice", "age": 30}
    
    def test_json_extraction_from_text(self):
        """Test extracting JSON from surrounding text."""
        parser = JsonOutputParser()
        
        text = 'Here is the result: {"name": "Alice", "age": 30} and that\'s it.'
        result = parser.parse(text)
        
        assert result == {"name": "Alice", "age": 30}
    
    def test_json_error_recovery(self):
        """Test JSON error recovery."""
        parser = JsonOutputParser(strict=False)
        
        # Test trailing comma fix
        json_text = '{"name": "Alice", "age": 30,}'
        result = parser.parse(json_text)
        assert result == {"name": "Alice", "age": 30}
    
    def test_invalid_json_strict_mode(self):
        """Test invalid JSON in strict mode."""
        parser = JsonOutputParser(strict=True)
        
        with pytest.raises(ValueError):
            parser.parse('{"name": "Alice", "age": 30,}')  # Trailing comma
    
    def test_format_instructions(self):
        """Test format instructions."""
        parser = JsonOutputParser()
        instructions = parser.get_format_instructions()
        assert "JSON" in instructions
    
    def test_format_instructions_with_schema(self):
        """Test format instructions with schema."""
        schema = {"name": str, "age": int}
        parser = JsonOutputParser(schema=schema)
        instructions = parser.get_format_instructions()
        assert "name" in instructions and "age" in instructions


class TestPydanticOutputParser:
    """Test PydanticOutputParser class."""
    
    def test_valid_pydantic_parsing(self):
        """Test parsing valid Pydantic model."""
        parser = PydanticOutputParser(pydantic_object=TestModel)
        
        json_text = '{"name": "Alice", "age": 30, "email": "alice@example.com"}'
        result = parser.parse(json_text)
        
        assert isinstance(result, TestModel)
        assert result.name == "Alice"
        assert result.age == 30
        assert result.email == "alice@example.com"
    
    def test_pydantic_validation_error(self):
        """Test Pydantic validation error."""
        parser = PydanticOutputParser(pydantic_object=TestModel)
        
        # Missing required field
        json_text = '{"name": "Alice", "age": 30}'
        
        with pytest.raises(ValueError, match="Pydantic validation failed"):
            parser.parse(json_text)
    
    def test_pydantic_type_error(self):
        """Test Pydantic type error."""
        parser = PydanticOutputParser(pydantic_object=TestModel)
        
        # Wrong type for age
        json_text = '{"name": "Alice", "age": "thirty", "email": "alice@example.com"}'
        
        with pytest.raises(ValueError):
            parser.parse(json_text)
    
    def test_format_instructions(self):
        """Test format instructions."""
        parser = PydanticOutputParser(pydantic_object=TestModel)
        instructions = parser.get_format_instructions()
        
        assert "JSON" in instructions
        assert "name" in instructions
        assert "age" in instructions
        assert "email" in instructions
    
    def test_to_langchain(self):
        """Test LangChain conversion."""
        parser = PydanticOutputParser(pydantic_object=TestModel)
        lc_parser = parser.to_langchain()
        assert lc_parser is not None


class TestListOutputParser:
    """Test ListOutputParser class."""
    
    def test_basic_list_parsing(self):
        """Test basic list parsing."""
        parser = ListOutputParser()
        
        text = "Item 1\nItem 2\nItem 3"
        result = parser.parse(text)
        
        assert result == ["Item 1", "Item 2", "Item 3"]
    
    def test_numbered_list_parsing(self):
        """Test numbered list parsing."""
        parser = ListOutputParser(numbered=True)
        
        text = "1. First item\n2. Second item\n3. Third item"
        result = parser.parse(text)
        
        assert result == ["First item", "Second item", "Third item"]
    
    def test_custom_separator(self):
        """Test custom separator."""
        parser = ListOutputParser(separator="|")
        
        text = "Item 1|Item 2|Item 3"
        result = parser.parse(text)
        
        assert result == ["Item 1", "Item 2", "Item 3"]
    
    def test_with_item_parser(self):
        """Test with custom item parser."""
        json_parser = JsonOutputParser()
        parser = ListOutputParser(item_parser=json_parser, separator="\n---\n")
        
        text = '{"name": "Alice"}\n---\n{"name": "Bob"}'
        result = parser.parse(text)
        
        assert len(result) == 2
        assert result[0] == {"name": "Alice"}
        assert result[1] == {"name": "Bob"}
    
    def test_empty_items_filtered(self):
        """Test that empty items are filtered out."""
        parser = ListOutputParser()
        
        text = "Item 1\n\nItem 2\n\n\nItem 3"
        result = parser.parse(text)
        
        assert result == ["Item 1", "Item 2", "Item 3"]
    
    def test_format_instructions(self):
        """Test format instructions."""
        parser = ListOutputParser()
        instructions = parser.get_format_instructions()
        assert "list" in instructions.lower()


class TestRegexOutputParser:
    """Test RegexOutputParser class."""
    
    def test_simple_regex_parsing(self):
        """Test simple regex parsing."""
        parser = RegexOutputParser(regex_pattern=r"Name: (\w+)")
        
        text = "Hello, my Name: Alice and I'm happy."
        result = parser.parse(text)
        
        assert result == "Alice"
    
    def test_multiple_groups(self):
        """Test multiple regex groups."""
        parser = RegexOutputParser(
            regex_pattern=r"Name: (\w+), Age: (\d+)",
            group_names=["name", "age"]
        )
        
        text = "Person details - Name: Alice, Age: 30"
        result = parser.parse(text)
        
        assert result == {"name": "Alice", "age": "30"}
    
    def test_multiple_groups_without_names(self):
        """Test multiple groups without names."""
        parser = RegexOutputParser(regex_pattern=r"(\w+): (\d+)")
        
        text = "Score: 95"
        result = parser.parse(text)
        
        assert result == ["Score", "95"]
    
    def test_no_match_error(self):
        """Test error when regex doesn't match."""
        parser = RegexOutputParser(regex_pattern=r"Name: (\w+)")
        
        text = "Hello, world!"
        
        with pytest.raises(ValueError, match="did not match"):
            parser.parse(text)
    
    def test_full_match_no_groups(self):
        """Test full match when no groups."""
        parser = RegexOutputParser(regex_pattern=r"Hello, \w+!")
        
        text = "Hello, Alice!"
        result = parser.parse(text)
        
        assert result == "Hello, Alice!"


class TestCreateParser:
    """Test parser factory function."""
    
    def test_create_str_parser(self):
        """Test creating string parser."""
        parser = create_parser('str')
        assert isinstance(parser, StrOutputParser)
    
    def test_create_json_parser(self):
        """Test creating JSON parser."""
        parser = create_parser('json')
        assert isinstance(parser, JsonOutputParser)
    
    def test_create_pydantic_parser(self):
        """Test creating Pydantic parser."""
        parser = create_parser('pydantic', pydantic_object=TestModel)
        assert isinstance(parser, PydanticOutputParser)
    
    def test_create_list_parser(self):
        """Test creating list parser."""
        parser = create_parser('list')
        assert isinstance(parser, ListOutputParser)
    
    def test_create_regex_parser(self):
        """Test creating regex parser."""
        parser = create_parser('regex', regex_pattern=r"(\w+)")
        assert isinstance(parser, RegexOutputParser)
    
    def test_invalid_parser_type(self):
        """Test error with invalid parser type."""
        with pytest.raises(ValueError, match="Unsupported parser type"):
            create_parser('invalid_type')
    
    def test_parser_with_config(self):
        """Test creating parser with configuration."""
        parser = create_parser('str', strip_whitespace=True, remove_empty_lines=True)
        assert isinstance(parser, StrOutputParser)
        assert parser.strip_whitespace is True
        assert parser.remove_empty_lines is True