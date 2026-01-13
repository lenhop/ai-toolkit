"""
Tests for ParserManager class.
"""

import pytest
from pydantic import BaseModel, Field
from unittest.mock import patch, MagicMock

from ai_toolkit.parsers.parser_manager import ParserManager
from ai_toolkit.parsers.output_parser import StrOutputParser, JsonOutputParser


class TestPersonModel(BaseModel):
    """Test Pydantic model."""
    name: str = Field(..., description="Person's name")
    age: int = Field(..., description="Person's age")


class TestParserManager:
    """Test ParserManager class."""
    
    def test_init(self):
        """Test ParserManager initialization."""
        manager = ParserManager()
        assert isinstance(manager, ParserManager)
        assert len(manager._parsers) == 0
    
    def test_create_parser(self):
        """Test creating and registering a parser."""
        manager = ParserManager()
        
        parser = manager.create_parser('test_str', 'str')
        
        assert isinstance(parser, StrOutputParser)
        assert 'test_str' in manager._parsers
        assert manager.get_parser('test_str') is parser
    
    def test_create_parser_with_config(self):
        """Test creating parser with configuration."""
        manager = ParserManager()
        
        parser = manager.create_parser('test_str', 'str', strip_whitespace=True)
        
        assert isinstance(parser, StrOutputParser)
        assert parser.strip_whitespace is True
    
    def test_get_parser(self):
        """Test getting parser by name."""
        manager = ParserManager()
        
        # Create parser
        created_parser = manager.create_parser('test_parser', 'str')
        
        # Get parser
        retrieved_parser = manager.get_parser('test_parser')
        assert retrieved_parser is created_parser
        
        # Get non-existent parser
        assert manager.get_parser('nonexistent') is None
    
    def test_parse_success(self):
        """Test successful parsing."""
        manager = ParserManager()
        
        manager.create_parser('json_parser', 'json')
        
        result = manager.parse('json_parser', '{"name": "Alice", "age": 30}')
        assert result == {"name": "Alice", "age": 30}
    
    def test_parse_parser_not_found(self):
        """Test parsing with non-existent parser."""
        manager = ParserManager()
        
        with pytest.raises(ValueError, match="Parser 'nonexistent' not found"):
            manager.parse('nonexistent', 'some text')
    
    def test_parse_with_error_recovery(self):
        """Test parsing with error recovery."""
        manager = ParserManager()
        
        manager.create_parser('json_parser', 'json')
        
        # This should trigger error recovery
        with patch.object(manager._error_handler, 'handle_error', return_value='{"fixed": true}'):
            # First attempt fails, recovery should work
            with patch.object(manager._parsers['json_parser'], 'parse', side_effect=[ValueError("Parse error"), {"fixed": True}]):
                result = manager.parse('json_parser', 'malformed json', retry_on_error=True)
                assert result == {"fixed": True}
    
    def test_parse_without_retry(self):
        """Test parsing without retry on error."""
        manager = ParserManager()
        
        manager.create_parser('json_parser', 'json')
        
        with pytest.raises(ValueError, match="Parsing failed"):
            manager.parse('json_parser', 'invalid json', retry_on_error=False)
    
    def test_parse_with_fallback(self):
        """Test parsing with fallback parsers."""
        manager = ParserManager()
        
        manager.create_parser('json_parser', 'json')
        manager.create_parser('str_parser', 'str')
        
        # JSON parser should fail, string parser should succeed
        result = manager.parse_with_fallback(['json_parser', 'str_parser'], 'plain text')
        assert result == 'plain text'
    
    def test_parse_with_fallback_all_fail(self):
        """Test fallback parsing when all parsers fail."""
        manager = ParserManager()
        
        manager.create_parser('json_parser', 'json')
        
        with pytest.raises(ValueError, match="All parsers failed"):
            manager.parse_with_fallback(['json_parser'], 'invalid json')
    
    def test_validate_output(self):
        """Test output validation."""
        manager = ParserManager()
        
        parser = manager.create_parser('str_parser', 'str')
        
        # Mock validation
        with patch.object(parser, 'validate_output', return_value=True):
            result = manager.validate_output('str_parser', 'test output')
            assert result is True
    
    def test_validate_output_parser_not_found(self):
        """Test validation with non-existent parser."""
        manager = ParserManager()
        
        with pytest.raises(ValueError, match="Parser 'nonexistent' not found"):
            manager.validate_output('nonexistent', 'output')
    
    def test_get_format_instructions(self):
        """Test getting format instructions."""
        manager = ParserManager()
        
        manager.create_parser('str_parser', 'str')
        
        instructions = manager.get_format_instructions('str_parser')
        assert isinstance(instructions, str)
        assert len(instructions) > 0
    
    def test_get_format_instructions_parser_not_found(self):
        """Test format instructions with non-existent parser."""
        manager = ParserManager()
        
        with pytest.raises(ValueError, match="Parser 'nonexistent' not found"):
            manager.get_format_instructions('nonexistent')
    
    def test_list_parsers(self):
        """Test listing parsers."""
        manager = ParserManager()
        
        manager.create_parser('str_parser', 'str')
        manager.create_parser('json_parser', 'json')
        
        parsers = manager.list_parsers()
        
        assert len(parsers) == 2
        parser_names = [p['name'] for p in parsers]
        assert 'str_parser' in parser_names
        assert 'json_parser' in parser_names
    
    def test_remove_parser(self):
        """Test removing parser."""
        manager = ParserManager()
        
        manager.create_parser('test_parser', 'str')
        assert manager.get_parser('test_parser') is not None
        
        # Remove parser
        removed = manager.remove_parser('test_parser')
        assert removed is True
        assert manager.get_parser('test_parser') is None
        
        # Try to remove non-existent parser
        removed = manager.remove_parser('nonexistent')
        assert removed is False
    
    def test_clear_parsers(self):
        """Test clearing all parsers."""
        manager = ParserManager()
        
        manager.create_parser('parser1', 'str')
        manager.create_parser('parser2', 'json')
        
        assert len(manager._parsers) == 2
        
        manager.clear_parsers()
        assert len(manager._parsers) == 0
    
    def test_convenience_methods(self):
        """Test convenience methods for creating parsers."""
        manager = ParserManager()
        
        # Test JSON parser creation
        json_parser = manager.create_json_parser('json_test', schema={'name': str})
        assert isinstance(json_parser, JsonOutputParser)
        
        # Test Pydantic parser creation
        pydantic_parser = manager.create_pydantic_parser('pydantic_test', TestPersonModel)
        assert 'pydantic_test' in manager._parsers
        
        # Test list parser creation
        list_parser = manager.create_list_parser('list_test', numbered=True)
        assert 'list_test' in manager._parsers
        
        # Test regex parser creation
        regex_parser = manager.create_regex_parser('regex_test', r'(\w+): (\d+)')
        assert 'regex_test' in manager._parsers
    
    def test_batch_parse(self):
        """Test batch parsing."""
        manager = ParserManager()
        
        manager.create_parser('str_parser', 'str')
        
        texts = ['text1', 'text2', 'text3']
        results = manager.batch_parse('str_parser', texts)
        
        assert len(results) == 3
        assert results == ['text1', 'text2', 'text3']
    
    def test_batch_parse_with_errors(self):
        """Test batch parsing with some errors."""
        manager = ParserManager()
        
        manager.create_parser('json_parser', 'json')
        
        texts = ['{"valid": "json"}', 'invalid json', '{"another": "valid"}']
        results = manager.batch_parse('json_parser', texts, continue_on_error=True)
        
        assert len(results) == 3
        assert results[0] == {"valid": "json"}
        assert results[1] is None  # Failed parse
        assert results[2] == {"another": "valid"}
    
    def test_batch_parse_stop_on_error(self):
        """Test batch parsing that stops on error."""
        manager = ParserManager()
        
        manager.create_parser('json_parser', 'json')
        
        texts = ['{"valid": "json"}', 'invalid json', '{"another": "valid"}']
        
        with pytest.raises(ValueError, match="Batch parsing failed"):
            manager.batch_parse('json_parser', texts, continue_on_error=False)
    
    def test_get_parser_stats(self):
        """Test getting parser statistics."""
        manager = ParserManager()
        
        manager.create_parser('str1', 'str')
        manager.create_parser('str2', 'str')
        manager.create_parser('json1', 'json')
        
        stats = manager.get_parser_stats()
        
        assert stats['total_parsers'] == 3
        assert 'StrOutputParser' in stats['parser_types']
        assert 'JsonOutputParser' in stats['parser_types']
        assert stats['parser_types']['StrOutputParser'] == 2
        assert stats['parser_types']['JsonOutputParser'] == 1
    
    def test_test_parser(self):
        """Test parser testing functionality."""
        manager = ParserManager()
        
        manager.create_parser('str_parser', 'str')
        
        result = manager.test_parser('str_parser', 'test text')
        
        assert result['parser_name'] == 'str_parser'
        assert result['success'] is True
        assert result['output'] == 'test text'
        assert 'format_instructions' in result
    
    def test_test_parser_with_error(self):
        """Test parser testing with error."""
        manager = ParserManager()
        
        manager.create_parser('json_parser', 'json')
        
        result = manager.test_parser('json_parser', 'invalid json')
        
        assert result['parser_name'] == 'json_parser'
        assert result['success'] is False
        assert result['error'] is not None
    
    def test_test_nonexistent_parser(self):
        """Test testing non-existent parser."""
        manager = ParserManager()
        
        result = manager.test_parser('nonexistent', 'test')
        
        assert 'error' in result
        assert 'not found' in result['error']
    
    def test_clone_parser(self):
        """Test cloning parser."""
        manager = ParserManager()
        
        # Create original parser
        original = manager.create_parser('original', 'str', strip_whitespace=True)
        
        # Clone with modifications
        cloned = manager.clone_parser('original', 'cloned', remove_empty_lines=True)
        
        assert 'cloned' in manager._parsers
        assert cloned is not original
        assert isinstance(cloned, StrOutputParser)
    
    def test_clone_nonexistent_parser(self):
        """Test cloning non-existent parser."""
        manager = ParserManager()
        
        with pytest.raises(ValueError, match="Source parser 'nonexistent' not found"):
            manager.clone_parser('nonexistent', 'cloned')