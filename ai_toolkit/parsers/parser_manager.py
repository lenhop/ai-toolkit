"""
Parser manager for centralized output parsing management.
"""

from typing import Dict, List, Any, Optional, Type, Union
from pydantic import BaseModel
import logging

from .output_parser import BaseOutputParser, create_parser
from .error_handler import OutputErrorHandler


logger = logging.getLogger(__name__)


class ParserManager:
    """
    Central manager for output parsers.
    
    Handles parser creation, caching, and parsing operations with error recovery.
    """
    
    def __init__(self):
        """Initialize parser manager."""
        self._parsers: Dict[str, BaseOutputParser] = {}
        self._error_handler = OutputErrorHandler()
    
    def create_parser(self, name: str, parser_type: str, **kwargs) -> BaseOutputParser:
        """
        Create and register a parser.
        
        Args:
            name: Parser name for caching
            parser_type: Type of parser ('str', 'json', 'pydantic', 'list', 'regex')
            **kwargs: Parser-specific configuration
            
        Returns:
            Created parser instance
        """
        parser = create_parser(parser_type, **kwargs)
        self._parsers[name] = parser
        
        logger.info(f"Created parser '{name}' of type '{parser_type}'")
        return parser
    
    def get_parser(self, name: str) -> Optional[BaseOutputParser]:
        """
        Get parser by name.
        
        Args:
            name: Parser name
            
        Returns:
            Parser instance or None if not found
        """
        return self._parsers.get(name)
    
    def parse(self, parser_name: str, text: str, retry_on_error: bool = True) -> Any:
        """
        Parse text using named parser with error handling.
        
        Args:
            parser_name: Name of parser to use
            text: Text to parse
            retry_on_error: Whether to retry with error recovery
            
        Returns:
            Parsed output
            
        Raises:
            ValueError: If parser not found or parsing fails
        """
        parser = self.get_parser(parser_name)
        if not parser:
            raise ValueError(f"Parser '{parser_name}' not found")
        
        try:
            return parser.parse(text)
        except Exception as e:
            if retry_on_error:
                logger.warning(f"Parser '{parser_name}' failed, attempting error recovery: {e}")
                return self._parse_with_recovery(parser, text, str(e))
            else:
                raise ValueError(f"Parsing failed with parser '{parser_name}': {e}")
    
    def _parse_with_recovery(self, parser: BaseOutputParser, text: str, error_msg: str) -> Any:
        """Parse with error recovery."""
        try:
            # Try error handler recovery
            recovered_text = self._error_handler.handle_error(text, error_msg, parser)
            return parser.parse(recovered_text)
        except Exception as recovery_error:
            logger.error(f"Error recovery also failed: {recovery_error}")
            # Re-raise the original error, not the recovery error
            raise ValueError(f"Parsing failed even after error recovery. Original error: {error_msg}")
    
    def parse_with_fallback(self, parser_names: List[str], text: str) -> Any:
        """
        Parse text with fallback parsers.
        
        Args:
            parser_names: List of parser names to try in order
            text: Text to parse
            
        Returns:
            Parsed output from first successful parser
            
        Raises:
            ValueError: If all parsers fail
        """
        errors = []
        
        for parser_name in parser_names:
            try:
                return self.parse(parser_name, text, retry_on_error=True)
            except Exception as e:
                errors.append(f"{parser_name}: {e}")
                continue
        
        raise ValueError(f"All parsers failed. Errors: {'; '.join(errors)}")
    
    def validate_output(self, parser_name: str, output: Any) -> bool:
        """
        Validate parsed output.
        
        Args:
            parser_name: Name of parser used
            output: Parsed output to validate
            
        Returns:
            True if valid
            
        Raises:
            ValueError: If parser not found or validation fails
        """
        parser = self.get_parser(parser_name)
        if not parser:
            raise ValueError(f"Parser '{parser_name}' not found")
        
        return parser.validate_output(output)
    
    def get_format_instructions(self, parser_name: str) -> str:
        """
        Get format instructions for a parser.
        
        Args:
            parser_name: Name of parser
            
        Returns:
            Format instructions string
            
        Raises:
            ValueError: If parser not found
        """
        parser = self.get_parser(parser_name)
        if not parser:
            raise ValueError(f"Parser '{parser_name}' not found")
        
        return parser.get_format_instructions()
    
    def list_parsers(self) -> List[Dict[str, Any]]:
        """
        List all registered parsers.
        
        Returns:
            List of parser information dictionaries
        """
        parsers_info = []
        
        for name, parser in self._parsers.items():
            info = {
                'name': name,
                'type': parser.__class__.__name__,
                'config': getattr(parser, 'config', {}),
                'format_instructions': parser.get_format_instructions()
            }
            parsers_info.append(info)
        
        return parsers_info
    
    def remove_parser(self, name: str) -> bool:
        """
        Remove parser by name.
        
        Args:
            name: Parser name
            
        Returns:
            True if parser was removed, False if not found
        """
        if name in self._parsers:
            del self._parsers[name]
            logger.info(f"Removed parser '{name}'")
            return True
        return False
    
    def clear_parsers(self) -> None:
        """Clear all parsers."""
        self._parsers.clear()
        logger.info("Cleared all parsers")
    
    def create_json_parser(self, name: str, schema: Optional[Dict[str, Any]] = None, 
                          strict: bool = False) -> BaseOutputParser:
        """
        Convenience method to create JSON parser.
        
        Args:
            name: Parser name
            schema: Optional JSON schema
            strict: Whether to use strict parsing
            
        Returns:
            JSON parser instance
        """
        return self.create_parser(name, 'json', schema=schema, strict=strict)
    
    def create_pydantic_parser(self, name: str, pydantic_object: Type[BaseModel]) -> BaseOutputParser:
        """
        Convenience method to create Pydantic parser.
        
        Args:
            name: Parser name
            pydantic_object: Pydantic model class
            
        Returns:
            Pydantic parser instance
        """
        return self.create_parser(name, 'pydantic', pydantic_object=pydantic_object)
    
    def create_list_parser(self, name: str, item_parser: Optional[BaseOutputParser] = None,
                          separator: str = '\n', numbered: bool = False) -> BaseOutputParser:
        """
        Convenience method to create list parser.
        
        Args:
            name: Parser name
            item_parser: Parser for individual items
            separator: Item separator
            numbered: Whether list is numbered
            
        Returns:
            List parser instance
        """
        return self.create_parser(name, 'list', item_parser=item_parser, 
                                separator=separator, numbered=numbered)
    
    def create_regex_parser(self, name: str, regex_pattern: str, 
                           group_names: Optional[List[str]] = None) -> BaseOutputParser:
        """
        Convenience method to create regex parser.
        
        Args:
            name: Parser name
            regex_pattern: Regex pattern
            group_names: Names for regex groups
            
        Returns:
            Regex parser instance
        """
        return self.create_parser(name, 'regex', regex_pattern=regex_pattern, 
                                group_names=group_names)
    
    def batch_parse(self, parser_name: str, texts: List[str], 
                   continue_on_error: bool = True) -> List[Any]:
        """
        Parse multiple texts with the same parser.
        
        Args:
            parser_name: Name of parser to use
            texts: List of texts to parse
            continue_on_error: Whether to continue if individual parsing fails
            
        Returns:
            List of parsed outputs (may contain None for failed parses)
        """
        results = []
        
        for i, text in enumerate(texts):
            try:
                result = self.parse(parser_name, text, retry_on_error=continue_on_error)
                results.append(result)
            except Exception as e:
                if continue_on_error:
                    logger.warning(f"Failed to parse text {i}: {e}")
                    results.append(None)
                else:
                    raise ValueError(f"Batch parsing failed at index {i}: {e}")
        
        return results
    
    def get_parser_stats(self) -> Dict[str, Any]:
        """
        Get statistics about registered parsers.
        
        Returns:
            Statistics dictionary
        """
        stats = {
            'total_parsers': len(self._parsers),
            'parser_types': {},
            'parsers': list(self._parsers.keys())
        }
        
        # Count parser types
        for parser in self._parsers.values():
            parser_type = parser.__class__.__name__
            stats['parser_types'][parser_type] = stats['parser_types'].get(parser_type, 0) + 1
        
        return stats
    
    def test_parser(self, parser_name: str, test_text: str) -> Dict[str, Any]:
        """
        Test a parser with sample text.
        
        Args:
            parser_name: Name of parser to test
            test_text: Sample text to parse
            
        Returns:
            Test results dictionary
        """
        parser = self.get_parser(parser_name)
        if not parser:
            return {'error': f"Parser '{parser_name}' not found"}
        
        result = {
            'parser_name': parser_name,
            'parser_type': parser.__class__.__name__,
            'input_text': test_text,
            'success': False,
            'output': None,
            'error': None,
            'format_instructions': parser.get_format_instructions()
        }
        
        try:
            output = self.parse(parser_name, test_text)
            result['success'] = True
            result['output'] = output
            result['output_type'] = type(output).__name__
        except Exception as e:
            result['error'] = str(e)
        
        return result
    
    def clone_parser(self, source_name: str, target_name: str, **modifications) -> BaseOutputParser:
        """
        Clone an existing parser with optional modifications.
        
        Args:
            source_name: Name of parser to clone
            target_name: Name for the new parser
            **modifications: Configuration modifications
            
        Returns:
            Cloned parser instance
            
        Raises:
            ValueError: If source parser not found
        """
        source_parser = self.get_parser(source_name)
        if not source_parser:
            raise ValueError(f"Source parser '{source_name}' not found")
        
        # Get source parser configuration
        config = getattr(source_parser, 'config', {}).copy()
        config.update(modifications)
        
        # Determine parser type
        parser_type_map = {
            'StrOutputParser': 'str',
            'JsonOutputParser': 'json',
            'PydanticOutputParser': 'pydantic',
            'ListOutputParser': 'list',
            'RegexOutputParser': 'regex'
        }
        
        parser_type = parser_type_map.get(source_parser.__class__.__name__, 'str')
        
        # Handle special cases for parser-specific attributes
        if hasattr(source_parser, 'pydantic_object'):
            config['pydantic_object'] = source_parser.pydantic_object
        if hasattr(source_parser, 'regex_pattern'):
            config['regex_pattern'] = source_parser.regex_pattern
        if hasattr(source_parser, 'schema'):
            config['schema'] = source_parser.schema
        
        # Create cloned parser
        cloned_parser = self.create_parser(target_name, parser_type, **config)
        
        logger.info(f"Cloned parser '{source_name}' to '{target_name}'")
        return cloned_parser