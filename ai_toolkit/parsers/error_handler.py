"""
Error handler for output parsing with recovery strategies.
"""

import json
import re
from typing import Any, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class OutputErrorHandler:
    """Handler for output parsing errors with recovery strategies."""
    
    def __init__(self):
        """Initialize error handler."""
        self.recovery_strategies = {
            'json': [
                self._fix_json_trailing_commas,
                self._fix_json_quotes,
                self._fix_json_booleans,
                self._extract_json_from_text,
                self._create_minimal_json
            ],
            'general': [
                self._clean_whitespace,
                self._remove_markdown_formatting,
                self._extract_content_blocks
            ]
        }
    
    def handle_error(self, text: str, error_msg: str, parser: Any) -> str:
        """
        Handle parsing error and attempt recovery.
        
        Args:
            text: Original text that failed to parse
            error_msg: Error message from parser
            parser: Parser instance that failed
            
        Returns:
            Recovered text
            
        Raises:
            ValueError: If recovery fails
        """
        parser_type = self._detect_parser_type(parser, error_msg)
        
        # Apply general recovery strategies first
        recovered_text = text
        for strategy in self.recovery_strategies['general']:
            try:
                recovered_text = strategy(recovered_text)
            except Exception as e:
                logger.debug(f"General recovery strategy failed: {e}")
                continue
        
        # Apply parser-specific recovery strategies
        if parser_type in self.recovery_strategies:
            for strategy in self.recovery_strategies[parser_type]:
                try:
                    recovered_text = strategy(recovered_text)
                    # Test if recovery worked by attempting a basic validation
                    if self._validate_recovery(recovered_text, parser_type):
                        logger.info(f"Successfully recovered text using {strategy.__name__}")
                        return recovered_text
                except Exception as e:
                    logger.debug(f"Recovery strategy {strategy.__name__} failed: {e}")
                    continue
        
        # If all strategies fail, return the best attempt
        return recovered_text
    
    def _detect_parser_type(self, parser: Any, error_msg: str) -> str:
        """Detect parser type from parser instance or error message."""
        if hasattr(parser, '__class__'):
            class_name = parser.__class__.__name__.lower()
            if 'json' in class_name:
                return 'json'
            elif 'pydantic' in class_name:
                return 'json'  # Pydantic parsers often use JSON
        
        # Detect from error message
        error_lower = error_msg.lower()
        if any(keyword in error_lower for keyword in ['json', 'decode', 'expecting']):
            return 'json'
        
        return 'general'
    
    def _validate_recovery(self, text: str, parser_type: str) -> bool:
        """Validate if recovery was successful."""
        if parser_type == 'json':
            try:
                json.loads(text)
                return True
            except json.JSONDecodeError:
                return False
        
        # For other types, just check if text is not empty
        return bool(text.strip())
    
    # General recovery strategies
    
    def _clean_whitespace(self, text: str) -> str:
        """Clean excessive whitespace."""
        # Remove leading/trailing whitespace
        text = text.strip()
        
        # Normalize internal whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text
    
    def _remove_markdown_formatting(self, text: str) -> str:
        """Remove markdown formatting that might interfere with parsing."""
        # Remove code block markers
        text = re.sub(r'```\w*\n?', '', text)
        text = re.sub(r'```', '', text)
        
        # Remove inline code markers
        text = re.sub(r'`([^`]+)`', r'\1', text)
        
        # Remove bold/italic markers
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
        text = re.sub(r'\*([^*]+)\*', r'\1', text)
        
        return text
    
    def _extract_content_blocks(self, text: str) -> str:
        """Extract content from structured blocks."""
        # Look for content between common delimiters
        patterns = [
            r'```(?:json|python|text)?\s*\n(.*?)\n```',
            r'<code>(.*?)</code>',
            r'<pre>(.*?)</pre>',
            r'Output:\s*(.*?)(?:\n\n|\Z)',
            r'Result:\s*(.*?)(?:\n\n|\Z)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return text
    
    # JSON-specific recovery strategies
    
    def _fix_json_trailing_commas(self, text: str) -> str:
        """Fix trailing commas in JSON."""
        # Remove trailing commas before closing brackets/braces
        text = re.sub(r',(\s*[}\]])', r'\1', text)
        return text
    
    def _fix_json_quotes(self, text: str) -> str:
        """Fix quote issues in JSON."""
        # Replace single quotes with double quotes
        text = text.replace("'", '"')
        
        # Fix unquoted keys (simple cases)
        text = re.sub(r'(\w+)(\s*:)', r'"\1"\2', text)
        
        # Fix missing quotes around string values (heuristic)
        text = re.sub(r':\s*([a-zA-Z][a-zA-Z0-9_\s]*[a-zA-Z0-9])(?=\s*[,}])', r': "\1"', text)
        
        return text
    
    def _fix_json_booleans(self, text: str) -> str:
        """Fix boolean and null values in JSON."""
        # Fix Python-style booleans
        text = re.sub(r'\bTrue\b', 'true', text)
        text = re.sub(r'\bFalse\b', 'false', text)
        text = re.sub(r'\bNone\b', 'null', text)
        
        return text
    
    def _extract_json_from_text(self, text: str) -> str:
        """Extract JSON object from surrounding text."""
        # Look for JSON-like structures
        patterns = [
            r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}',  # Nested objects
            r'\{.*?\}',  # Simple objects
            r'\[.*?\]',  # Arrays
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            for match in matches:
                try:
                    # Test if it's valid JSON
                    json.loads(match)
                    return match
                except json.JSONDecodeError:
                    continue
        
        return text
    
    def _create_minimal_json(self, text: str) -> str:
        """Create minimal valid JSON from text."""
        # Only create minimal JSON if the text looks like it might have been intended as JSON
        # Check for some JSON-like indicators
        if any(indicator in text.lower() for indicator in ['{', '}', '[', ']', '"', 'true', 'false', 'null']):
            cleaned_text = text.replace('"', '\\"').replace('\n', '\\n')
            return f'{{"content": "{cleaned_text}"}}'
        else:
            # If it doesn't look like JSON at all, don't try to force it
            raise ValueError("Text does not appear to be JSON-like")
    
    # Retry mechanisms
    
    def retry_parse(self, parser: Any, text: str, max_attempts: int = 3) -> Any:
        """
        Retry parsing with progressive error recovery.
        
        Args:
            parser: Parser instance
            text: Text to parse
            max_attempts: Maximum retry attempts
            
        Returns:
            Parsed result
            
        Raises:
            ValueError: If all attempts fail
        """
        last_error = None
        current_text = text
        
        for attempt in range(max_attempts):
            try:
                return parser.parse(current_text)
            except Exception as e:
                last_error = e
                logger.debug(f"Parse attempt {attempt + 1} failed: {e}")
                
                if attempt < max_attempts - 1:
                    # Apply error recovery for next attempt
                    current_text = self.handle_error(current_text, str(e), parser)
        
        raise ValueError(f"Failed to parse after {max_attempts} attempts. Last error: {last_error}")
    
    def fix_json(self, json_text: str) -> str:
        """
        Comprehensive JSON fixing.
        
        Args:
            json_text: Potentially malformed JSON text
            
        Returns:
            Fixed JSON text
        """
        text = json_text
        
        # Apply all JSON recovery strategies
        for strategy in self.recovery_strategies['json']:
            try:
                text = strategy(text)
            except Exception as e:
                logger.debug(f"JSON fix strategy {strategy.__name__} failed: {e}")
                continue
        
        return text
    
    def handle_parse_error(self, error: Exception, text: str, parser_type: str = 'general') -> Dict[str, Any]:
        """
        Handle and analyze parsing errors.
        
        Args:
            error: Exception that occurred
            text: Text that failed to parse
            parser_type: Type of parser that failed
            
        Returns:
            Error analysis dictionary
        """
        error_info = {
            'error_type': type(error).__name__,
            'error_message': str(error),
            'parser_type': parser_type,
            'text_length': len(text),
            'text_preview': text[:200] + '...' if len(text) > 200 else text,
            'suggested_fixes': []
        }
        
        # Analyze error and suggest fixes
        error_msg = str(error).lower()
        
        if 'json' in error_msg or 'decode' in error_msg:
            error_info['suggested_fixes'].extend([
                'Check for trailing commas',
                'Verify quote usage (use double quotes)',
                'Check boolean values (true/false, not True/False)',
                'Ensure proper JSON structure'
            ])
        
        if 'expecting' in error_msg:
            error_info['suggested_fixes'].append('Check for missing or extra characters')
        
        if 'validation' in error_msg:
            error_info['suggested_fixes'].extend([
                'Check data types match expected schema',
                'Verify required fields are present',
                'Check field value constraints'
            ])
        
        return error_info
    
    def get_recovery_suggestions(self, text: str, parser_type: str = 'general') -> List[str]:
        """
        Get recovery suggestions for failed parsing.
        
        Args:
            text: Text that failed to parse
            parser_type: Type of parser
            
        Returns:
            List of recovery suggestions
        """
        suggestions = []
        
        # General suggestions
        if not text.strip():
            suggestions.append("Text is empty or contains only whitespace")
        
        if len(text) > 10000:
            suggestions.append("Text is very long, consider truncating")
        
        # Parser-specific suggestions
        if parser_type == 'json':
            if '{' not in text and '[' not in text:
                suggestions.append("Text doesn't appear to contain JSON structure")
            
            if text.count('{') != text.count('}'):
                suggestions.append("Mismatched curly braces")
            
            if text.count('[') != text.count(']'):
                suggestions.append("Mismatched square brackets")
            
            if "'" in text:
                suggestions.append("Consider replacing single quotes with double quotes")
        
        return suggestions