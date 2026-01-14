"""
Prompt loader for loading templates from files and directories.

This module provides functionality to load prompt templates from various sources
including files, directories, and different formats (YAML, JSON, text).

Classes:
    PromptLoader: Loader for prompt templates from files
        - Loads templates from YAML, JSON, and text files
        - Supports batch loading from directories
        - Validates template format and structure
        
        Methods:
            __init__(base_path): Initialize loader with base path
            load_from_file(file_path): Load single template from file
            load_from_dir(dir_path, pattern, recursive): Load templates from directory
            validate_template(template_data): Validate template structure
            save_template(template, file_path): Save template to file
            list_templates(dir_path): List available templates in directory
            get_template_info(file_path): Get template metadata
"""

import os
import yaml
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import logging

from .prompt_templates import BasePromptTemplate, create_template, detect_template_type


logger = logging.getLogger(__name__)


class PromptLoader:
    """Loader for prompt templates from files and directories."""
    
    SUPPORTED_EXTENSIONS = {'.yaml', '.yml', '.json', '.txt', '.md'}
    
    def __init__(self, base_path: Optional[Union[str, Path]] = None):
        """
        Initialize prompt loader.
        
        Args:
            base_path: Base directory path for loading templates
        """
        self.base_path = Path(base_path) if base_path else Path.cwd()
        self._template_cache: Dict[str, BasePromptTemplate] = {}
    
    def load_from_file(self, file_path: Union[str, Path]) -> BasePromptTemplate:
        """
        Load prompt template from a single file.
        
        Args:
            file_path: Path to the template file
            
        Returns:
            Loaded prompt template
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is invalid
        """
        file_path = Path(file_path)
        
        # Make path relative to base_path if not absolute
        if not file_path.is_absolute():
            file_path = self.base_path / file_path
        
        if not file_path.exists():
            raise FileNotFoundError(f"Template file not found: {file_path}")
        
        if file_path.suffix not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file extension: {file_path.suffix}. "
                           f"Supported: {self.SUPPORTED_EXTENSIONS}")
        
        try:
            template_data = self._load_file_content(file_path)
            template = self._create_template_from_data(template_data, file_path)
            
            # Cache the template
            try:
                cache_key = str(file_path.relative_to(self.base_path))
            except ValueError:
                # If file_path is not relative to base_path, use the filename
                cache_key = file_path.name
            self._template_cache[cache_key] = template
            
            logger.info(f"Loaded template from {file_path}")
            return template
            
        except Exception as e:
            raise ValueError(f"Failed to load template from {file_path}: {e}")
    
    def load_from_dir(self, dir_path: Union[str, Path], recursive: bool = True) -> Dict[str, BasePromptTemplate]:
        """
        Load all prompt templates from a directory.
        
        Args:
            dir_path: Directory path containing template files
            recursive: Whether to search subdirectories recursively
            
        Returns:
            Dictionary mapping template names to template objects
            
        Raises:
            FileNotFoundError: If directory doesn't exist
        """
        dir_path = Path(dir_path)
        
        # Make path relative to base_path if not absolute
        if not dir_path.is_absolute():
            dir_path = self.base_path / dir_path
        
        if not dir_path.exists():
            raise FileNotFoundError(f"Template directory not found: {dir_path}")
        
        if not dir_path.is_dir():
            raise ValueError(f"Path is not a directory: {dir_path}")
        
        templates = {}
        
        # Get file pattern for search
        pattern = "**/*" if recursive else "*"
        
        for file_path in dir_path.glob(pattern):
            if file_path.is_file() and file_path.suffix in self.SUPPORTED_EXTENSIONS:
                try:
                    template = self.load_from_file(file_path)
                    
                    # Use template name as key, fallback to filename
                    template_name = template.name if hasattr(template, 'name') and template.name else file_path.stem
                    templates[template_name] = template
                    
                except Exception as e:
                    logger.warning(f"Failed to load template from {file_path}: {e}")
                    continue
        
        logger.info(f"Loaded {len(templates)} templates from {dir_path}")
        return templates
    
    def _load_file_content(self, file_path: Path) -> Dict[str, Any]:
        """Load and parse file content based on extension."""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if file_path.suffix in {'.yaml', '.yml'}:
            return yaml.safe_load(content) or {}
        
        elif file_path.suffix == '.json':
            return json.loads(content)
        
        elif file_path.suffix in {'.txt', '.md'}:
            # For plain text files, create a simple template structure
            return {
                'name': file_path.stem,
                'description': f"Template loaded from {file_path.name}",
                'template': content,
                'category': 'text'
            }
        
        else:
            raise ValueError(f"Unsupported file extension: {file_path.suffix}")
    
    def _create_template_from_data(self, template_data: Dict[str, Any], file_path: Path) -> BasePromptTemplate:
        """Create template object from loaded data."""
        # Ensure required fields
        if 'name' not in template_data:
            template_data['name'] = file_path.stem
        
        if 'description' not in template_data:
            template_data['description'] = f"Template loaded from {file_path.name}"
        
        # Detect template type if not specified
        template_type = template_data.get('type', detect_template_type(template_data))
        
        # Remove 'type' from data as it's not part of the template schema
        template_data = {k: v for k, v in template_data.items() if k != 'type'}
        
        return create_template(template_type, **template_data)
    
    def validate_template(self, template_data: Dict[str, Any]) -> bool:
        """
        Validate template data structure.
        
        Args:
            template_data: Template data to validate
            
        Returns:
            True if valid
            
        Raises:
            ValueError: If template data is invalid
        """
        required_fields = ['template']
        
        # Check required fields
        for field in required_fields:
            if field not in template_data:
                raise ValueError(f"Missing required field: {field}")
        
        # Validate template content
        template_content = template_data['template']
        if not template_content or not str(template_content).strip():
            raise ValueError("Template content cannot be empty")
        
        # Try to create template to validate structure
        try:
            template_type = template_data.get('type', detect_template_type(template_data))
            create_template(template_type, **template_data)
        except Exception as e:
            raise ValueError(f"Invalid template structure: {e}")
        
        return True
    
    def get_cached_template(self, template_name: str) -> Optional[BasePromptTemplate]:
        """
        Get cached template by name.
        
        Args:
            template_name: Name of the template
            
        Returns:
            Cached template or None if not found
        """
        return self._template_cache.get(template_name)
    
    def clear_cache(self) -> None:
        """Clear template cache."""
        self._template_cache.clear()
        logger.info("Template cache cleared")
    
    def list_cached_templates(self) -> List[str]:
        """
        List names of cached templates.
        
        Returns:
            List of cached template names
        """
        return list(self._template_cache.keys())
    
    def save_template(self, template: BasePromptTemplate, file_path: Union[str, Path], 
                     format: str = 'yaml') -> None:
        """
        Save template to file.
        
        Args:
            template: Template to save
            file_path: Output file path
            format: Output format ('yaml', 'json')
            
        Raises:
            ValueError: If format is not supported
        """
        file_path = Path(file_path)
        
        # Make path relative to base_path if not absolute
        if not file_path.is_absolute():
            file_path = self.base_path / file_path
        
        # Ensure directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert template to dictionary
        template_data = template.dict()
        
        # Add type information
        template_data['type'] = template.__class__.__name__.replace('PromptTemplate', '').lower()
        
        # Save based on format
        if format.lower() == 'yaml':
            with open(file_path, 'w', encoding='utf-8') as f:
                yaml.dump(template_data, f, default_flow_style=False, allow_unicode=True)
        
        elif format.lower() == 'json':
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(template_data, f, indent=2, ensure_ascii=False)
        
        else:
            raise ValueError(f"Unsupported format: {format}. Supported: yaml, json")
        
        logger.info(f"Template saved to {file_path}")
    
    def find_templates(self, pattern: str, search_path: Optional[Union[str, Path]] = None) -> List[Path]:
        """
        Find template files matching a pattern.
        
        Args:
            pattern: Glob pattern to search for
            search_path: Directory to search in (defaults to base_path)
            
        Returns:
            List of matching file paths
        """
        search_path = Path(search_path) if search_path else self.base_path
        
        if not search_path.is_absolute():
            search_path = self.base_path / search_path
        
        matching_files = []
        
        for file_path in search_path.rglob(pattern):
            if file_path.is_file() and file_path.suffix in self.SUPPORTED_EXTENSIONS:
                matching_files.append(file_path)
        
        return matching_files
    
    def get_template_info(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Get information about a template file without loading it.
        
        Args:
            file_path: Path to template file
            
        Returns:
            Template information dictionary
        """
        file_path = Path(file_path)
        
        if not file_path.is_absolute():
            file_path = self.base_path / file_path
        
        if not file_path.exists():
            raise FileNotFoundError(f"Template file not found: {file_path}")
        
        try:
            template_data = self._load_file_content(file_path)
            
            return {
                'name': template_data.get('name', file_path.stem),
                'description': template_data.get('description', ''),
                'category': template_data.get('category', 'general'),
                'version': template_data.get('version', '1.0'),
                'variables': template_data.get('variables', []),
                'type': template_data.get('type', detect_template_type(template_data)),
                'file_path': str(file_path),
                'file_size': file_path.stat().st_size,
                'modified_time': file_path.stat().st_mtime
            }
            
        except Exception as e:
            return {
                'name': file_path.stem,
                'error': str(e),
                'file_path': str(file_path)
            }