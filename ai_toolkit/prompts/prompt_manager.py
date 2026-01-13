"""
Prompt manager for centralized prompt template management.
"""

from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import logging

from .prompt_templates import BasePromptTemplate, create_template, detect_template_type
from .prompt_loader import PromptLoader


logger = logging.getLogger(__name__)


class PromptManager:
    """
    Central manager for prompt templates.
    
    Handles template loading, caching, rendering, and management.
    """
    
    def __init__(self, template_paths: Optional[Union[str, Path, List[Union[str, Path]]]] = None):
        """
        Initialize prompt manager.
        
        Args:
            template_paths: Path(s) to template directories or files
        """
        self._templates: Dict[str, BasePromptTemplate] = {}
        self._loaders: Dict[str, PromptLoader] = {}
        self._default_loader = PromptLoader()
        
        # Load templates from provided paths
        if template_paths:
            self.load_templates(template_paths)
        else:
            # Try to load from default locations
            self._load_default_templates()
    
    def _load_default_templates(self) -> None:
        """Load templates from default locations."""
        default_paths = [
            Path("config/prompts"),
            Path("prompts"),
            Path("templates"),
            Path("../config/prompts"),
        ]
        
        for path in default_paths:
            if path.exists() and path.is_dir():
                try:
                    self.load_templates(path)
                    logger.info(f"Loaded templates from default path: {path}")
                    break
                except Exception as e:
                    logger.debug(f"Failed to load from default path {path}: {e}")
    
    def load_templates(self, paths: Union[str, Path, List[Union[str, Path]]]) -> int:
        """
        Load templates from specified paths.
        
        Args:
            paths: Path(s) to template directories or files
            
        Returns:
            Number of templates loaded
        """
        if not isinstance(paths, list):
            paths = [paths]
        
        loaded_count = 0
        
        for path in paths:
            path = Path(path)
            
            try:
                if path.is_file():
                    # Load single template file
                    template = self._default_loader.load_from_file(path)
                    self._templates[template.name] = template
                    loaded_count += 1
                    
                elif path.is_dir():
                    # Load all templates from directory
                    loader = PromptLoader(path)
                    templates = loader.load_from_dir(".", recursive=True)
                    
                    for name, template in templates.items():
                        self._templates[name] = template
                        loaded_count += 1
                    
                    # Cache the loader for this path
                    self._loaders[str(path)] = loader
                    
                else:
                    logger.warning(f"Path does not exist: {path}")
                    
            except Exception as e:
                logger.error(f"Failed to load templates from {path}: {e}")
        
        logger.info(f"Loaded {loaded_count} templates")
        return loaded_count
    
    def load_template(self, name: str, template_data: Dict[str, Any]) -> BasePromptTemplate:
        """
        Load template from data dictionary.
        
        Args:
            name: Template name
            template_data: Template configuration data
            
        Returns:
            Loaded template
        """
        # Ensure name is set
        template_data['name'] = name
        
        # Detect template type if not specified
        template_type = template_data.get('type', detect_template_type(template_data))
        
        # Remove 'type' from data as it's not part of the template schema
        template_data = {k: v for k, v in template_data.items() if k != 'type'}
        
        template = create_template(template_type, **template_data)
        self._templates[name] = template
        
        logger.info(f"Loaded template: {name}")
        return template
    
    def get_template(self, name: str) -> Optional[BasePromptTemplate]:
        """
        Get template by name.
        
        Args:
            name: Template name
            
        Returns:
            Template instance or None if not found
        """
        return self._templates.get(name)
    
    def render_template(self, template_name: str, **kwargs) -> str:
        """
        Render template with provided variables.
        
        Args:
            template_name: Template name
            **kwargs: Template variables
            
        Returns:
            Rendered template string
            
        Raises:
            ValueError: If template not found or variables are invalid
        """
        template = self.get_template(template_name)
        if not template:
            raise ValueError(f"Template not found: {template_name}")
        
        try:
            return template.render(**kwargs)
        except Exception as e:
            raise ValueError(f"Failed to render template '{template_name}': {e}")
    
    def save_template(self, template: BasePromptTemplate, file_path: Union[str, Path], 
                     format: str = 'yaml') -> None:
        """
        Save template to file.
        
        Args:
            template: Template to save
            file_path: Output file path
            format: Output format ('yaml', 'json')
        """
        self._default_loader.save_template(template, file_path, format)
        
        # Add to templates if not already present
        if template.name not in self._templates:
            self._templates[template.name] = template
    
    def create_template(self, name: str, template_type: str, **kwargs) -> BasePromptTemplate:
        """
        Create a new template.
        
        Args:
            name: Template name
            template_type: Type of template ('chat', 'system', 'few_shot', 'simple')
            **kwargs: Template configuration
            
        Returns:
            Created template
        """
        kwargs['name'] = name
        template = create_template(template_type, **kwargs)
        self._templates[name] = template
        
        logger.info(f"Created template: {name} (type: {template_type})")
        return template
    
    def list_templates(self) -> List[Dict[str, Any]]:
        """
        List all available templates.
        
        Returns:
            List of template information dictionaries
        """
        templates_info = []
        
        for name, template in self._templates.items():
            info = {
                'name': template.name,
                'description': template.description,
                'category': template.category,
                'version': template.version,
                'type': template.__class__.__name__.replace('PromptTemplate', '').lower(),
                'variables': template.get_variables(),
                'metadata': template.metadata
            }
            templates_info.append(info)
        
        return templates_info
    
    def list_template_names(self) -> List[str]:
        """
        List all template names.
        
        Returns:
            List of template names
        """
        return list(self._templates.keys())
    
    def remove_template(self, name: str) -> bool:
        """
        Remove template by name.
        
        Args:
            name: Template name
            
        Returns:
            True if template was removed, False if not found
        """
        if name in self._templates:
            del self._templates[name]
            logger.info(f"Removed template: {name}")
            return True
        return False
    
    def clear_templates(self) -> None:
        """Clear all templates."""
        self._templates.clear()
        logger.info("Cleared all templates")
    
    def get_templates_by_category(self, category: str) -> Dict[str, BasePromptTemplate]:
        """
        Get templates by category.
        
        Args:
            category: Template category
            
        Returns:
            Dictionary of templates in the category
        """
        return {
            name: template for name, template in self._templates.items()
            if template.category == category
        }
    
    def search_templates(self, query: str, search_fields: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Search templates by query.
        
        Args:
            query: Search query
            search_fields: Fields to search in (default: ['name', 'description', 'category'])
            
        Returns:
            List of matching template information
        """
        if search_fields is None:
            search_fields = ['name', 'description', 'category']
        
        query = query.lower()
        matching_templates = []
        
        for name, template in self._templates.items():
            match = False
            
            for field in search_fields:
                if hasattr(template, field):
                    field_value = str(getattr(template, field)).lower()
                    if query in field_value:
                        match = True
                        break
            
            if match:
                info = {
                    'name': template.name,
                    'description': template.description,
                    'category': template.category,
                    'type': template.__class__.__name__.replace('PromptTemplate', '').lower(),
                    'variables': template.get_variables()
                }
                matching_templates.append(info)
        
        return matching_templates
    
    def validate_template_variables(self, template_name: str, **kwargs) -> bool:
        """
        Validate variables for a template.
        
        Args:
            template_name: Template name
            **kwargs: Variables to validate
            
        Returns:
            True if variables are valid
            
        Raises:
            ValueError: If template not found or variables are invalid
        """
        template = self.get_template(template_name)
        if not template:
            raise ValueError(f"Template not found: {template_name}")
        
        return template.validate_variables(**kwargs)
    
    def get_template_variables(self, template_name: str) -> List[str]:
        """
        Get required variables for a template.
        
        Args:
            template_name: Template name
            
        Returns:
            List of required variables
            
        Raises:
            ValueError: If template not found
        """
        template = self.get_template(template_name)
        if not template:
            raise ValueError(f"Template not found: {template_name}")
        
        return template.get_variables()
    
    def clone_template(self, source_name: str, target_name: str, **modifications) -> BasePromptTemplate:
        """
        Clone an existing template with optional modifications.
        
        Args:
            source_name: Name of template to clone
            target_name: Name for the new template
            **modifications: Fields to modify in the cloned template
            
        Returns:
            Cloned template
            
        Raises:
            ValueError: If source template not found
        """
        source_template = self.get_template(source_name)
        if not source_template:
            raise ValueError(f"Source template not found: {source_name}")
        
        # Get template data
        template_data = source_template.model_dump()
        
        # Apply modifications
        template_data.update(modifications)
        template_data['name'] = target_name
        
        # Detect template type
        template_type = detect_template_type(template_data)
        
        # Create new template
        cloned_template = create_template(template_type, **template_data)
        self._templates[target_name] = cloned_template
        
        logger.info(f"Cloned template '{source_name}' to '{target_name}'")
        return cloned_template
    
    def get_template_info(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a template.
        
        Args:
            name: Template name
            
        Returns:
            Template information dictionary or None if not found
        """
        template = self.get_template(name)
        if not template:
            return None
        
        return {
            'name': template.name,
            'description': template.description,
            'category': template.category,
            'version': template.version,
            'type': template.__class__.__name__.replace('PromptTemplate', '').lower(),
            'variables': template.get_variables(),
            'metadata': template.metadata,
            'template_content': template.template[:200] + '...' if len(template.template) > 200 else template.template
        }
    
    def export_templates(self, output_path: Union[str, Path], format: str = 'yaml') -> None:
        """
        Export all templates to a directory.
        
        Args:
            output_path: Output directory path
            format: Export format ('yaml', 'json')
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        exported_count = 0
        
        for name, template in self._templates.items():
            try:
                # Create safe filename
                safe_name = "".join(c for c in name if c.isalnum() or c in (' ', '-', '_')).rstrip()
                filename = f"{safe_name}.{format}"
                file_path = output_path / filename
                
                self.save_template(template, file_path, format)
                exported_count += 1
                
            except Exception as e:
                logger.error(f"Failed to export template '{name}': {e}")
        
        logger.info(f"Exported {exported_count} templates to {output_path}")
    
    def reload_templates(self) -> int:
        """
        Reload all templates from their original sources.
        
        Returns:
            Number of templates reloaded
        """
        # Clear current templates
        self.clear_templates()
        
        # Reload from all cached loaders
        loaded_count = 0
        for path, loader in self._loaders.items():
            try:
                templates = loader.load_from_dir(".", recursive=True)
                for name, template in templates.items():
                    self._templates[name] = template
                    loaded_count += 1
            except Exception as e:
                logger.error(f"Failed to reload templates from {path}: {e}")
        
        # Try default locations if no loaders
        if not self._loaders:
            self._load_default_templates()
        
        logger.info(f"Reloaded {loaded_count} templates")
        return loaded_count