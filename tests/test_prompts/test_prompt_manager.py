"""
Tests for PromptManager class.
"""

import pytest
import tempfile
import yaml
from pathlib import Path
from unittest.mock import patch, MagicMock

from ai_toolkit.prompts.prompt_manager import PromptManager
from ai_toolkit.prompts.prompt_templates import SimplePromptTemplate, ChatPromptTemplate


class TestPromptManager:
    """Test PromptManager class."""
    
    def test_init_without_paths(self):
        """Test initializing PromptManager without template paths."""
        manager = PromptManager()
        assert isinstance(manager, PromptManager)
        assert len(manager._templates) >= 0  # May load from default paths
    
    def test_create_template(self):
        """Test creating a new template."""
        manager = PromptManager()
        
        template = manager.create_template(
            name="test_template",
            template_type="simple",
            description="Test template",
            template="Hello {name}"
        )
        
        assert isinstance(template, SimplePromptTemplate)
        assert template.name == "test_template"
        assert "test_template" in manager._templates
    
    def test_load_template_from_data(self):
        """Test loading template from data dictionary."""
        manager = PromptManager()
        
        template_data = {
            'description': 'Test template',
            'template': 'Hello {name}',
            'category': 'greeting'
        }
        
        template = manager.load_template("test_template", template_data)
        
        assert template.name == "test_template"
        assert template.template == "Hello {name}"
        assert "test_template" in manager._templates
    
    def test_get_template(self):
        """Test getting template by name."""
        manager = PromptManager()
        
        # Create a template
        manager.create_template(
            name="test_template",
            template_type="simple",
            description="Test template",
            template="Hello {name}"
        )
        
        # Get the template
        template = manager.get_template("test_template")
        assert template is not None
        assert template.name == "test_template"
        
        # Get non-existent template
        template = manager.get_template("nonexistent")
        assert template is None
    
    def test_render_template(self):
        """Test rendering template with variables."""
        manager = PromptManager()
        
        manager.create_template(
            name="greeting",
            template_type="simple",
            description="Greeting template",
            template="Hello {name}, welcome to {place}!"
        )
        
        rendered = manager.render_template("greeting", name="Alice", place="Wonderland")
        assert rendered == "Hello Alice, welcome to Wonderland!"
    
    def test_render_template_not_found(self):
        """Test error when rendering non-existent template."""
        manager = PromptManager()
        
        with pytest.raises(ValueError, match="Template not found"):
            manager.render_template("nonexistent", name="Alice")
    
    def test_render_template_missing_variables(self):
        """Test error when rendering template with missing variables."""
        manager = PromptManager()
        
        manager.create_template(
            name="greeting",
            template_type="simple",
            description="Greeting template",
            template="Hello {name}, welcome to {place}!"
        )
        
        with pytest.raises(ValueError, match="Failed to render template"):
            manager.render_template("greeting", name="Alice")  # missing 'place'
    
    def test_list_templates(self):
        """Test listing all templates."""
        manager = PromptManager()
        
        # Create some templates
        manager.create_template(
            name="template1",
            template_type="simple",
            description="First template",
            template="Hello {name}"
        )
        
        manager.create_template(
            name="template2",
            template_type="chat",
            description="Second template",
            template="Hi {name}",
            human_message="Hi {name}"
        )
        
        templates = manager.list_templates()
        
        assert len(templates) >= 2
        template_names = [t['name'] for t in templates]
        assert "template1" in template_names
        assert "template2" in template_names
    
    def test_list_template_names(self):
        """Test listing template names."""
        manager = PromptManager()
        
        manager.create_template(
            name="template1",
            template_type="simple",
            description="First template",
            template="Hello {name}"
        )
        
        names = manager.list_template_names()
        assert "template1" in names
    
    def test_remove_template(self):
        """Test removing template."""
        manager = PromptManager()
        
        manager.create_template(
            name="test_template",
            template_type="simple",
            description="Test template",
            template="Hello {name}"
        )
        
        # Template should exist
        assert manager.get_template("test_template") is not None
        
        # Remove template
        removed = manager.remove_template("test_template")
        assert removed is True
        assert manager.get_template("test_template") is None
        
        # Try to remove non-existent template
        removed = manager.remove_template("nonexistent")
        assert removed is False
    
    def test_clear_templates(self):
        """Test clearing all templates."""
        manager = PromptManager()
        
        manager.create_template(
            name="template1",
            template_type="simple",
            description="First template",
            template="Hello {name}"
        )
        
        assert len(manager._templates) >= 1
        
        manager.clear_templates()
        assert len(manager._templates) == 0
    
    def test_get_templates_by_category(self):
        """Test getting templates by category."""
        manager = PromptManager()
        
        manager.create_template(
            name="greeting1",
            template_type="simple",
            description="Greeting template",
            template="Hello {name}",
            category="greeting"
        )
        
        manager.create_template(
            name="greeting2",
            template_type="simple",
            description="Another greeting template",
            template="Hi {name}",
            category="greeting"
        )
        
        manager.create_template(
            name="farewell",
            template_type="simple",
            description="Farewell template",
            template="Goodbye {name}",
            category="farewell"
        )
        
        greeting_templates = manager.get_templates_by_category("greeting")
        assert len(greeting_templates) == 2
        assert "greeting1" in greeting_templates
        assert "greeting2" in greeting_templates
        
        farewell_templates = manager.get_templates_by_category("farewell")
        assert len(farewell_templates) == 1
        assert "farewell" in farewell_templates
    
    def test_search_templates(self):
        """Test searching templates."""
        manager = PromptManager()
        
        manager.create_template(
            name="greeting_template",
            template_type="simple",
            description="A friendly greeting template",
            template="Hello {name}",
            category="greeting"
        )
        
        manager.create_template(
            name="farewell_template",
            template_type="simple",
            description="A polite farewell template",
            template="Goodbye {name}",
            category="farewell"
        )
        
        # Search by name
        results = manager.search_templates("greeting")
        assert len(results) == 1
        assert results[0]['name'] == "greeting_template"
        
        # Search by description
        results = manager.search_templates("friendly")
        assert len(results) == 1
        assert results[0]['name'] == "greeting_template"
        
        # Search by category
        results = manager.search_templates("farewell")
        assert len(results) == 1
        assert results[0]['name'] == "farewell_template"
    
    def test_validate_template_variables(self):
        """Test validating template variables."""
        manager = PromptManager()
        
        manager.create_template(
            name="greeting",
            template_type="simple",
            description="Greeting template",
            template="Hello {name}, you are {age} years old"
        )
        
        # Valid variables
        assert manager.validate_template_variables("greeting", name="Alice", age=25) is True
        
        # Missing variables
        with pytest.raises(ValueError, match="Missing required variables"):
            manager.validate_template_variables("greeting", name="Alice")
        
        # Non-existent template
        with pytest.raises(ValueError, match="Template not found"):
            manager.validate_template_variables("nonexistent", name="Alice")
    
    def test_get_template_variables(self):
        """Test getting template variables."""
        manager = PromptManager()
        
        manager.create_template(
            name="greeting",
            template_type="simple",
            description="Greeting template",
            template="Hello {name}, you are {age} years old"
        )
        
        variables = manager.get_template_variables("greeting")
        assert set(variables) == {"name", "age"}
        
        # Non-existent template
        with pytest.raises(ValueError, match="Template not found"):
            manager.get_template_variables("nonexistent")
    
    def test_clone_template(self):
        """Test cloning template."""
        manager = PromptManager()
        
        manager.create_template(
            name="original",
            template_type="simple",
            description="Original template",
            template="Hello {name}",
            category="greeting"
        )
        
        cloned = manager.clone_template(
            "original",
            "cloned",
            description="Cloned template",
            template="Hi {name}"
        )
        
        assert cloned.name == "cloned"
        assert cloned.description == "Cloned template"
        assert cloned.template == "Hi {name}"
        assert cloned.category == "greeting"  # Inherited from original
        assert "cloned" in manager._templates
        
        # Clone non-existent template
        with pytest.raises(ValueError, match="Source template not found"):
            manager.clone_template("nonexistent", "new_clone")
    
    def test_get_template_info(self):
        """Test getting template information."""
        manager = PromptManager()
        
        manager.create_template(
            name="test_template",
            template_type="simple",
            description="Test template",
            template="Hello {name}",
            category="greeting",
            version="1.0"
        )
        
        info = manager.get_template_info("test_template")
        
        assert info is not None
        assert info['name'] == "test_template"
        assert info['description'] == "Test template"
        assert info['category'] == "greeting"
        assert info['version'] == "1.0"
        assert info['type'] == "simple"
        assert info['variables'] == ["name"]
        assert "Hello {name}" in info['template_content']
        
        # Non-existent template
        info = manager.get_template_info("nonexistent")
        assert info is None
    
    def test_load_templates_from_file(self):
        """Test loading templates from file."""
        template_data = {
            'name': 'file_template',
            'description': 'Template from file',
            'template': 'Hello {name}',
            'category': 'greeting'
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(template_data, f)
            temp_path = f.name
        
        try:
            manager = PromptManager(temp_path)
            
            template = manager.get_template('file_template')
            assert template is not None
            assert template.name == 'file_template'
            assert template.description == 'Template from file'
            
        finally:
            Path(temp_path).unlink()
    
    def test_load_templates_from_directory(self):
        """Test loading templates from directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create template files
            template1_data = {
                'name': 'template1',
                'description': 'First template',
                'template': 'Hello {name}',
                'category': 'greeting'
            }
            
            template2_data = {
                'name': 'template2',
                'description': 'Second template',
                'template': 'Goodbye {name}',
                'category': 'farewell'
            }
            
            with open(temp_path / 'template1.yaml', 'w') as f:
                yaml.dump(template1_data, f)
            
            with open(temp_path / 'template2.yaml', 'w') as f:
                yaml.dump(template2_data, f)
            
            manager = PromptManager(temp_path)
            
            assert manager.get_template('template1') is not None
            assert manager.get_template('template2') is not None
            assert len(manager.list_template_names()) >= 2