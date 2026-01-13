#!/usr/bin/env python3
"""
Integration test for the prompt management toolkit.
"""

import os
import tempfile
import yaml
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)

from ai_toolkit.prompts import PromptManager, PromptLoader, create_template


def test_prompt_templates_basic():
    """Test basic prompt template functionality."""
    print("🧪 Testing Prompt Templates Basic Functionality...")
    
    try:
        # Test creating different types of templates
        templates = {}
        
        # Simple template
        simple_template = create_template(
            'simple',
            name='greeting',
            description='Simple greeting template',
            template='Hello {name}, welcome to {place}!'
        )
        templates['simple'] = simple_template
        print("✅ Simple template created successfully")
        
        # Chat template
        chat_template = create_template(
            'chat',
            name='assistant_chat',
            description='Assistant chat template',
            template='How can I help you with {topic}?',
            human_message='I need help with {topic}',
            system_message='You are a helpful assistant specializing in {topic}'
        )
        templates['chat'] = chat_template
        print("✅ Chat template created successfully")
        
        # System template
        system_template = create_template(
            'system',
            name='code_reviewer',
            description='Code review system template',
            template='You are a code reviewer. Review the following {language} code.',
            instructions=['Check for bugs', 'Suggest improvements', 'Verify best practices'],
            constraints=['Be constructive', 'Provide examples']
        )
        templates['system'] = system_template
        print("✅ System template created successfully")
        
        # Few-shot template
        few_shot_template = create_template(
            'few_shot',
            name='translation',
            description='Translation few-shot template',
            template='Translate to {target_language}: {text}',
            examples=[
                {'text': 'Hello', 'target_language': 'Spanish', 'translation': 'Hola'},
                {'text': 'Goodbye', 'target_language': 'Spanish', 'translation': 'Adiós'}
            ],
            example_template='English: {text}\n{target_language}: {translation}',
            suffix='English: {text}\n{target_language}:'
        )
        templates['few_shot'] = few_shot_template
        print("✅ Few-shot template created successfully")
        
        print(f"✅ Created {len(templates)} different template types")
        return True
        
    except Exception as e:
        print(f"❌ Prompt templates basic test failed: {e}")
        return False


def test_template_rendering():
    """Test template rendering functionality."""
    print("\n🧪 Testing Template Rendering...")
    
    try:
        # Create templates for testing
        simple_template = create_template(
            'simple',
            name='greeting',
            description='Greeting template',
            template='Hello {name}, you are {age} years old and live in {city}.'
        )
        
        chat_template = create_template(
            'chat',
            name='support_chat',
            description='Customer support chat',
            template='How can I help you?',
            human_message='I have a problem with {product}',
            system_message='You are a customer support agent for {company}'
        )
        
        # Test rendering
        simple_rendered = simple_template.render(name="Alice", age=25, city="New York")
        print(f"✅ Simple template rendered: {simple_rendered[:50]}...")
        
        chat_rendered = chat_template.render(product="laptop", company="TechCorp")
        print(f"✅ Chat template rendered successfully")
        
        # Test variable validation
        try:
            simple_template.render(name="Bob")  # Missing age and city
            print("❌ Should have failed with missing variables")
            return False
        except ValueError as e:
            if "Missing required variables" in str(e):
                print("✅ Variable validation working correctly")
            else:
                print(f"❌ Unexpected error: {e}")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Template rendering test failed: {e}")
        return False


def test_prompt_loader():
    """Test prompt loader functionality."""
    print("\n🧪 Testing Prompt Loader...")
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create test template files
            template_data = {
                'name': 'test_template',
                'description': 'Test template from file',
                'template': 'Hello {name}, today is {day}',
                'category': 'greeting',
                'version': '1.0'
            }
            
            # YAML file
            yaml_file = temp_path / 'test_template.yaml'
            with open(yaml_file, 'w') as f:
                yaml.dump(template_data, f)
            
            # JSON file
            import json
            json_file = temp_path / 'json_template.json'
            json_data = {
                'name': 'json_template',
                'description': 'Template from JSON',
                'template': 'Welcome {user} to {service}',
                'category': 'welcome'
            }
            with open(json_file, 'w') as f:
                json.dump(json_data, f)
            
            # Text file
            text_file = temp_path / 'simple.txt'
            with open(text_file, 'w') as f:
                f.write('This is a simple template with {variable}')
            
            # Test loader
            loader = PromptLoader(temp_path)
            
            # Load single file
            template = loader.load_from_file('test_template.yaml')
            print(f"✅ Loaded template from YAML: {template.name}")
            
            # Load from directory
            templates = loader.load_from_dir('.')
            print(f"✅ Loaded {len(templates)} templates from directory")
            
            # Validate template
            assert loader.validate_template(template_data) is True
            print("✅ Template validation working")
            
            return True
            
    except Exception as e:
        print(f"❌ Prompt loader test failed: {e}")
        return False


def test_prompt_manager():
    """Test prompt manager functionality."""
    print("\n🧪 Testing Prompt Manager...")
    
    try:
        manager = PromptManager()
        
        # Create templates
        manager.create_template(
            name='greeting',
            template_type='simple',
            description='Greeting template',
            template='Hello {name}!',
            category='greeting'
        )
        
        manager.create_template(
            name='farewell',
            template_type='simple',
            description='Farewell template',
            template='Goodbye {name}, see you {when}!',
            category='farewell'
        )
        
        print(f"✅ Created templates in manager")
        
        # Test template operations
        template = manager.get_template('greeting')
        assert template is not None
        print("✅ Retrieved template successfully")
        
        # Test rendering
        rendered = manager.render_template('greeting', name='Alice')
        assert rendered == 'Hello Alice!'
        print("✅ Template rendering through manager works")
        
        # Test listing
        templates = manager.list_templates()
        assert len(templates) >= 2
        print(f"✅ Listed {len(templates)} templates")
        
        # Test search
        greeting_templates = manager.search_templates('greeting')
        assert len(greeting_templates) >= 1
        print("✅ Template search working")
        
        # Test categories
        greeting_category = manager.get_templates_by_category('greeting')
        assert 'greeting' in greeting_category
        print("✅ Category filtering working")
        
        # Test cloning
        cloned = manager.clone_template('greeting', 'greeting_clone', 
                                      description='Cloned greeting')
        assert cloned.name == 'greeting_clone'
        print("✅ Template cloning working")
        
        # Test variables
        variables = manager.get_template_variables('farewell')
        assert set(variables) == {'name', 'when'}
        print("✅ Variable extraction working")
        
        return True
        
    except Exception as e:
        print(f"❌ Prompt manager test failed: {e}")
        return False


def test_existing_templates():
    """Test loading existing template files."""
    print("\n🧪 Testing Existing Template Files...")
    
    try:
        # Check if config/prompts directory exists
        prompts_dir = Path('config/prompts')
        if prompts_dir.exists():
            manager = PromptManager(prompts_dir)
            templates = manager.list_templates()
            print(f"✅ Loaded {len(templates)} templates from config/prompts")
            
            # Test rendering existing templates if any
            for template_info in templates[:2]:  # Test first 2 templates
                template_name = template_info['name']
                variables = template_info['variables']
                
                if variables:
                    # Create dummy values for variables
                    dummy_values = {}
                    for var in variables:
                        if var in ['name', 'user_name']:
                            dummy_values[var] = 'Alice'
                        elif var in ['language', 'lang']:
                            dummy_values[var] = 'Python'
                        elif var in ['code']:
                            dummy_values[var] = 'print("hello")'
                        elif var in ['user_input', 'input']:
                            dummy_values[var] = 'Hello world'
                        else:
                            dummy_values[var] = 'test_value'
                    
                    try:
                        rendered = manager.render_template(template_name, **dummy_values)
                        print(f"✅ Rendered existing template '{template_name}'")
                    except Exception as e:
                        print(f"⚠️  Could not render template '{template_name}': {e}")
                else:
                    print(f"✅ Template '{template_name}' has no variables")
        else:
            print("ℹ️  No config/prompts directory found, creating test templates")
            
            # Create some test templates in memory
            manager = PromptManager()
            manager.create_template(
                name='system_chat',
                template_type='chat',
                description='System chat template',
                template='How can I help you?',
                human_message='{user_input}',
                system_message='You are a helpful AI assistant.'
            )
            
            manager.create_template(
                name='code_review',
                template_type='system',
                description='Code review template',
                template='Please review this {language} code: {code}',
                instructions=['Check for bugs', 'Suggest improvements'],
                constraints=['Be constructive', 'Provide examples']
            )
            
            templates = manager.list_templates()
            print(f"✅ Created {len(templates)} test templates")
        
        return True
        
    except Exception as e:
        print(f"❌ Existing templates test failed: {e}")
        return False


def test_langchain_integration():
    """Test LangChain integration."""
    print("\n🧪 Testing LangChain Integration...")
    
    try:
        # Create templates and convert to LangChain format
        simple_template = create_template(
            'simple',
            name='simple_test',
            description='Simple template for LangChain',
            template='Translate "{text}" to {language}'
        )
        
        chat_template = create_template(
            'chat',
            name='chat_test',
            description='Chat template for LangChain',
            template='Help with translation',
            human_message='Translate "{text}" to {language}',
            system_message='You are a professional translator'
        )
        
        # Convert to LangChain format
        lc_simple = simple_template.to_langchain()
        lc_chat = chat_template.to_langchain()
        
        print("✅ Simple template converted to LangChain format")
        print("✅ Chat template converted to LangChain format")
        
        # Test that LangChain templates have expected attributes
        assert hasattr(lc_simple, 'template')
        assert hasattr(lc_chat, 'messages')
        
        print("✅ LangChain templates have expected structure")
        
        return True
        
    except Exception as e:
        print(f"❌ LangChain integration test failed: {e}")
        return False


def main():
    """Run all integration tests."""
    print("🚀 Starting Prompt Management Toolkit Integration Tests...\n")
    
    tests = [
        test_prompt_templates_basic,
        test_template_rendering,
        test_prompt_loader,
        test_prompt_manager,
        test_existing_templates,
        test_langchain_integration,
    ]
    
    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test_func.__name__} crashed: {e}")
            results.append(False)
    
    # Summary
    print("\n" + "="*60)
    print("📊 Integration Test Results Summary:")
    print("="*60)
    
    passed = sum(results)
    total = len(results)
    
    for i, (test_func, result) in enumerate(zip(tests, results)):
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_func.__name__}")
    
    print(f"\n📈 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All integration tests passed! Prompt management toolkit is ready.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)