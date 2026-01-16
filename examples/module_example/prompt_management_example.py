#!/usr/bin/env python3
"""
Example usage of the AI Toolkit Prompt Management system.
"""

import tempfile
import yaml
from pathlib import Path

from ai_toolkit.prompts import PromptManager, PromptLoader, create_template


def basic_template_usage():
    """Demonstrate basic template creation and usage."""
    print("🚀 Basic Template Usage Example")
    print("=" * 50)
    
    # Create different types of templates
    templates = {}
    
    # 1. Simple template
    simple_template = create_template(
        'simple',
        name='greeting',
        description='Simple greeting template',
        template='Hello {name}, welcome to {place}!',
        category='greeting'
    )
    templates['simple'] = simple_template
    print("✅ Simple template created")
    
    # 2. Chat template
    chat_template = create_template(
        'chat',
        name='customer_support',
        description='Customer support chat template',
        template='How can I help you?',
        human_message='I have an issue with {product}',
        system_message='You are a helpful customer support agent for {company}. Be polite and professional.',
        category='support'
    )
    templates['chat'] = chat_template
    print("✅ Chat template created")
    
    # 3. System template
    system_template = create_template(
        'system',
        name='code_reviewer',
        description='Code review system template',
        template='Review the following {language} code for quality and best practices.',
        instructions=[
            'Check for potential bugs',
            'Suggest performance improvements',
            'Verify coding standards compliance',
            'Recommend better practices'
        ],
        constraints=[
            'Be constructive in feedback',
            'Provide specific examples',
            'Explain the reasoning behind suggestions'
        ],
        category='development'
    )
    templates['system'] = system_template
    print("✅ System template created")
    
    # 4. Few-shot template
    few_shot_template = create_template(
        'few_shot',
        name='sentiment_analysis',
        description='Sentiment analysis with examples',
        template='Analyze sentiment: {text}',
        examples=[
            {'text': 'I love this product!', 'sentiment': 'positive'},
            {'text': 'This is terrible.', 'sentiment': 'negative'},
            {'text': 'It\'s okay, nothing special.', 'sentiment': 'neutral'}
        ],
        example_template='Text: {text}\nSentiment: {sentiment}',
        prefix='Analyze the sentiment of the following text examples:',
        suffix='Text: {text}\nSentiment:',
        category='analysis'
    )
    templates['few_shot'] = few_shot_template
    print("✅ Few-shot template created")
    
    print(f"\n📊 Created {len(templates)} different template types")
    
    # Test rendering
    print("\n🎨 Template Rendering Examples:")
    
    # Render simple template
    simple_rendered = simple_template.render(name="Alice", place="AI Toolkit")
    print(f"Simple: {simple_rendered}")
    
    # Render chat template
    chat_rendered = chat_template.render(product="laptop", company="TechCorp")
    print(f"Chat: {chat_rendered[:100]}...")
    
    # Render system template
    system_rendered = system_template.render(language="Python")
    print(f"System: {system_rendered[:100]}...")
    
    # Render few-shot template
    few_shot_rendered = few_shot_template.render(text="This is amazing!")
    print(f"Few-shot: {few_shot_rendered[:100]}...")
    
    print()


def prompt_manager_usage():
    """Demonstrate PromptManager functionality."""
    print("⚙️  Prompt Manager Usage Example")
    print("=" * 50)
    
    # Initialize PromptManager
    manager = PromptManager()
    
    # Create templates using manager
    manager.create_template(
        name='email_template',
        template_type='simple',
        description='Professional email template',
        template='Dear {recipient},\n\n{message}\n\nBest regards,\n{sender}',
        category='communication'
    )
    
    manager.create_template(
        name='meeting_summary',
        template_type='system',
        description='Meeting summary template',
        template='Summarize the key points from this {meeting_type} meeting.',
        instructions=[
            'Extract main decisions made',
            'List action items with owners',
            'Note any unresolved issues'
        ],
        category='business'
    )
    
    manager.create_template(
        name='translation_helper',
        template_type='few_shot',
        description='Translation assistant',
        template='Translate to {target_language}: {text}',
        examples=[
            {'text': 'Good morning', 'target_language': 'Spanish', 'translation': 'Buenos días'},
            {'text': 'Thank you', 'target_language': 'French', 'translation': 'Merci'}
        ],
        example_template='{text} → {translation}',
        suffix='Translate to {target_language}: {text} →',
        category='language'
    )
    
    print("✅ Created templates in manager")
    
    # List all templates
    templates = manager.list_templates()
    print(f"📋 Total templates: {len(templates)}")
    for template in templates:
        print(f"   - {template['name']} ({template['type']}) - {template['category']}")
    
    # Search templates
    business_templates = manager.search_templates('meeting')
    print(f"\n🔍 Found {len(business_templates)} templates matching 'meeting'")
    
    # Get templates by category
    communication_templates = manager.get_templates_by_category('communication')
    print(f"📂 Communication category: {len(communication_templates)} templates")
    
    # Render template through manager
    email_content = manager.render_template(
        'email_template',
        recipient='John Doe',
        message='I hope this email finds you well. I wanted to follow up on our previous discussion.',
        sender='Alice Smith'
    )
    print(f"\n📧 Rendered email:\n{email_content}")
    
    # Clone template
    cloned = manager.clone_template(
        'email_template',
        'urgent_email_template',
        description='Urgent email template',
        template='URGENT: Dear {recipient},\n\n{message}\n\nPlease respond ASAP.\n\nBest regards,\n{sender}'
    )
    print(f"✅ Cloned template: {cloned.name}")
    
    # Get template info
    info = manager.get_template_info('translation_helper')
    print(f"\n📊 Template info for '{info['name']}':")
    print(f"   Type: {info['type']}")
    print(f"   Variables: {info['variables']}")
    print(f"   Category: {info['category']}")
    
    print()


def file_operations_example():
    """Demonstrate file loading and saving operations."""
    print("📁 File Operations Example")
    print("=" * 50)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create template files
        templates_data = {
            'greeting.yaml': {
                'name': 'greeting',
                'description': 'Greeting template',
                'template': 'Hello {name}, how are you today?',
                'category': 'social',
                'version': '1.0'
            },
            'code_review.yaml': {
                'name': 'code_review',
                'type': 'system',
                'description': 'Code review template',
                'template': 'Please review this {language} code: {code}',
                'instructions': ['Check for bugs', 'Suggest improvements'],
                'category': 'development',
                'version': '2.0'
            },
            'translation.json': {
                'name': 'translation',
                'type': 'few_shot',
                'description': 'Translation template',
                'template': 'Translate: {text}',
                'examples': [
                    {'text': 'Hello', 'translation': 'Hola'},
                    {'text': 'Goodbye', 'translation': 'Adiós'}
                ],
                'example_template': '{text} = {translation}',
                'suffix': 'Translate: {text} =',
                'category': 'language'
            }
        }
        
        # Save template files
        for filename, data in templates_data.items():
            file_path = temp_path / filename
            if filename.endswith('.yaml'):
                with open(file_path, 'w') as f:
                    yaml.dump(data, f)
            else:  # JSON
                import json
                with open(file_path, 'w') as f:
                    json.dump(data, f, indent=2)
        
        print(f"✅ Created {len(templates_data)} template files")
        
        # Load templates using PromptLoader
        loader = PromptLoader(temp_path)
        
        # Load single file
        greeting_template = loader.load_from_file('greeting.yaml')
        print(f"✅ Loaded single template: {greeting_template.name}")
        
        # Load all templates from directory
        all_templates = loader.load_from_dir('.')
        print(f"✅ Loaded {len(all_templates)} templates from directory")
        
        # Load templates using PromptManager
        manager = PromptManager(temp_path)
        manager_templates = manager.list_templates()
        print(f"✅ PromptManager loaded {len(manager_templates)} templates")
        
        # Test rendering loaded templates
        rendered_greeting = manager.render_template('greeting', name='Bob')
        print(f"📝 Rendered greeting: {rendered_greeting}")
        
        # Save a new template
        new_template = create_template(
            'simple',
            name='farewell',
            description='Farewell template',
            template='Goodbye {name}, see you {when}!',
            category='social'
        )
        
        save_path = temp_path / 'farewell.yaml'
        loader.save_template(new_template, save_path)
        print(f"✅ Saved new template to {save_path.name}")
        
        # Verify saved template
        saved_template = loader.load_from_file('farewell.yaml')
        print(f"✅ Verified saved template: {saved_template.name}")
    
    print()


def langchain_integration_example():
    """Demonstrate LangChain integration."""
    print("🔗 LangChain Integration Example")
    print("=" * 50)
    
    # Create templates
    simple_template = create_template(
        'simple',
        name='simple_prompt',
        description='Simple prompt for LangChain',
        template='Explain {concept} in simple terms'
    )
    
    chat_template = create_template(
        'chat',
        name='chat_prompt',
        description='Chat prompt for LangChain',
        template='Help with question',
        human_message='Can you help me understand {topic}?',
        system_message='You are an expert teacher who explains complex topics simply'
    )
    
    # Convert to LangChain format
    lc_simple = simple_template.to_langchain()
    lc_chat = chat_template.to_langchain()
    
    print("✅ Converted templates to LangChain format")
    print(f"   Simple template type: {type(lc_simple).__name__}")
    print(f"   Chat template type: {type(lc_chat).__name__}")
    
    # Show LangChain template structure
    print(f"   Simple template has 'template' attribute: {hasattr(lc_simple, 'template')}")
    print(f"   Chat template has 'messages' attribute: {hasattr(lc_chat, 'messages')}")
    
    if hasattr(lc_chat, 'messages'):
        print(f"   Chat template has {len(lc_chat.messages)} messages")
    
    print("✅ LangChain integration working correctly")
    print()


def advanced_features_example():
    """Demonstrate advanced features."""
    print("🎯 Advanced Features Example")
    print("=" * 50)
    
    manager = PromptManager()
    
    # Create templates with metadata
    manager.create_template(
        name='advanced_template',
        template_type='system',
        description='Advanced template with metadata',
        template='Process {data_type} data using {method}',
        instructions=['Validate input', 'Apply processing', 'Return results'],
        category='data_processing',
        version='3.0',
        metadata={
            'author': 'AI Toolkit Team',
            'created_date': '2025-01-13',
            'complexity': 'intermediate',
            'use_cases': ['data analysis', 'batch processing']
        }
    )
    
    # Template validation
    try:
        manager.validate_template_variables('advanced_template', data_type='CSV', method='pandas')
        print("✅ Template variables validation passed")
    except ValueError as e:
        print(f"❌ Validation failed: {e}")
    
    # Get template variables
    variables = manager.get_template_variables('advanced_template')
    print(f"📋 Template variables: {variables}")
    
    # Advanced search
    search_results = manager.search_templates('processing', ['name', 'description', 'category'])
    print(f"🔍 Search results: {len(search_results)} templates found")
    
    # Template info with metadata
    info = manager.get_template_info('advanced_template')
    print(f"📊 Template metadata: {info['metadata']}")
    
    # Export templates
    with tempfile.TemporaryDirectory() as temp_dir:
        export_path = Path(temp_dir)
        manager.export_templates(export_path, format='yaml')
        exported_files = list(export_path.glob('*.yaml'))
        print(f"📤 Exported {len(exported_files)} templates")
    
    print()


def main():
    """Run all examples."""
    print("🎓 AI Toolkit Prompt Management Examples")
    print("=" * 60)
    print()
    
    examples = [
        basic_template_usage,
        prompt_manager_usage,
        file_operations_example,
        langchain_integration_example,
        advanced_features_example,
    ]
    
    for example in examples:
        try:
            example()
        except Exception as e:
            print(f"❌ Example {example.__name__} failed: {e}")
            print()
    
    print("🎉 Prompt Management Examples Complete!")
    print()
    print("💡 Key Features Demonstrated:")
    print("   ✅ Multiple template types (Simple, Chat, System, Few-shot)")
    print("   ✅ Template creation, rendering, and validation")
    print("   ✅ File loading and saving (YAML, JSON, TXT)")
    print("   ✅ Template management and organization")
    print("   ✅ Search and categorization")
    print("   ✅ LangChain integration")
    print("   ✅ Advanced features (cloning, metadata, export)")


if __name__ == "__main__":
    main()