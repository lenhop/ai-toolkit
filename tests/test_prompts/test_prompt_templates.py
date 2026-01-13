"""
Tests for prompt template classes.
"""

import pytest
from ai_toolkit.prompts.prompt_templates import (
    ChatPromptTemplate,
    SystemPromptTemplate,
    FewShotPromptTemplate,
    SimplePromptTemplate,
    create_template,
    detect_template_type
)


class TestChatPromptTemplate:
    """Test ChatPromptTemplate class."""
    
    def test_create_chat_template(self):
        """Test creating a chat prompt template."""
        template = ChatPromptTemplate(
            name="test_chat",
            description="Test chat template",
            template="Hello {name}",
            human_message="Hello {name}",
            system_message="You are a helpful assistant."
        )
        
        assert template.name == "test_chat"
        assert template.human_message == "Hello {name}"
        assert template.system_message == "You are a helpful assistant."
        assert "name" in template.variables
    
    def test_chat_template_render(self):
        """Test rendering chat template."""
        template = ChatPromptTemplate(
            name="test_chat",
            description="Test chat template",
            template="Hello {name}",
            human_message="Hello {name}",
            system_message="You are a helpful assistant."
        )
        
        rendered = template.render(name="Alice")
        
        assert "System: You are a helpful assistant." in rendered
        assert "Human: Hello Alice" in rendered
    
    def test_chat_template_to_langchain(self):
        """Test converting chat template to LangChain format."""
        template = ChatPromptTemplate(
            name="test_chat",
            description="Test chat template",
            template="Hello {name}",
            human_message="Hello {name}",
            system_message="You are a helpful assistant."
        )
        
        langchain_template = template.to_langchain()
        assert langchain_template is not None
        # LangChain ChatPromptTemplate should have messages
        assert hasattr(langchain_template, 'messages')
    
    def test_chat_template_missing_variables(self):
        """Test error when required variables are missing."""
        template = ChatPromptTemplate(
            name="test_chat",
            description="Test chat template",
            template="Hello {name}",
            human_message="Hello {name}"
        )
        
        with pytest.raises(ValueError, match="Missing required variables"):
            template.render()


class TestSystemPromptTemplate:
    """Test SystemPromptTemplate class."""
    
    def test_create_system_template(self):
        """Test creating a system prompt template."""
        template = SystemPromptTemplate(
            name="test_system",
            description="Test system template",
            template="You are a {role}",
            instructions=["Be helpful", "Be accurate"],
            constraints=["No harmful content"]
        )
        
        assert template.name == "test_system"
        assert template.instructions == ["Be helpful", "Be accurate"]
        assert template.constraints == ["No harmful content"]
        assert "role" in template.variables
    
    def test_system_template_render(self):
        """Test rendering system template."""
        template = SystemPromptTemplate(
            name="test_system",
            description="Test system template",
            template="You are a {role}",
            instructions=["Be helpful", "Be accurate"]
        )
        
        rendered = template.render(role="assistant")
        
        assert "You are a assistant" in rendered
        assert "Instructions:" in rendered
        assert "1. Be helpful" in rendered
        assert "2. Be accurate" in rendered


class TestFewShotPromptTemplate:
    """Test FewShotPromptTemplate class."""
    
    def test_create_few_shot_template(self):
        """Test creating a few-shot prompt template."""
        examples = [
            {"input": "Hello", "output": "Hi there!"},
            {"input": "Goodbye", "output": "See you later!"}
        ]
        
        template = FewShotPromptTemplate(
            name="test_few_shot",
            description="Test few-shot template",
            template="Input: {input}\nOutput:",
            examples=examples,
            example_template="Input: {input}\nOutput: {output}",
            suffix="Input: {input}\nOutput:"
        )
        
        assert template.name == "test_few_shot"
        assert len(template.examples) == 2
        assert template.example_template == "Input: {input}\nOutput: {output}"
    
    def test_few_shot_template_render(self):
        """Test rendering few-shot template."""
        examples = [
            {"input": "Hello", "output": "Hi there!"}
        ]
        
        template = FewShotPromptTemplate(
            name="test_few_shot",
            description="Test few-shot template",
            template="Input: {input}\nOutput:",
            examples=examples,
            example_template="Input: {input}\nOutput: {output}",
            suffix="Input: {input}\nOutput:"
        )
        
        rendered = template.render(input="How are you?")
        
        assert "Input: Hello\nOutput: Hi there!" in rendered
        assert "Input: How are you?\nOutput:" in rendered
    
    def test_few_shot_template_empty_examples(self):
        """Test error when examples list is empty."""
        with pytest.raises(ValueError, match="Examples list cannot be empty"):
            FewShotPromptTemplate(
                name="test_few_shot",
                description="Test few-shot template",
                template="Input: {input}\nOutput:",
                examples=[],
                example_template="Input: {input}\nOutput: {output}",
                suffix="Input: {input}\nOutput:"
            )


class TestSimplePromptTemplate:
    """Test SimplePromptTemplate class."""
    
    def test_create_simple_template(self):
        """Test creating a simple prompt template."""
        template = SimplePromptTemplate(
            name="test_simple",
            description="Test simple template",
            template="Hello {name}, you are {age} years old."
        )
        
        assert template.name == "test_simple"
        assert template.template == "Hello {name}, you are {age} years old."
        assert set(template.variables) == {"name", "age"}
    
    def test_simple_template_render(self):
        """Test rendering simple template."""
        template = SimplePromptTemplate(
            name="test_simple",
            description="Test simple template",
            template="Hello {name}, you are {age} years old."
        )
        
        rendered = template.render(name="Alice", age=25)
        assert rendered == "Hello Alice, you are 25 years old."
    
    def test_simple_template_to_langchain(self):
        """Test converting simple template to LangChain format."""
        template = SimplePromptTemplate(
            name="test_simple",
            description="Test simple template",
            template="Hello {name}"
        )
        
        langchain_template = template.to_langchain()
        assert langchain_template is not None
        assert hasattr(langchain_template, 'template')


class TestTemplateFactory:
    """Test template factory functions."""
    
    def test_create_template_chat(self):
        """Test creating chat template via factory."""
        template = create_template(
            'chat',
            name="test_chat",
            description="Test chat template",
            template="Hello {name}",
            human_message="Hello {name}"
        )
        
        assert isinstance(template, ChatPromptTemplate)
        assert template.name == "test_chat"
    
    def test_create_template_system(self):
        """Test creating system template via factory."""
        template = create_template(
            'system',
            name="test_system",
            description="Test system template",
            template="You are a {role}"
        )
        
        assert isinstance(template, SystemPromptTemplate)
        assert template.name == "test_system"
    
    def test_create_template_few_shot(self):
        """Test creating few-shot template via factory."""
        template = create_template(
            'few_shot',
            name="test_few_shot",
            description="Test few-shot template",
            template="Input: {input}\nOutput:",
            examples=[{"input": "Hello", "output": "Hi"}],
            example_template="Input: {input}\nOutput: {output}",
            suffix="Input: {input}\nOutput:"
        )
        
        assert isinstance(template, FewShotPromptTemplate)
        assert template.name == "test_few_shot"
    
    def test_create_template_simple(self):
        """Test creating simple template via factory."""
        template = create_template(
            'simple',
            name="test_simple",
            description="Test simple template",
            template="Hello {name}"
        )
        
        assert isinstance(template, SimplePromptTemplate)
        assert template.name == "test_simple"
    
    def test_create_template_invalid_type(self):
        """Test error with invalid template type."""
        with pytest.raises(ValueError, match="Unsupported template type"):
            create_template(
                'invalid_type',
                name="test",
                description="Test",
                template="Hello"
            )


class TestTemplateDetection:
    """Test template type detection."""
    
    def test_detect_chat_template(self):
        """Test detecting chat template type."""
        template_data = {
            'name': 'test',
            'template': 'Hello',
            'human_message': 'Hello {name}',
            'system_message': 'You are helpful'
        }
        
        detected_type = detect_template_type(template_data)
        assert detected_type == 'chat'
    
    def test_detect_few_shot_template(self):
        """Test detecting few-shot template type."""
        template_data = {
            'name': 'test',
            'template': 'Hello',
            'examples': [{'input': 'hi', 'output': 'hello'}],
            'example_template': 'Input: {input}\nOutput: {output}'
        }
        
        detected_type = detect_template_type(template_data)
        assert detected_type == 'few_shot'
    
    def test_detect_system_template(self):
        """Test detecting system template type."""
        template_data = {
            'name': 'test',
            'template': 'You are helpful',
            'instructions': ['Be nice'],
            'category': 'system'
        }
        
        detected_type = detect_template_type(template_data)
        assert detected_type == 'system'
    
    def test_detect_simple_template(self):
        """Test detecting simple template type (default)."""
        template_data = {
            'name': 'test',
            'template': 'Hello {name}'
        }
        
        detected_type = detect_template_type(template_data)
        assert detected_type == 'simple'


class TestTemplateValidation:
    """Test template validation."""
    
    def test_empty_template_content(self):
        """Test error with empty template content."""
        with pytest.raises(ValueError, match="Template content cannot be empty"):
            SimplePromptTemplate(
                name="test",
                description="Test",
                template=""
            )
    
    def test_variable_extraction(self):
        """Test automatic variable extraction from template."""
        template = SimplePromptTemplate(
            name="test",
            description="Test",
            template="Hello {name}, you are {age} years old. Welcome {name}!"
        )
        
        # Should extract unique variables
        assert set(template.variables) == {"name", "age"}
    
    def test_validate_variables_success(self):
        """Test successful variable validation."""
        template = SimplePromptTemplate(
            name="test",
            description="Test",
            template="Hello {name}"
        )
        
        assert template.validate_variables(name="Alice") is True
    
    def test_validate_variables_missing(self):
        """Test error when variables are missing."""
        template = SimplePromptTemplate(
            name="test",
            description="Test",
            template="Hello {name}, you are {age} years old."
        )
        
        with pytest.raises(ValueError, match="Missing required variables"):
            template.validate_variables(name="Alice")  # missing 'age'