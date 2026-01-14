"""
Prompt template classes for different types of prompts.

This module provides various prompt template classes for managing and rendering
prompts with different structures and use cases.

Classes:
    BasePromptTemplate: Abstract base class for all prompt templates
        - Defines interface for prompt templates
        - Handles variable validation and rendering
        
        Methods:
            __init__(template, variables, name, description): Initialize template
            render(**kwargs): Render template with variables (abstract)
            validate_variables(**kwargs): Validate provided variables
            get_variables(): Get list of required variables
            to_dict(): Convert template to dictionary
            from_dict(data): Create template from dictionary (class method)
    
    SimplePromptTemplate: Simple string-based prompt template
        - Basic template with variable substitution
        - Uses Python string formatting
        
        Methods:
            render(**kwargs): Render template with string formatting
    
    ChatPromptTemplate: Chat-based prompt template
        - Supports multi-turn conversations
        - Compatible with LangChain ChatPromptTemplate
        
        Methods:
            render(**kwargs): Render chat template
            add_message(role, content): Add message to template
            to_langchain(): Convert to LangChain ChatPromptTemplate
    
    SystemPromptTemplate: System message prompt template
        - Specialized for system instructions
        - Prepends system role automatically
        
        Methods:
            render(**kwargs): Render system prompt
    
    FewShotPromptTemplate: Few-shot learning prompt template
        - Includes examples for few-shot learning
        - Formats examples with template
        
        Methods:
            render(**kwargs): Render few-shot prompt with examples
            add_example(input, output): Add example to template
            set_examples(examples): Set all examples at once
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field, validator
from langchain_core.prompts import ChatPromptTemplate as LangChainChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.prompts.few_shot import FewShotPromptTemplate
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.example_selectors import BaseExampleSelector
import re


class BasePromptTemplate(BaseModel, ABC):
    """Abstract base class for prompt templates."""
    
    name: str = Field(..., description="Template name")
    description: str = Field(..., description="Template description")
    template: str = Field(..., description="Template content")
    variables: List[str] = Field(default_factory=list, description="Template variables")
    category: str = Field(default="general", description="Template category")
    version: str = Field(default="1.0", description="Template version")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    class Config:
        """Pydantic configuration."""
        extra = "forbid"
        validate_assignment = True
    
    @validator('template')
    def validate_template(cls, v):
        """Validate template is not empty."""
        if not v or not v.strip():
            raise ValueError("Template content cannot be empty")
        return v.strip()
    
    def __init__(self, **data):
        # Extract variables from template if not provided
        if 'variables' not in data or not data['variables']:
            if 'template' in data:
                template = data['template']
                variables = re.findall(r'\{([^}]+)\}', template)
                data['variables'] = list(set(variables))  # Remove duplicates
        super().__init__(**data)
    
    @abstractmethod
    def to_langchain(self) -> Any:
        """Convert to LangChain prompt template."""
        pass
    
    @abstractmethod
    def render(self, **kwargs) -> str:
        """Render template with provided variables."""
        pass
    
    def get_variables(self) -> List[str]:
        """Get list of template variables."""
        return self.variables
    
    def validate_variables(self, **kwargs) -> bool:
        """Validate that all required variables are provided."""
        missing_vars = set(self.variables) - set(kwargs.keys())
        if missing_vars:
            raise ValueError(f"Missing required variables: {missing_vars}")
        return True


class ChatPromptTemplate(BasePromptTemplate):
    """Chat prompt template for conversational AI."""
    
    system_message: Optional[str] = Field(default=None, description="System message")
    human_message: str = Field(..., description="Human message template")
    ai_message: Optional[str] = Field(default=None, description="AI message template")
    
    def __init__(self, **data):
        # If template is provided but human_message is not, use template as human_message
        if 'template' in data and 'human_message' not in data:
            data['human_message'] = data['template']
        elif 'human_message' in data and 'template' not in data:
            data['template'] = data['human_message']
        super().__init__(**data)
    
    def to_langchain(self) -> LangChainChatPromptTemplate:
        """Convert to LangChain ChatPromptTemplate."""
        messages = []
        
        if self.system_message:
            messages.append(SystemMessagePromptTemplate.from_template(self.system_message))
        
        messages.append(HumanMessagePromptTemplate.from_template(self.human_message))
        
        if self.ai_message:
            from langchain_core.prompts import AIMessagePromptTemplate
            messages.append(AIMessagePromptTemplate.from_template(self.ai_message))
        
        return LangChainChatPromptTemplate.from_messages(messages)
    
    def render(self, **kwargs) -> str:
        """Render chat template."""
        self.validate_variables(**kwargs)
        
        result = []
        
        if self.system_message:
            system_rendered = self.system_message.format(**kwargs)
            result.append(f"System: {system_rendered}")
        
        human_rendered = self.human_message.format(**kwargs)
        result.append(f"Human: {human_rendered}")
        
        if self.ai_message:
            ai_rendered = self.ai_message.format(**kwargs)
            result.append(f"AI: {ai_rendered}")
        
        return "\n\n".join(result)


class SystemPromptTemplate(BasePromptTemplate):
    """System prompt template for system instructions."""
    
    instructions: List[str] = Field(default_factory=list, description="System instructions")
    constraints: List[str] = Field(default_factory=list, description="System constraints")
    examples: List[Dict[str, str]] = Field(default_factory=list, description="Example interactions")
    
    def to_langchain(self) -> SystemMessagePromptTemplate:
        """Convert to LangChain SystemMessagePromptTemplate."""
        return SystemMessagePromptTemplate.from_template(self.template)
    
    def render(self, **kwargs) -> str:
        """Render system template."""
        self.validate_variables(**kwargs)
        
        # Start with the main template
        result = [self.template.format(**kwargs)]
        
        # Add instructions if provided
        if self.instructions:
            result.append("\nInstructions:")
            for i, instruction in enumerate(self.instructions, 1):
                result.append(f"{i}. {instruction}")
        
        # Add constraints if provided
        if self.constraints:
            result.append("\nConstraints:")
            for constraint in self.constraints:
                result.append(f"- {constraint}")
        
        # Add examples if provided
        if self.examples:
            result.append("\nExamples:")
            for i, example in enumerate(self.examples, 1):
                result.append(f"Example {i}:")
                for key, value in example.items():
                    result.append(f"  {key}: {value}")
        
        return "\n".join(result)


class FewShotPromptTemplate(BasePromptTemplate):
    """Few-shot prompt template with examples."""
    
    examples: List[Dict[str, str]] = Field(..., description="Training examples")
    example_template: str = Field(..., description="Template for each example")
    prefix: str = Field(default="", description="Prefix before examples")
    suffix: str = Field(..., description="Suffix after examples")
    input_variables: List[str] = Field(default_factory=list, description="Input variables for the final prompt")
    example_separator: str = Field(default="\n\n", description="Separator between examples")
    
    def __init__(self, **data):
        # If template is provided but suffix is not, use template as suffix
        if 'template' in data and 'suffix' not in data:
            data['suffix'] = data['template']
        elif 'suffix' in data and 'template' not in data:
            data['template'] = data['suffix']
        super().__init__(**data)
    
    @validator('examples')
    def validate_examples(cls, v):
        """Validate examples are not empty."""
        if not v:
            raise ValueError("Examples list cannot be empty for few-shot template")
        return v
    
    def to_langchain(self) -> FewShotPromptTemplate:
        """Convert to LangChain FewShotPromptTemplate."""
        example_prompt = PromptTemplate(
            input_variables=list(self.examples[0].keys()) if self.examples else [],
            template=self.example_template
        )
        
        return FewShotPromptTemplate(
            examples=self.examples,
            example_prompt=example_prompt,
            prefix=self.prefix,
            suffix=self.suffix,
            input_variables=self.input_variables or self.variables,
            example_separator=self.example_separator
        )
    
    def render(self, **kwargs) -> str:
        """Render few-shot template."""
        self.validate_variables(**kwargs)
        
        result = []
        
        # Add prefix
        if self.prefix:
            result.append(self.prefix)
        
        # Add examples
        rendered_examples = []
        for example in self.examples:
            rendered_example = self.example_template.format(**example)
            rendered_examples.append(rendered_example)
        
        if rendered_examples:
            result.append(self.example_separator.join(rendered_examples))
        
        # Add suffix with variables
        if self.suffix:
            suffix_rendered = self.suffix.format(**kwargs)
            result.append(suffix_rendered)
        
        return "\n\n".join(result)


class SimplePromptTemplate(BasePromptTemplate):
    """Simple string-based prompt template."""
    
    def to_langchain(self) -> PromptTemplate:
        """Convert to LangChain PromptTemplate."""
        return PromptTemplate(
            input_variables=self.variables,
            template=self.template
        )
    
    def render(self, **kwargs) -> str:
        """Render simple template."""
        self.validate_variables(**kwargs)
        return self.template.format(**kwargs)


def create_template(template_type: str, **kwargs) -> BasePromptTemplate:
    """
    Factory function to create prompt templates.
    
    Args:
        template_type: Type of template ('chat', 'system', 'few_shot', 'simple')
        **kwargs: Template configuration
        
    Returns:
        Prompt template instance
        
    Raises:
        ValueError: If template type is not supported
    """
    template_classes = {
        'chat': ChatPromptTemplate,
        'system': SystemPromptTemplate,
        'few_shot': FewShotPromptTemplate,
        'simple': SimplePromptTemplate,
    }
    
    if template_type not in template_classes:
        raise ValueError(f"Unsupported template type: {template_type}. "
                        f"Supported types: {list(template_classes.keys())}")
    
    template_class = template_classes[template_type]
    return template_class(**kwargs)


def detect_template_type(template_data: Dict[str, Any]) -> str:
    """
    Detect template type from template data.
    
    Args:
        template_data: Template configuration dictionary
        
    Returns:
        Detected template type
    """
    # Check for chat template indicators
    if any(key in template_data for key in ['system_message', 'human_message', 'ai_message']):
        return 'chat'
    
    # Check for few-shot template indicators
    if 'examples' in template_data and 'example_template' in template_data:
        return 'few_shot'
    
    # Check for system template indicators
    if any(key in template_data for key in ['instructions', 'constraints']) or \
       template_data.get('category') == 'system':
        return 'system'
    
    # Default to simple template
    return 'simple'