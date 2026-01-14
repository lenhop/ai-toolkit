"""
Model provider implementations for different AI services.

This module provides provider implementations for various AI model services,
including DeepSeek, Qwen (Alibaba), and GLM (Zhipu AI).

Classes:
    BaseModelProvider: Abstract base class for model providers
        - Defines interface for all model providers
        - Handles model creation and caching
        - Validates provider-specific configurations
        
        Methods:
            __init__(config): Initialize provider with configuration
            create_model(): Create and return a model instance (abstract)
            validate_config(): Validate provider-specific configuration (abstract)
            get_model(): Get or create model instance (lazy loading)
            reset_model(): Reset the cached model instance
            provider_name: Get provider name property
    
    DeepSeekProvider: DeepSeek model provider
        - Supports deepseek-chat and deepseek-coder models
        - Uses OpenAI-compatible API
        
        Methods:
            validate_config(): Validate DeepSeek configuration
            create_model(): Create DeepSeek model instance
    
    QwenProvider: Qwen (Alibaba) model provider
        - Supports qwen-turbo, qwen-plus, qwen-max models
        - Uses DashScope API
        
        Methods:
            validate_config(): Validate Qwen configuration
            create_model(): Create Qwen model instance
    
    GLMProvider: GLM (Zhipu AI) model provider
        - Supports glm-4, glm-4-air, glm-3-turbo models
        - Uses Zhipu AI API
        
        Methods:
            validate_config(): Validate GLM configuration
            create_model(): Create GLM model instance

Functions:
    create_provider(provider_name, config): Factory function to create provider instances
    get_provider_class(provider_name): Get provider class by name
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_community.chat_models import ChatOpenAI
from zhipuai import ZhipuAI

from .model_config import ModelConfig


class BaseModelProvider(ABC):
    """Abstract base class for model providers."""
    
    def __init__(self, config: ModelConfig):
        """
        Initialize the provider with configuration.
        
        Args:
            config: Model configuration
        """
        self.config = config
        self._model: Optional[BaseLanguageModel] = None
    
    @abstractmethod
    def create_model(self) -> BaseLanguageModel:
        """
        Create and return a model instance.
        
        Returns:
            Configured model instance
        """
        pass
    
    @abstractmethod
    def validate_config(self) -> bool:
        """
        Validate provider-specific configuration.
        
        Returns:
            True if configuration is valid
            
        Raises:
            ValueError: If configuration is invalid
        """
        pass
    
    def get_model(self) -> BaseLanguageModel:
        """
        Get or create model instance (lazy loading).
        
        Returns:
            Model instance
        """
        if self._model is None:
            self._model = self.create_model()
        return self._model
    
    def reset_model(self) -> None:
        """Reset the cached model instance."""
        self._model = None
    
    @property
    def provider_name(self) -> str:
        """Get provider name."""
        return self.__class__.__name__.replace('Provider', '').lower()


class DeepSeekProvider(BaseModelProvider):
    """DeepSeek model provider."""
    
    SUPPORTED_MODELS = ['deepseek-chat', 'deepseek-coder']
    DEFAULT_BASE_URL = 'https://api.deepseek.com'
    
    def validate_config(self) -> bool:
        """Validate DeepSeek-specific configuration."""
        if self.config.model not in self.SUPPORTED_MODELS:
            raise ValueError(f"Unsupported DeepSeek model: {self.config.model}. "
                           f"Supported models: {self.SUPPORTED_MODELS}")
        
        if not self.config.api_key.startswith('sk-'):
            raise ValueError("DeepSeek API key should start with 'sk-'")
        
        return True
    
    def create_model(self) -> BaseChatModel:
        """Create DeepSeek model instance."""
        self.validate_config()
        
        return ChatOpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            model=self.config.model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            top_p=self.config.top_p,
            frequency_penalty=self.config.frequency_penalty,
            presence_penalty=self.config.presence_penalty,
            request_timeout=self.config.timeout,
            max_retries=self.config.max_retries,
            **self.config.extra_params
        )


class QwenProvider(BaseModelProvider):
    """Qwen (Tongyi Qianwen) model provider."""
    
    SUPPORTED_MODELS = ['qwen-turbo', 'qwen-plus', 'qwen-max', 'qwen-long']
    DEFAULT_BASE_URL = 'https://dashscope.aliyuncs.com/compatible-mode/v1'
    
    def validate_config(self) -> bool:
        """Validate Qwen-specific configuration."""
        if self.config.model not in self.SUPPORTED_MODELS:
            raise ValueError(f"Unsupported Qwen model: {self.config.model}. "
                           f"Supported models: {self.SUPPORTED_MODELS}")
        
        if not self.config.api_key.startswith('sk-'):
            raise ValueError("Qwen API key should start with 'sk-'")
        
        return True
    
    def create_model(self) -> BaseChatModel:
        """Create Qwen model instance."""
        self.validate_config()
        
        return ChatOpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            model=self.config.model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            top_p=self.config.top_p,
            frequency_penalty=self.config.frequency_penalty,
            presence_penalty=self.config.presence_penalty,
            request_timeout=self.config.timeout,
            max_retries=self.config.max_retries,
            **self.config.extra_params
        )


class GLMProvider(BaseModelProvider):
    """GLM (ChatGLM) model provider using zhipuai library."""
    
    SUPPORTED_MODELS = ['glm-4.6', 'glm-4', 'glm-3-turbo']
    DEFAULT_BASE_URL = 'https://open.bigmodel.cn/api/paas/v4'
    
    def validate_config(self) -> bool:
        """Validate GLM-specific configuration."""
        if self.config.model not in self.SUPPORTED_MODELS:
            raise ValueError(f"Unsupported GLM model: {self.config.model}. "
                           f"Supported models: {self.SUPPORTED_MODELS}")
        
        # GLM API keys have a different format
        if len(self.config.api_key) < 10:
            raise ValueError("GLM API key appears to be invalid (too short)")
        
        return True
    
    def create_model(self) -> 'GLMChatModel':
        """Create GLM model instance."""
        self.validate_config()
        
        return GLMChatModel(
            api_key=self.config.api_key,
            model=self.config.model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            top_p=self.config.top_p,
            timeout=self.config.timeout,
            max_retries=self.config.max_retries,
            **self.config.extra_params
        )


class GLMChatModel(BaseChatModel):
    """
    Custom LangChain-compatible wrapper for GLM models using zhipuai library.
    """
    
    def __init__(
        self,
        api_key: str,
        model: str = "glm-4.6",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        top_p: float = 0.9,
        timeout: int = 60,
        max_retries: int = 3,
        **kwargs
    ):
        """Initialize GLM chat model."""
        super().__init__(**kwargs)
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.timeout = timeout
        self.max_retries = max_retries
        
        # Initialize ZhipuAI client
        self.client = ZhipuAI(api_key=api_key)
    
    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        """Generate response from GLM model."""
        from langchain_core.messages import AIMessage
        from langchain_core.outputs import ChatGeneration, ChatResult
        
        # Convert LangChain messages to GLM format
        glm_messages = []
        for message in messages:
            if hasattr(message, 'content'):
                role = 'user'
                if hasattr(message, 'type'):
                    if message.type == 'ai':
                        role = 'assistant'
                    elif message.type == 'system':
                        role = 'system'
                
                glm_messages.append({
                    'role': role,
                    'content': message.content
                })
        
        try:
            # Make API call to GLM
            response = self.client.chat.completions.create(
                model=self.model,
                messages=glm_messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=self.top_p,
                **kwargs
            )
            
            # Extract response content
            content = response.choices[0].message.content
            
            # Create LangChain-compatible response
            message = AIMessage(content=content)
            generation = ChatGeneration(message=message)
            
            return ChatResult(generations=[generation])
            
        except Exception as e:
            raise ValueError(f"GLM API call failed: {str(e)}")
    
    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        """Async generate (not implemented for GLM yet)."""
        # For now, use sync version
        return self._generate(messages, stop, run_manager, **kwargs)
    
    @property
    def _llm_type(self) -> str:
        """Return type of language model."""
        return "glm-chat"
    
    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get identifying parameters."""
        return {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
        }


def get_provider_class(provider_name: str) -> type:
    """
    Get provider class by name.
    
    Args:
        provider_name: Name of the provider ('deepseek', 'qwen', 'glm')
        
    Returns:
        Provider class
        
    Raises:
        ValueError: If provider is not supported
    """
    providers = {
        'deepseek': DeepSeekProvider,
        'qwen': QwenProvider,
        'glm': GLMProvider,
    }
    
    if provider_name.lower() not in providers:
        raise ValueError(f"Unsupported provider: {provider_name}. "
                        f"Supported providers: {list(providers.keys())}")
    
    return providers[provider_name.lower()]


def create_provider(provider_name: str, config: ModelConfig) -> BaseModelProvider:
    """
    Create provider instance.
    
    Args:
        provider_name: Name of the provider
        config: Model configuration
        
    Returns:
        Provider instance
    """
    provider_class = get_provider_class(provider_name)
    return provider_class(config)