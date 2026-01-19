"""
Simplified Model Manager - Single file for all model operations.

Supports OpenAI-compatible models and native SDKs for non-compatible ones.

"""

import os
from typing import Dict, Optional, Any
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult

try:
    from langchain_openai import ChatOpenAI
except ImportError:
    from langchain_community.chat_models import ChatOpenAI

from zhipuai import ZhipuAI


class ModelManager:
    """
    Simplified model manager - handles all providers in one place.
    
    Pre-configured providers (with defaults):
        - deepseek: OpenAI-compatible
        - qwen: OpenAI-compatible  
        - glm: Native SDK (zhipuai)
    
    Custom providers:
        For providers not in the pre-configured list, you must provide:
        - api_key: API key (required)
        - model: Model name (required)
        - base_url: Base URL for API (required for OpenAI-compatible)
        - openai_compatible: Whether provider is OpenAI-compatible (default: True)
    """
    
    # Pre-configured provider configurations
    PROVIDERS = {
        'deepseek': {
            'base_url': 'https://api.deepseek.com',
            'default_model': 'deepseek-chat',
            'openai_compatible': True
        },
        'qwen': {
            'base_url': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
            'default_model': 'qwen-turbo',
            'openai_compatible': True
        },
        'glm': {
            'base_url': 'https://open.bigmodel.cn/api/paas/v4',
            'default_model': 'glm-4',
            'openai_compatible': False
        }
    }
    
    def __init__(self):
        """Initialize model manager with empty cache."""
        self._cache: Dict[str, BaseChatModel] = {}
    
    def create_model(
        self,
        provider: str,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        openai_compatible: Optional[bool] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs
    ) -> BaseChatModel:
        """
        Create a model instance.
        
        Args:
            provider: Provider name (pre-configured: 'deepseek', 'qwen', 'glm', or any custom provider)
            model: Model name (required for custom providers, optional for pre-configured)
            api_key: API key (required for custom providers, optional for pre-configured - loads from env)
            base_url: Base URL for API (required for custom providers if openai_compatible=True)
            openai_compatible: Whether provider is OpenAI-compatible (default: True for custom providers)
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum tokens in response
            **kwargs: Additional parameters passed to model constructor
            
        Returns:
            Model instance ready to use
            
        Example:
            >>> manager = ModelManager()
            >>> # Pre-configured provider - auto-load from environment
            >>> model = manager.create_model("deepseek")
            >>> # Pre-configured provider - with API key
            >>> model = manager.create_model("deepseek", api_key="sk-...")
            >>> # Custom OpenAI-compatible provider
            >>> model = manager.create_model(
            ...     "custom_provider",
            ...     model="custom-model",
            ...     api_key="sk-...",
            ...     base_url="https://api.custom.com"
            ... )
            >>> # Custom provider with explicit compatibility flag
            >>> model = manager.create_model(
            ...     "another_provider",
            ...     model="model-name",
            ...     api_key="key-...",
            ...     base_url="https://api.example.com",
            ...     openai_compatible=True
            ... )
        """
        provider = provider.lower()
        
        # Check if provider is pre-configured
        is_preconfigured = provider in self.PROVIDERS
        
        if is_preconfigured:
            # Use pre-configured settings
            config = self.PROVIDERS[provider]
            model_name = model or config['default_model']
            provider_base_url = config['base_url']
            provider_openai_compatible = config['openai_compatible']
            
            # Load API key from parameter or environment
            if api_key is None:
                api_key = os.getenv(f"{provider.upper()}_API_KEY")
            
            if not api_key:
                raise ValueError(
                    f"Missing API key. Provide api_key parameter or set "
                    f"{provider.upper()}_API_KEY environment variable"
                )
        else:
            # Custom provider - validate required parameters
            if not api_key:
                # Try environment variable as fallback
                api_key = os.getenv(f"{provider.upper()}_API_KEY")
                if not api_key:
                    raise ValueError(
                        f"Missing API key for custom provider '{provider}'. "
                        f"Provide api_key parameter or set {provider.upper()}_API_KEY environment variable"
                    )
            
            if not model:
                raise ValueError(
                    f"Missing model name for custom provider '{provider}'. "
                    f"Model name is required for custom providers."
                )
            model_name = model
            
            # For custom providers, base_url is required if OpenAI-compatible
            if openai_compatible is None:
                # Default to True for custom providers
                provider_openai_compatible = True
            else:
                provider_openai_compatible = openai_compatible
            
            if provider_openai_compatible:
                if not base_url:
                    raise ValueError(
                        f"Missing base_url for custom OpenAI-compatible provider '{provider}'. "
                        f"base_url is required for OpenAI-compatible providers."
                    )
                provider_base_url = base_url
            else:
                # For non-OpenAI-compatible custom providers, base_url is optional
                # (may be used for other purposes or not needed)
                provider_base_url = base_url
        
        # Create cache key
        cache_key = f"{provider}:{model_name}"
        
        # Return cached model if exists
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # Create model based on compatibility
        if provider_openai_compatible:
            # Use OpenAI-compatible interface
            model_instance = ChatOpenAI(
                api_key=api_key,
                base_url=provider_base_url,
                model=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
        else:
            # Use native SDK (currently only GLM is supported)
            if provider == 'glm':
                model_instance = GLMChatModel(
                    api_key=api_key,
                    model=model_name,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs
                )
            else:
                raise ValueError(
                    f"Native SDK not implemented for provider '{provider}'. "
                    f"Only 'glm' provider supports native SDK. "
                    f"For other providers, set openai_compatible=True and provide base_url."
                )
        
        # Cache and return
        self._cache[cache_key] = model_instance
        return model_instance
    
    def get_model(self, provider: str, model: Optional[str] = None) -> Optional[BaseChatModel]:
        """
        Get cached model instance.
        
        Args:
            provider: Provider name
            model: Model name (optional)
            
        Returns:
            Cached model or None
        """
        provider = provider.lower()
        if model:
            cache_key = f"{provider}:{model}"
        else:
            # Find first model for this provider
            for key in self._cache.keys():
                if key.startswith(f"{provider}:"):
                    return self._cache[key]
            return None
        
        return self._cache.get(cache_key)
    
    def list_providers(self) -> list:
        """
        List pre-configured providers.
        
        Note: Custom providers can be used even if not listed here.
        They just require explicit parameters when creating models.
        
        Returns:
            List of pre-configured provider names
        """
        return list(self.PROVIDERS.keys())
    
    def clear_cache(self):
        """Clear all cached models."""
        self._cache.clear()


class GLMChatModel(BaseChatModel):
    """
    LangChain wrapper for GLM models using native zhipuai SDK.
    
    GLM doesn't support OpenAI format well, so we use their native SDK.
    """
    
    api_key: str
    model: str = "glm-4"
    temperature: float = 0.7
    max_tokens: int = 4096
    client: Any = None
    
    class Config:
        """Pydantic config."""
        arbitrary_types_allowed = True
    
    def __init__(
        self,
        api_key: str,
        model: str = "glm-4",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs
    ):
        """Initialize GLM model."""
        super().__init__(
            api_key=api_key,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            client=ZhipuAI(api_key=api_key),
            **kwargs
        )
    
    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        """Generate response using GLM API."""
        # Convert LangChain messages to GLM format
        glm_messages = []
        for msg in messages:
            if isinstance(msg, AIMessage):
                role = 'assistant'
            elif isinstance(msg, SystemMessage):
                role = 'system'
            else:
                role = 'user'
            
            glm_messages.append({
                'role': role,
                'content': str(msg.content)
            })
        
        # Call GLM API
        response = self.client.chat.completions.create(
            model=self.model,
            messages=glm_messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            **kwargs
        )
        
        # Convert to LangChain format
        content = response.choices[0].message.content
        message = AIMessage(content=content)
        generation = ChatGeneration(message=message)
        
        return ChatResult(generations=[generation])
    
    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        """Async generate (wraps sync for now)."""
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self._generate(messages, stop, run_manager, **kwargs)
        )
    
    @property
    def _llm_type(self) -> str:
        """Return model type."""
        return "glm-chat"
    
    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Return identifying parameters."""
        return {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
