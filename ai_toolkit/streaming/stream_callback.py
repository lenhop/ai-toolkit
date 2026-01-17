"""
Stream Callback for LangChain models.

Simple callback handler for processing streaming output from LangChain models.
Based on official LangChain streaming patterns.

Official Documentation:
    https://docs.langchain.com/oss/python/langchain-streaming

Classes:
    StreamCallback: Basic callback for token-by-token streaming
        - on_llm_new_token(): Handle each token
        - on_llm_end(): Handle completion
        - get_accumulated_text(): Get full response

Usage:
    >>> callback = StreamCallback()
    >>> for chunk in model.stream("Hello", config={"callbacks": [callback]}):
    ...     pass  # Callback handles tokens
    >>> print(callback.get_accumulated_text())
"""

from typing import Any, Dict, List, Optional, Callable
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult


class StreamCallback(BaseCallbackHandler):
    """
    Simple callback handler for streaming output.
    
    Captures tokens as they stream and provides accumulated text.
    """
    
    def __init__(self,
                 on_token: Optional[Callable[[str], None]] = None,
                 on_complete: Optional[Callable[[str], None]] = None,
                 verbose: bool = False):
        """
        Initialize the callback.
        
        Args:
            on_token: Function called for each token (optional)
            on_complete: Function called when complete (optional)
            verbose: Print tokens as they arrive
        """
        super().__init__()
        self.on_token = on_token
        self.on_complete = on_complete
        self.verbose = verbose
        self._accumulated_text = ""
    
    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Called when a new token is generated."""
        self._accumulated_text += token
        
        if self.on_token:
            self.on_token(token)
        
        if self.verbose:
            print(token, end="", flush=True)
    
    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Called when LLM completes."""
        if self.on_complete:
            self.on_complete(self._accumulated_text)
        
        if self.verbose:
            print()  # New line
    
    def get_accumulated_text(self) -> str:
        """Get the full accumulated text."""
        return self._accumulated_text
    
    def reset(self) -> None:
        """Reset the accumulated text."""
        self._accumulated_text = ""
