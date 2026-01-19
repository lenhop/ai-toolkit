"""
Stream Handler - Simple interface for streaming LLM output.

Based on LangChain streaming patterns.
Official Documentation: https://docs.langchain.com/oss/python/langchain/streaming/overview

Classes:
    StreamHandler: Simple wrapper for model.stream() with convenience methods
"""

from typing import Any, Callable, Dict, Generator, Optional
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage


class StreamHandler:
    """
    Simple interface for streaming operations.
    
    Provides convenience methods for common streaming patterns.
    For most cases, you can just use model.stream() directly.
    
    Example:
        handler = StreamHandler(model)
        
        # Simple token streaming
        for token in handler.stream_tokens("Hello!"):
            print(token, end="", flush=True)
    """
    
    def __init__(self, model: BaseChatModel):
        """
        Initialize the stream handler.
        
        Args:
            model: LangChain chat model to stream from
        """
        self.model = model
    
    def stream_tokens(
        self,
        messages: Any,
        **kwargs
    ) -> Generator[str, None, None]:
        """
        Stream LLM tokens as they arrive.
        
        This is a convenience wrapper around model.stream() that:
        - Normalizes message input (string, dict, or list)
        - Extracts content from chunks
        - Yields plain text tokens
        
        Args:
            messages: Input messages (string, dict, or list of messages)
            **kwargs: Additional arguments for model.stream()
        
        Yields:
            str: Each token as it arrives
        
        Example:
            for token in handler.stream_tokens("What is AI?"):
                print(token, end="", flush=True)
        """
        # Normalize messages to list format
        if isinstance(messages, str):
            messages = [HumanMessage(content=messages)]
        elif isinstance(messages, dict):
            messages = [messages]
        
        # Stream tokens
        for chunk in self.model.stream(messages, **kwargs):
            if hasattr(chunk, 'content') and chunk.content:
                yield chunk.content
    
    def stream_tool_calls(
        self,
        messages: Any,
        on_partial: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        on_complete: Optional[Callable[[str, Any], None]] = None,
        **kwargs
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Stream tool call construction and execution.
        
        Yields both partial tool calls (as they're built) and complete results.
        
        Args:
            messages: Input messages
            on_partial: Callback for partial tool calls (tool_name, partial_args)
            on_complete: Callback for completed tool calls (tool_name, result)
            **kwargs: Additional arguments for model.stream()
        
        Yields:
            dict: Stream events with 'type', 'tool_name', and 'data'
        
        Example:
            for event in handler.stream_tool_calls(
                "What's the weather?",
                on_partial=lambda name, args: print(f"Building {name}..."),
                on_complete=lambda name, result: print(f"Got: {result}")
            ):
                if event['type'] == 'token':
                    print(event['data'], end="")
        """
        # Normalize messages
        if isinstance(messages, str):
            messages = [HumanMessage(content=messages)]
        elif isinstance(messages, dict):
            messages = [messages]
        
        # Track tool call state
        current_tool = None
        partial_args = {}
        
        # Stream chunks
        for chunk in self.model.stream(messages, **kwargs):
            # Handle tool call chunks
            if hasattr(chunk, 'tool_call_chunks') and chunk.tool_call_chunks:
                for tool_chunk in chunk.tool_call_chunks:
                    tool_name = tool_chunk.get('name')
                    args = tool_chunk.get('args', '')
                    
                    if tool_name:
                        current_tool = tool_name
                        partial_args = {}
                    
                    if args and current_tool:
                        # Accumulate partial args
                        if isinstance(args, str):
                            partial_args['_raw'] = partial_args.get('_raw', '') + args
                        else:
                            partial_args.update(args)
                        
                        if on_partial:
                            on_partial(current_tool, partial_args)
                        
                        yield {
                            'type': 'tool_partial',
                            'tool_name': current_tool,
                            'data': partial_args
                        }
            
            # Handle complete tool calls
            if hasattr(chunk, 'tool_calls') and chunk.tool_calls:
                for tool_call in chunk.tool_calls:
                    tool_name = tool_call.get('name')
                    result = tool_call.get('args', {})
                    
                    if on_complete:
                        on_complete(tool_name, result)
                    
                    yield {
                        'type': 'tool_complete',
                        'tool_name': tool_name,
                        'data': result
                    }
            
            # Handle regular content
            if hasattr(chunk, 'content') and chunk.content:
                yield {
                    'type': 'token',
                    'tool_name': None,
                    'data': chunk.content
                }
