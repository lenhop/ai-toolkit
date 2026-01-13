"""
Stream Callback for handling streaming output from LangChain models.

This module provides callback handlers for processing streaming output
from LangChain models in real-time.
"""

import time
from typing import Any, Dict, List, Optional, Union, Callable
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from langchain_core.messages import BaseMessage

from .stream_handler import StreamHandler, StreamChunk


class StreamCallback(BaseCallbackHandler):
    """
    Callback handler for processing streaming output from LangChain models.
    
    This callback integrates with LangChain's callback system to handle
    streaming tokens and provide real-time processing capabilities.
    """
    
    def __init__(self,
                 stream_handler: Optional[StreamHandler] = None,
                 on_chunk_callback: Optional[Callable[[StreamChunk], None]] = None,
                 on_complete_callback: Optional[Callable[[str], None]] = None,
                 on_error_callback: Optional[Callable[[Exception], None]] = None,
                 session_id: Optional[str] = None,
                 verbose: bool = False):
        """
        Initialize the StreamCallback.
        
        Args:
            stream_handler: StreamHandler instance to use
            on_chunk_callback: Callback function for each chunk
            on_complete_callback: Callback function when streaming completes
            on_error_callback: Callback function for errors
            session_id: Session ID for the stream
            verbose: Whether to enable verbose logging
        """
        super().__init__()
        self.stream_handler = stream_handler or StreamHandler()
        self.on_chunk_callback = on_chunk_callback
        self.on_complete_callback = on_complete_callback
        self.on_error_callback = on_error_callback
        self.session_id = session_id
        self.verbose = verbose
        
        # Internal state
        self._current_run_id: Optional[str] = None
        self._accumulated_content = ""
        self._chunk_count = 0
        self._start_time: Optional[float] = None
    
    def on_llm_start(self, 
                    serialized: Dict[str, Any], 
                    prompts: List[str], 
                    **kwargs: Any) -> None:
        """Called when LLM starts running."""
        self._start_time = time.time()
        self._accumulated_content = ""
        self._chunk_count = 0
        
        # Start a new session if not provided
        if self.session_id is None:
            self.session_id = self.stream_handler.start_session()
        else:
            self.stream_handler.start_session(self.session_id)
        
        if self.verbose:
            print(f"🚀 Starting LLM stream (session: {self.session_id})")
    
    def on_llm_new_token(self, 
                        token: str, 
                        **kwargs: Any) -> None:
        """Called when a new token is generated."""
        try:
            # Handle the token as a stream chunk
            chunk = self.stream_handler.handle_stream(
                chunk=token,
                session_id=self.session_id,
                metadata={
                    'token_index': self._chunk_count,
                    'timestamp': time.time(),
                    'run_id': self._current_run_id
                }
            )
            
            self._accumulated_content += token
            self._chunk_count += 1
            
            # Call user callback if provided
            if self.on_chunk_callback:
                self.on_chunk_callback(chunk)
            
            if self.verbose:
                print(f"📝 Token {self._chunk_count}: '{token}'")
                
        except Exception as e:
            self._handle_error(e)
    
    def on_llm_end(self, 
                  response: LLMResult, 
                  **kwargs: Any) -> None:
        """Called when LLM ends running."""
        try:
            # End the streaming session
            session = self.stream_handler.end_session(self.session_id)
            
            # Get final aggregated content
            final_content = self.stream_handler.aggregate_stream(self.session_id)
            
            # Call completion callback if provided
            if self.on_complete_callback:
                self.on_complete_callback(final_content)
            
            if self.verbose and session:
                duration = session.duration
                print(f"✅ LLM stream completed (session: {self.session_id})")
                print(f"   Duration: {duration:.2f}s")
                print(f"   Tokens: {self._chunk_count}")
                print(f"   Characters: {len(final_content)}")
                if duration > 0:
                    print(f"   Rate: {self._chunk_count/duration:.1f} tokens/s")
                    
        except Exception as e:
            self._handle_error(e)
    
    def on_llm_error(self, 
                    error: Union[Exception, KeyboardInterrupt], 
                    **kwargs: Any) -> None:
        """Called when LLM encounters an error."""
        self._handle_error(error)
    
    def on_chain_start(self, 
                      serialized: Dict[str, Any], 
                      inputs: Dict[str, Any], 
                      **kwargs: Any) -> None:
        """Called when a chain starts running."""
        run_id = kwargs.get('run_id')
        if run_id:
            self._current_run_id = str(run_id)
        
        if self.verbose:
            print(f"🔗 Chain started (run_id: {self._current_run_id})")
    
    def on_chain_end(self, 
                    outputs: Dict[str, Any], 
                    **kwargs: Any) -> None:
        """Called when a chain ends running."""
        if self.verbose:
            print(f"🔗 Chain completed (run_id: {self._current_run_id})")
    
    def on_chain_error(self, 
                      error: Union[Exception, KeyboardInterrupt], 
                      **kwargs: Any) -> None:
        """Called when a chain encounters an error."""
        self._handle_error(error)
    
    def _handle_error(self, error: Union[Exception, KeyboardInterrupt]) -> None:
        """Handle errors during streaming."""
        if self.on_error_callback:
            self.on_error_callback(error)
        
        if self.verbose:
            print(f"❌ Stream error: {error}")
        
        # End session on error
        if self.session_id:
            self.stream_handler.end_session(self.session_id)
    
    def get_accumulated_content(self) -> str:
        """Get the accumulated content from the current stream."""
        return self._accumulated_content
    
    def get_chunk_count(self) -> int:
        """Get the number of chunks processed."""
        return self._chunk_count
    
    def get_session_statistics(self) -> Dict[str, Any]:
        """Get statistics for the current session."""
        if self.session_id:
            return self.stream_handler.get_statistics(self.session_id)
        return {}


class MultiStreamCallback(BaseCallbackHandler):
    """
    Callback handler that manages multiple stream callbacks.
    
    This allows handling multiple concurrent streams or applying
    multiple processing strategies to the same stream.
    """
    
    def __init__(self, callbacks: List[StreamCallback]):
        """
        Initialize with a list of stream callbacks.
        
        Args:
            callbacks: List of StreamCallback instances
        """
        super().__init__()
        self.callbacks = callbacks
    
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> None:
        """Forward to all callbacks."""
        for callback in self.callbacks:
            callback.on_llm_start(serialized, prompts, **kwargs)
    
    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Forward to all callbacks."""
        for callback in self.callbacks:
            callback.on_llm_new_token(token, **kwargs)
    
    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Forward to all callbacks."""
        for callback in self.callbacks:
            callback.on_llm_end(response, **kwargs)
    
    def on_llm_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any) -> None:
        """Forward to all callbacks."""
        for callback in self.callbacks:
            callback.on_llm_error(error, **kwargs)
    
    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any) -> None:
        """Forward to all callbacks."""
        for callback in self.callbacks:
            callback.on_chain_start(serialized, inputs, **kwargs)
    
    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        """Forward to all callbacks."""
        for callback in self.callbacks:
            callback.on_chain_end(outputs, **kwargs)
    
    def on_chain_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any) -> None:
        """Forward to all callbacks."""
        for callback in self.callbacks:
            callback.on_chain_error(error, **kwargs)


class BufferedStreamCallback(StreamCallback):
    """
    Stream callback that buffers chunks and processes them in batches.
    
    This is useful for reducing the frequency of processing when dealing
    with high-frequency streaming output.
    """
    
    def __init__(self,
                 buffer_size: int = 10,
                 flush_interval: float = 1.0,
                 **kwargs):
        """
        Initialize the buffered stream callback.
        
        Args:
            buffer_size: Number of tokens to buffer before processing
            flush_interval: Time interval to flush buffer (seconds)
            **kwargs: Arguments passed to StreamCallback
        """
        super().__init__(**kwargs)
        self.buffer_size = buffer_size
        self.flush_interval = flush_interval
        self._buffer: List[str] = []
        self._last_flush_time = time.time()
    
    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Buffer tokens and process in batches."""
        self._buffer.append(token)
        current_time = time.time()
        
        # Flush if buffer is full or interval has passed
        if (len(self._buffer) >= self.buffer_size or 
            current_time - self._last_flush_time >= self.flush_interval):
            self._flush_buffer()
    
    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Flush remaining buffer and complete."""
        self._flush_buffer()
        super().on_llm_end(response, **kwargs)
    
    def _flush_buffer(self) -> None:
        """Flush the token buffer."""
        if not self._buffer:
            return
        
        # Combine buffered tokens
        combined_content = "".join(self._buffer)
        
        try:
            # Handle as a single chunk
            chunk = self.stream_handler.handle_stream(
                chunk=combined_content,
                session_id=self.session_id,
                metadata={
                    'token_count': len(self._buffer),
                    'timestamp': time.time(),
                    'run_id': self._current_run_id,
                    'is_buffered': True
                }
            )
            
            self._accumulated_content += combined_content
            self._chunk_count += len(self._buffer)
            
            # Call user callback if provided
            if self.on_chunk_callback:
                self.on_chunk_callback(chunk)
            
            if self.verbose:
                print(f"📦 Buffered chunk: {len(self._buffer)} tokens, '{combined_content[:50]}...'")
                
        except Exception as e:
            self._handle_error(e)
        finally:
            # Clear buffer
            self._buffer.clear()
            self._last_flush_time = time.time()