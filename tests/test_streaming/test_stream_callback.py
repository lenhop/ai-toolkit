"""
Tests for StreamCallback classes.
"""

import pytest
import time
from unittest.mock import Mock, MagicMock, patch

from langchain_core.outputs import LLMResult, Generation
from langchain_core.messages import AIMessage

from ai_toolkit.streaming.stream_callback import (
    StreamCallback, 
    MultiStreamCallback, 
    BufferedStreamCallback
)
from ai_toolkit.streaming.stream_handler import StreamHandler


class TestStreamCallback:
    """Test StreamCallback class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.stream_handler = StreamHandler()
        self.callback = StreamCallback(stream_handler=self.stream_handler)
    
    def test_initialization(self):
        """Test StreamCallback initialization."""
        # Test with custom parameters
        on_chunk = Mock()
        on_complete = Mock()
        on_error = Mock()
        
        callback = StreamCallback(
            stream_handler=self.stream_handler,
            on_chunk_callback=on_chunk,
            on_complete_callback=on_complete,
            on_error_callback=on_error,
            session_id="test_session",
            verbose=True
        )
        
        assert callback.stream_handler == self.stream_handler
        assert callback.on_chunk_callback == on_chunk
        assert callback.on_complete_callback == on_complete
        assert callback.on_error_callback == on_error
        assert callback.session_id == "test_session"
        assert callback.verbose is True
    
    def test_on_llm_start(self):
        """Test LLM start callback."""
        serialized = {"name": "test_llm"}
        prompts = ["Hello world"]
        
        self.callback.on_llm_start(serialized, prompts)
        
        assert self.callback.session_id is not None
        assert self.callback._start_time is not None
        assert self.callback._accumulated_content == ""
        assert self.callback._chunk_count == 0
    
    def test_on_llm_new_token(self):
        """Test new token callback."""
        # Start LLM first
        self.callback.on_llm_start({}, ["test"])
        
        # Mock chunk callback
        chunk_callback = Mock()
        self.callback.on_chunk_callback = chunk_callback
        
        # Process tokens
        self.callback.on_llm_new_token("Hello")
        self.callback.on_llm_new_token(" ")
        self.callback.on_llm_new_token("world")
        
        assert self.callback._chunk_count == 3
        assert self.callback._accumulated_content == "Hello world"
        assert chunk_callback.call_count == 3
        
        # Check session has chunks
        session = self.callback.stream_handler.get_session(self.callback.session_id)
        assert len(session.chunks) == 3
    
    def test_on_llm_end(self):
        """Test LLM end callback."""
        # Start and process tokens
        self.callback.on_llm_start({}, ["test"])
        self.callback.on_llm_new_token("Hello")
        self.callback.on_llm_new_token(" world")
        
        # Mock completion callback
        complete_callback = Mock()
        self.callback.on_complete_callback = complete_callback
        
        # Create mock LLM result
        generation = Generation(text="Hello world")
        llm_result = LLMResult(generations=[[generation]])
        
        # End LLM
        self.callback.on_llm_end(llm_result)
        
        # Check completion callback was called
        complete_callback.assert_called_once_with("Hello world")
        
        # Check session is ended
        session = self.callback.stream_handler.get_session(self.callback.session_id)
        assert session.is_complete
    
    def test_on_llm_error(self):
        """Test LLM error callback."""
        # Start LLM
        self.callback.on_llm_start({}, ["test"])
        
        # Mock error callback
        error_callback = Mock()
        self.callback.on_error_callback = error_callback
        
        # Trigger error
        test_error = Exception("Test error")
        self.callback.on_llm_error(test_error)
        
        # Check error callback was called
        error_callback.assert_called_once_with(test_error)
        
        # Check session is ended
        session = self.callback.stream_handler.get_session(self.callback.session_id)
        assert session.is_complete
    
    def test_chain_callbacks(self):
        """Test chain start/end callbacks."""
        # Test chain start
        serialized = {"name": "test_chain"}
        inputs = {"input": "test"}
        run_id = "test_run_123"
        
        self.callback.on_chain_start(serialized, inputs, run_id=run_id)
        assert self.callback._current_run_id == run_id
        
        # Test chain end
        outputs = {"output": "result"}
        self.callback.on_chain_end(outputs)
        
        # Test chain error
        error_callback = Mock()
        self.callback.on_error_callback = error_callback
        
        test_error = Exception("Chain error")
        self.callback.on_chain_error(test_error)
        error_callback.assert_called_once_with(test_error)
    
    def test_get_accumulated_content(self):
        """Test getting accumulated content."""
        self.callback.on_llm_start({}, ["test"])
        self.callback.on_llm_new_token("Hello")
        self.callback.on_llm_new_token(" world")
        
        assert self.callback.get_accumulated_content() == "Hello world"
    
    def test_get_chunk_count(self):
        """Test getting chunk count."""
        self.callback.on_llm_start({}, ["test"])
        self.callback.on_llm_new_token("token1")
        self.callback.on_llm_new_token("token2")
        
        assert self.callback.get_chunk_count() == 2
    
    def test_get_session_statistics(self):
        """Test getting session statistics."""
        self.callback.on_llm_start({}, ["test"])
        self.callback.on_llm_new_token("Hello")
        self.callback.on_llm_new_token(" world")
        
        stats = self.callback.get_session_statistics()
        assert stats["chunk_count"] == 2
        assert stats["total_characters"] == 11  # "Hello world"
    
    def test_verbose_mode(self):
        """Test verbose mode output."""
        callback = StreamCallback(verbose=True)
        
        with patch('builtins.print') as mock_print:
            callback.on_llm_start({}, ["test"])
            callback.on_llm_new_token("Hello")
            
            # Check that print was called
            assert mock_print.call_count >= 2
    
    def test_error_handling_in_token_processing(self):
        """Test error handling during token processing."""
        # Mock stream handler to raise error
        mock_handler = Mock()
        mock_handler.handle_stream.side_effect = Exception("Handler error")
        
        callback = StreamCallback(stream_handler=mock_handler)
        error_callback = Mock()
        callback.on_error_callback = error_callback
        
        callback.on_llm_start({}, ["test"])
        callback.on_llm_new_token("test")
        
        # Error callback should be called
        error_callback.assert_called_once()


class TestMultiStreamCallback:
    """Test MultiStreamCallback class."""
    
    def test_initialization(self):
        """Test MultiStreamCallback initialization."""
        callback1 = StreamCallback()
        callback2 = StreamCallback()
        
        multi_callback = MultiStreamCallback([callback1, callback2])
        assert len(multi_callback.callbacks) == 2
    
    def test_forwarding_to_all_callbacks(self):
        """Test that all methods are forwarded to all callbacks."""
        # Create mock callbacks
        callback1 = Mock(spec=StreamCallback)
        callback2 = Mock(spec=StreamCallback)
        
        multi_callback = MultiStreamCallback([callback1, callback2])
        
        # Test LLM callbacks
        serialized = {"name": "test"}
        prompts = ["test"]
        
        multi_callback.on_llm_start(serialized, prompts)
        callback1.on_llm_start.assert_called_once_with(serialized, prompts)
        callback2.on_llm_start.assert_called_once_with(serialized, prompts)
        
        multi_callback.on_llm_new_token("token")
        callback1.on_llm_new_token.assert_called_once_with("token")
        callback2.on_llm_new_token.assert_called_once_with("token")
        
        # Test chain callbacks
        inputs = {"input": "test"}
        multi_callback.on_chain_start(serialized, inputs)
        callback1.on_chain_start.assert_called_once_with(serialized, inputs)
        callback2.on_chain_start.assert_called_once_with(serialized, inputs)
        
        # Test error callbacks
        error = Exception("test")
        multi_callback.on_llm_error(error)
        callback1.on_llm_error.assert_called_once_with(error)
        callback2.on_llm_error.assert_called_once_with(error)


class TestBufferedStreamCallback:
    """Test BufferedStreamCallback class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.stream_handler = StreamHandler()
        self.callback = BufferedStreamCallback(
            stream_handler=self.stream_handler,
            buffer_size=3,
            flush_interval=1.0
        )
    
    def test_initialization(self):
        """Test BufferedStreamCallback initialization."""
        assert self.callback.buffer_size == 3
        assert self.callback.flush_interval == 1.0
        assert self.callback._buffer == []
    
    def test_buffering_tokens(self):
        """Test token buffering."""
        self.callback.on_llm_start({}, ["test"])
        
        # Mock chunk callback
        chunk_callback = Mock()
        self.callback.on_chunk_callback = chunk_callback
        
        # Add tokens (less than buffer size)
        self.callback.on_llm_new_token("Hello")
        self.callback.on_llm_new_token(" ")
        
        # Should not flush yet
        assert chunk_callback.call_count == 0
        assert len(self.callback._buffer) == 2
        
        # Add third token (reaches buffer size)
        self.callback.on_llm_new_token("world")
        
        # Should flush now
        assert chunk_callback.call_count == 1
        assert len(self.callback._buffer) == 0
        
        # Check the flushed content
        call_args = chunk_callback.call_args[0][0]
        assert call_args.content == "Hello world"
        assert call_args.metadata["is_buffered"] is True
        assert call_args.metadata["token_count"] == 3
    
    def test_flush_on_interval(self):
        """Test flushing based on time interval."""
        callback = BufferedStreamCallback(
            stream_handler=self.stream_handler,
            buffer_size=10,  # Large buffer
            flush_interval=0.1  # Short interval
        )
        
        callback.on_llm_start({}, ["test"])
        chunk_callback = Mock()
        callback.on_chunk_callback = chunk_callback
        
        # Add token
        callback.on_llm_new_token("test")
        assert chunk_callback.call_count == 0
        
        # Wait for interval
        time.sleep(0.15)
        
        # Add another token (should trigger flush due to interval)
        callback.on_llm_new_token("token")
        assert chunk_callback.call_count == 1
    
    def test_flush_on_end(self):
        """Test flushing remaining buffer on LLM end."""
        self.callback.on_llm_start({}, ["test"])
        
        chunk_callback = Mock()
        self.callback.on_chunk_callback = chunk_callback
        
        # Add tokens (less than buffer size)
        self.callback.on_llm_new_token("Hello")
        self.callback.on_llm_new_token(" world")
        
        # Should not flush yet
        assert chunk_callback.call_count == 0
        
        # End LLM (should flush remaining buffer)
        generation = Generation(text="Hello world")
        llm_result = LLMResult(generations=[[generation]])
        self.callback.on_llm_end(llm_result)
        
        # Should have flushed
        assert chunk_callback.call_count == 1
        call_args = chunk_callback.call_args[0][0]
        assert call_args.content == "Hello world"
    
    def test_empty_buffer_flush(self):
        """Test flushing empty buffer."""
        self.callback.on_llm_start({}, ["test"])
        
        chunk_callback = Mock()
        self.callback.on_chunk_callback = chunk_callback
        
        # Flush empty buffer
        self.callback._flush_buffer()
        
        # Should not call callback
        assert chunk_callback.call_count == 0
    
    def test_error_handling_in_flush(self):
        """Test error handling during buffer flush."""
        # Mock stream handler to raise error
        mock_handler = Mock()
        mock_handler.handle_stream.side_effect = Exception("Handler error")
        
        callback = BufferedStreamCallback(
            stream_handler=mock_handler,
            buffer_size=2
        )
        
        error_callback = Mock()
        callback.on_error_callback = error_callback
        
        callback.on_llm_start({}, ["test"])
        
        # Add tokens to trigger flush
        callback.on_llm_new_token("token1")
        callback.on_llm_new_token("token2")  # Should trigger flush and error
        
        # Error callback should be called
        error_callback.assert_called_once()
        
        # Buffer should be cleared even on error
        assert len(callback._buffer) == 0
    
    def test_accumulated_content_tracking(self):
        """Test that accumulated content is tracked correctly."""
        self.callback.on_llm_start({}, ["test"])
        
        # Add tokens
        self.callback.on_llm_new_token("Hello")
        self.callback.on_llm_new_token(" ")
        self.callback.on_llm_new_token("world")  # Triggers flush
        
        # Check accumulated content
        assert self.callback.get_accumulated_content() == "Hello world"
        assert self.callback.get_chunk_count() == 3