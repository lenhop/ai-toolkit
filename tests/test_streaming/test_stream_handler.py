"""
Tests for StreamHandler class.
"""

import pytest
import time
from unittest.mock import Mock, patch

from ai_toolkit.streaming.stream_handler import StreamHandler, StreamChunk, StreamSession


class TestStreamChunk:
    """Test StreamChunk dataclass."""
    
    def test_stream_chunk_creation(self):
        """Test creating a StreamChunk."""
        chunk = StreamChunk(
            content="Hello",
            metadata={"type": "text"},
            chunk_id="chunk_1"
        )
        
        assert chunk.content == "Hello"
        assert chunk.metadata == {"type": "text"}
        assert chunk.chunk_id == "chunk_1"
        assert chunk.is_final is False
        assert isinstance(chunk.timestamp, float)
    
    def test_stream_chunk_defaults(self):
        """Test StreamChunk with default values."""
        chunk = StreamChunk(content="Test")
        
        assert chunk.content == "Test"
        assert chunk.metadata == {}
        assert chunk.chunk_id is None
        assert chunk.is_final is False


class TestStreamSession:
    """Test StreamSession dataclass."""
    
    def test_stream_session_creation(self):
        """Test creating a StreamSession."""
        session = StreamSession(session_id="test_session")
        
        assert session.session_id == "test_session"
        assert session.chunks == []
        assert session.total_content == ""
        assert session.metadata == {}
        assert isinstance(session.start_time, float)
        assert session.end_time is None
    
    def test_session_properties(self):
        """Test StreamSession properties."""
        session = StreamSession(session_id="test")
        
        # Test duration
        assert session.duration > 0
        
        # Test chunk count
        assert session.chunk_count == 0
        session.chunks.append(StreamChunk(content="test"))
        assert session.chunk_count == 1
        
        # Test is_complete
        assert not session.is_complete
        session.end_time = time.time()
        assert session.is_complete


class TestStreamHandler:
    """Test StreamHandler class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.handler = StreamHandler()
    
    def test_initialization(self):
        """Test StreamHandler initialization."""
        handler = StreamHandler(
            buffer_size=500,
            auto_aggregate=False,
            chunk_separator="|"
        )
        
        assert handler.buffer_size == 500
        assert handler.auto_aggregate is False
        assert handler.chunk_separator == "|"
    
    def test_start_session(self):
        """Test starting a streaming session."""
        # Test auto-generated session ID
        session_id = self.handler.start_session()
        assert session_id is not None
        assert session_id in self.handler._sessions
        assert self.handler._current_session == session_id
        
        # Test custom session ID
        custom_id = "custom_session"
        session_id2 = self.handler.start_session(custom_id)
        assert session_id2 == custom_id
        assert custom_id in self.handler._sessions
    
    def test_end_session(self):
        """Test ending a streaming session."""
        session_id = self.handler.start_session()
        
        # End the session
        session = self.handler.end_session(session_id)
        assert session is not None
        assert session.session_id == session_id
        assert session.end_time is not None
        assert self.handler._current_session is None
    
    def test_handle_stream_string(self):
        """Test handling string chunks."""
        session_id = self.handler.start_session()
        
        chunk = self.handler.handle_stream("Hello world")
        
        assert chunk.content == "Hello world"
        assert chunk.metadata == {}
        
        session = self.handler.get_session(session_id)
        assert len(session.chunks) == 1
        assert session.total_content == "Hello world"
    
    def test_handle_stream_dict(self):
        """Test handling dictionary chunks."""
        session_id = self.handler.start_session()
        
        # Test content key
        chunk1 = self.handler.handle_stream({"content": "Hello", "type": "text"})
        assert chunk1.content == "Hello"
        assert chunk1.metadata["type"] == "text"
        
        # Test text key
        chunk2 = self.handler.handle_stream({"text": "World", "id": "123"})
        assert chunk2.content == "World"
        assert chunk2.metadata["id"] == "123"
        
        # Test delta format
        chunk3 = self.handler.handle_stream({"delta": {"content": "!"}})
        assert chunk3.content == "!"
    
    def test_handle_stream_object(self):
        """Test handling object chunks."""
        session_id = self.handler.start_session()
        
        # Mock object with content attribute
        mock_obj = Mock()
        mock_obj.content = "Mock content"
        mock_obj.additional_kwargs = {"key": "value"}
        
        chunk = self.handler.handle_stream(mock_obj)
        assert chunk.content == "Mock content"
        assert chunk.metadata["key"] == "value"
    
    def test_format_chunk(self):
        """Test chunk formatting."""
        # Test string formatting
        chunk = self.handler.format_chunk("test", {"meta": "data"})
        assert chunk.content == "test"
        assert chunk.metadata["meta"] == "data"
        
        # Test dict formatting
        chunk = self.handler.format_chunk(
            {"content": "hello", "id": "123"}, 
            {"extra": "info"}
        )
        assert chunk.content == "hello"
        assert chunk.metadata["id"] == "123"
        assert chunk.metadata["extra"] == "info"
    
    def test_aggregate_stream(self):
        """Test stream aggregation."""
        session_id = self.handler.start_session()
        
        # Add some chunks
        self.handler.handle_stream("Hello")
        self.handler.handle_stream(" ")
        self.handler.handle_stream("world")
        
        # Test aggregation
        result = self.handler.aggregate_stream()
        assert result == "Hello world"
        
        # Test with custom separator
        result = self.handler.aggregate_stream(separator="|")
        assert result == "Hello| |world"
    
    def test_session_management(self):
        """Test session management methods."""
        # Test empty state
        assert self.handler.list_sessions() == []
        assert self.handler.get_current_session() is None
        
        # Create sessions
        session1 = self.handler.start_session("session1")
        session2 = self.handler.start_session("session2")
        
        # Test listing
        sessions = self.handler.list_sessions()
        assert "session1" in sessions
        assert "session2" in sessions
        
        # Test getting sessions
        assert self.handler.get_session("session1") is not None
        assert self.handler.get_current_session().session_id == "session2"
        
        # Test clearing
        self.handler.clear_sessions(keep_current=False)
        assert self.handler.list_sessions() == []
        assert self.handler.get_current_session() is None
    
    def test_buffer_size_limit(self):
        """Test buffer size limiting."""
        handler = StreamHandler(buffer_size=2)
        
        # Create more sessions than buffer size
        handler.start_session("session1")
        handler.start_session("session2")
        handler.start_session("session3")  # Should remove session1
        
        sessions = handler.list_sessions()
        assert len(sessions) <= 2
        assert "session1" not in sessions
        assert "session2" in sessions
        assert "session3" in sessions
    
    def test_get_statistics(self):
        """Test getting streaming statistics."""
        session_id = self.handler.start_session()
        
        # Add some chunks
        self.handler.handle_stream("Hello")
        self.handler.handle_stream(" world!")
        
        # End session to get accurate duration
        self.handler.end_session(session_id)
        
        stats = self.handler.get_statistics(session_id)
        
        assert stats["session_id"] == session_id
        assert stats["chunk_count"] == 2
        assert stats["total_characters"] == 12  # "Hello world!"
        assert stats["average_chunk_size"] == 6.0
        assert stats["duration_seconds"] > 0
        assert stats["is_complete"] is True
    
    def test_stream_iterator(self):
        """Test stream iterator."""
        session_id = self.handler.start_session()
        
        # Add some chunks
        self.handler.handle_stream("chunk1")
        self.handler.handle_stream("chunk2")
        self.handler.end_session(session_id)
        
        # Test iterator
        chunks = list(self.handler.stream_iterator(session_id))
        assert len(chunks) == 2
        assert chunks[0].content == "chunk1"
        assert chunks[1].content == "chunk2"
    
    def test_stream_iterator_follow(self):
        """Test stream iterator with follow mode."""
        session_id = self.handler.start_session()
        
        # Add initial chunk
        self.handler.handle_stream("initial")
        
        # Test iterator with timeout
        chunks = []
        for chunk in self.handler.stream_iterator(session_id, follow=True, timeout=0.1):
            chunks.append(chunk)
            if len(chunks) >= 1:  # Stop after first chunk
                break
        
        assert len(chunks) == 1
        assert chunks[0].content == "initial"
    
    def test_auto_aggregate_disabled(self):
        """Test with auto-aggregation disabled."""
        handler = StreamHandler(auto_aggregate=False)
        session_id = handler.start_session()
        
        handler.handle_stream("Hello")
        handler.handle_stream(" world")
        
        session = handler.get_session(session_id)
        assert session.total_content == ""  # Not auto-aggregated
        
        # Manual aggregation should still work
        result = handler.aggregate_stream(session_id)
        assert result == "Hello world"
    
    def test_chunk_metadata_handling(self):
        """Test handling of chunk metadata."""
        session_id = self.handler.start_session()
        
        chunk = self.handler.handle_stream(
            "test content",
            metadata={"source": "test", "priority": "high"}
        )
        
        assert chunk.metadata["source"] == "test"
        assert chunk.metadata["priority"] == "high"
        
        # Test final chunk marking
        chunk = self.handler.handle_stream(
            {"content": "final", "is_final": True}
        )
        assert chunk.is_final is True
    
    def test_session_without_current(self):
        """Test handling chunks without current session."""
        # No current session set
        assert self.handler._current_session is None
        
        # Should auto-create session
        chunk = self.handler.handle_stream("test")
        assert chunk.content == "test"
        assert self.handler._current_session is not None
        
        # Should use the auto-created session
        session = self.handler.get_current_session()
        assert len(session.chunks) == 1