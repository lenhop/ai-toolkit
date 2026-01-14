"""
Stream Handler for processing streaming output from AI models.

This module provides the StreamHandler class for handling, formatting,
and aggregating streaming output chunks from AI models.

Classes:
    StreamHandler: Handler for streaming output processing
        - Manages streaming sessions and chunks
        - Formats and aggregates streaming data
        - Tracks streaming statistics
        
        Methods:
            __init__(session_id, buffer_size): Initialize stream handler
            handle_stream(chunk): Process single streaming chunk
            format_chunk(chunk): Format chunk for display
            aggregate_stream(): Get aggregated stream content
            get_session_info(): Get session information
            clear_buffer(): Clear stream buffer
            is_complete(): Check if stream is complete
            get_statistics(): Get streaming statistics
"""

import time
from typing import Any, Dict, List, Optional, Union, Iterator
from dataclasses import dataclass, field
from langchain_core.outputs import LLMResult
from langchain_core.messages import BaseMessage


@dataclass
class StreamChunk:
    """Represents a single chunk of streaming data."""
    content: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    chunk_id: Optional[str] = None
    is_final: bool = False


@dataclass
class StreamSession:
    """Represents a streaming session with accumulated data."""
    session_id: str
    chunks: List[StreamChunk] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    total_content: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def duration(self) -> float:
        """Get the duration of the streaming session."""
        end = self.end_time or time.time()
        return end - self.start_time
    
    @property
    def chunk_count(self) -> int:
        """Get the number of chunks received."""
        return len(self.chunks)
    
    @property
    def is_complete(self) -> bool:
        """Check if the streaming session is complete."""
        return self.end_time is not None


class StreamHandler:
    """
    Handler for processing streaming output from AI models.
    
    This class provides methods to handle, format, and aggregate
    streaming output chunks from various AI models.
    """
    
    def __init__(self, 
                 buffer_size: int = 1000,
                 auto_aggregate: bool = True,
                 chunk_separator: str = ""):
        """
        Initialize the StreamHandler.
        
        Args:
            buffer_size: Maximum number of chunks to keep in memory
            auto_aggregate: Whether to automatically aggregate chunks
            chunk_separator: Separator between chunks when aggregating
        """
        self.buffer_size = buffer_size
        self.auto_aggregate = auto_aggregate
        self.chunk_separator = chunk_separator
        self._sessions: Dict[str, StreamSession] = {}
        self._current_session: Optional[str] = None
    
    def start_session(self, session_id: Optional[str] = None) -> str:
        """
        Start a new streaming session.
        
        Args:
            session_id: Optional session ID, auto-generated if not provided
            
        Returns:
            The session ID
        """
        if session_id is None:
            session_id = f"session_{int(time.time() * 1000)}"
        
        self._sessions[session_id] = StreamSession(session_id=session_id)
        self._current_session = session_id
        
        # Clean up old sessions if buffer is full
        if len(self._sessions) > self.buffer_size:
            oldest_session = min(self._sessions.keys(), 
                               key=lambda k: self._sessions[k].start_time)
            del self._sessions[oldest_session]
        
        return session_id
    
    def end_session(self, session_id: Optional[str] = None) -> Optional[StreamSession]:
        """
        End a streaming session.
        
        Args:
            session_id: Session ID to end, uses current session if not provided
            
        Returns:
            The completed session or None if not found
        """
        session_id = session_id or self._current_session
        if session_id and session_id in self._sessions:
            session = self._sessions[session_id]
            session.end_time = time.time()
            if session_id == self._current_session:
                self._current_session = None
            return session
        return None
    
    def handle_stream(self, 
                     chunk: Union[str, Dict[str, Any], Any],
                     session_id: Optional[str] = None,
                     metadata: Optional[Dict[str, Any]] = None) -> StreamChunk:
        """
        Handle a single streaming chunk.
        
        Args:
            chunk: The streaming chunk data
            session_id: Session ID, uses current session if not provided
            metadata: Additional metadata for the chunk
            
        Returns:
            Processed StreamChunk object
        """
        # Use current session or create new one
        if session_id is None:
            session_id = self._current_session
        if session_id is None:
            session_id = self.start_session()
        
        # Ensure session exists
        if session_id not in self._sessions:
            self.start_session(session_id)
        
        # Format the chunk
        formatted_chunk = self.format_chunk(chunk, metadata or {})
        
        # Add to session
        session = self._sessions[session_id]
        session.chunks.append(formatted_chunk)
        
        # Auto-aggregate if enabled
        if self.auto_aggregate:
            session.total_content += formatted_chunk.content
        
        return formatted_chunk
    
    def format_chunk(self, 
                    chunk: Union[str, Dict[str, Any], Any],
                    metadata: Dict[str, Any]) -> StreamChunk:
        """
        Format a raw chunk into a StreamChunk object.
        
        Args:
            chunk: Raw chunk data
            metadata: Chunk metadata
            
        Returns:
            Formatted StreamChunk object
        """
        content = ""
        chunk_metadata = metadata.copy()
        
        # Extract content based on chunk type
        if isinstance(chunk, str):
            content = chunk
        elif isinstance(chunk, dict):
            # Handle various dictionary formats
            if 'content' in chunk:
                content = str(chunk['content'])
            elif 'text' in chunk:
                content = str(chunk['text'])
            elif 'delta' in chunk and isinstance(chunk['delta'], dict):
                if 'content' in chunk['delta']:
                    content = str(chunk['delta']['content'])
                elif 'text' in chunk['delta']:
                    content = str(chunk['delta']['text'])
            else:
                content = str(chunk)
            
            # Extract metadata
            for key, value in chunk.items():
                if key not in ['content', 'text', 'delta']:
                    chunk_metadata[key] = value
        elif hasattr(chunk, 'content'):
            # Handle LangChain message objects
            content = str(chunk.content)
            if hasattr(chunk, 'additional_kwargs'):
                chunk_metadata.update(chunk.additional_kwargs)
        else:
            content = str(chunk)
        
        return StreamChunk(
            content=content,
            metadata=chunk_metadata,
            chunk_id=chunk_metadata.get('id'),
            is_final=chunk_metadata.get('is_final', False)
        )
    
    def aggregate_stream(self, 
                        session_id: Optional[str] = None,
                        separator: Optional[str] = None) -> str:
        """
        Aggregate all chunks in a session into a single string.
        
        Args:
            session_id: Session ID, uses current session if not provided
            separator: Separator between chunks, uses default if not provided
            
        Returns:
            Aggregated content string
        """
        session_id = session_id or self._current_session
        if session_id not in self._sessions:
            return ""
        
        session = self._sessions[session_id]
        separator = separator if separator is not None else self.chunk_separator
        
        if self.auto_aggregate and separator == self.chunk_separator:
            return session.total_content
        else:
            return separator.join(chunk.content for chunk in session.chunks)
    
    def get_session(self, session_id: str) -> Optional[StreamSession]:
        """
        Get a streaming session by ID.
        
        Args:
            session_id: Session ID
            
        Returns:
            StreamSession object or None if not found
        """
        return self._sessions.get(session_id)
    
    def get_current_session(self) -> Optional[StreamSession]:
        """
        Get the current streaming session.
        
        Returns:
            Current StreamSession object or None
        """
        if self._current_session:
            return self._sessions.get(self._current_session)
        return None
    
    def list_sessions(self) -> List[str]:
        """
        List all session IDs.
        
        Returns:
            List of session IDs
        """
        return list(self._sessions.keys())
    
    def clear_sessions(self, keep_current: bool = True) -> None:
        """
        Clear all sessions.
        
        Args:
            keep_current: Whether to keep the current session
        """
        if keep_current and self._current_session:
            current_session = self._sessions.get(self._current_session)
            self._sessions.clear()
            if current_session:
                self._sessions[self._current_session] = current_session
        else:
            self._sessions.clear()
            self._current_session = None
    
    def get_statistics(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get streaming statistics for a session.
        
        Args:
            session_id: Session ID, uses current session if not provided
            
        Returns:
            Dictionary with streaming statistics
        """
        session_id = session_id or self._current_session
        if session_id not in self._sessions:
            return {}
        
        session = self._sessions[session_id]
        
        # Calculate statistics
        total_chars = sum(len(chunk.content) for chunk in session.chunks)
        avg_chunk_size = total_chars / len(session.chunks) if session.chunks else 0
        
        # Calculate streaming rate
        duration = session.duration
        chars_per_second = total_chars / duration if duration > 0 else 0
        chunks_per_second = len(session.chunks) / duration if duration > 0 else 0
        
        return {
            'session_id': session_id,
            'chunk_count': len(session.chunks),
            'total_characters': total_chars,
            'average_chunk_size': avg_chunk_size,
            'duration_seconds': duration,
            'characters_per_second': chars_per_second,
            'chunks_per_second': chunks_per_second,
            'is_complete': session.is_complete,
            'start_time': session.start_time,
            'end_time': session.end_time
        }
    
    def stream_iterator(self, 
                       session_id: Optional[str] = None,
                       follow: bool = False,
                       timeout: Optional[float] = None) -> Iterator[StreamChunk]:
        """
        Create an iterator over stream chunks.
        
        Args:
            session_id: Session ID, uses current session if not provided
            follow: Whether to wait for new chunks (like tail -f)
            timeout: Timeout for waiting for new chunks
            
        Yields:
            StreamChunk objects
        """
        session_id = session_id or self._current_session
        if session_id not in self._sessions:
            return
        
        session = self._sessions[session_id]
        index = 0
        start_time = time.time()
        
        while True:
            # Yield available chunks
            while index < len(session.chunks):
                yield session.chunks[index]
                index += 1
            
            # If not following or session is complete, stop
            if not follow or session.is_complete:
                break
            
            # Check timeout
            if timeout and (time.time() - start_time) > timeout:
                break
            
            # Wait a bit before checking for new chunks
            time.sleep(0.01)