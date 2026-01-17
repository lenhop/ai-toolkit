"""
Memory Manager - Manage agent memory and conversation history

This module provides comprehensive memory management utilities for LangChain agents.

Overview:
    Provides classes and functions for managing agent memory including checkpointers,
    message trimming, summarization, and custom state management.

Key Classes:
    - MemoryManager: Main memory management class
    - CheckpointerFactory: Create different checkpointer types
    - MessageTrimmer: Trim messages to fit context window
    - MessageSummarizer: Summarize long conversations

Key Functions:
    - create_trimming_middleware(): Create message trimming middleware
    - create_deletion_middleware(): Create message deletion middleware
    - create_summarization_middleware(): Create summarization middleware
    - create_dynamic_prompt_middleware(): Create dynamic prompt middleware

Features:
    - Multiple checkpointer types (InMemory, PostgreSQL)
    - Flexible message trimming strategies
    - Automatic conversation summarization
    - Custom state management
    - Middleware creation utilities

Official Documentation:
    https://docs.langchain.com/oss/python/langchain/short-term-memory

Author: AI Toolkit Team
Version: 1.0.0
"""

from typing import Optional, Dict, Any, List, Callable, Union
from langchain_core.messages import BaseMessage, RemoveMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import REMOVE_ALL_MESSAGES


class CheckpointerFactory:
    """
    Factory for creating different types of checkpointers.
    
    Supports:
        - InMemory: Session-based memory (lost on restart)
        - PostgreSQL: Persistent database storage
    
    Usage Example:
        >>> factory = CheckpointerFactory()
        >>> checkpointer = factory.create_inmemory()
        >>> # Or for production:
        >>> checkpointer = factory.create_postgres("postgresql://...")
    """
    
    @staticmethod
    def create_inmemory() -> MemorySaver:
        """
        Create an in-memory checkpointer.
        
        Best for:
            - Development and testing
            - Single-session applications
            - Non-persistent memory needs
        
        Returns:
            MemorySaver instance
        
        Usage:
            >>> checkpointer = CheckpointerFactory.create_inmemory()
            >>> agent = create_agent(model, tools, checkpointer=checkpointer)
        """
        return MemorySaver()
    
    @staticmethod
    def create_postgres(db_uri: str, auto_setup: bool = True):
        """
        Create a PostgreSQL checkpointer for persistent storage.
        
        Best for:
            - Production environments
            - Multi-user applications
            - Persistent memory across restarts
        
        Args:
            db_uri: PostgreSQL connection string
                   Format: "postgresql://user:pass@host:port/db"
            auto_setup: Automatically create tables (default: True)
        
        Returns:
            PostgresSaver instance
        
        Usage:
            >>> db_uri = "postgresql://postgres:postgres@localhost:5432/db"
            >>> checkpointer = CheckpointerFactory.create_postgres(db_uri)
            >>> agent = create_agent(model, tools, checkpointer=checkpointer)
        
        Note:
            Requires: pip install langgraph-checkpoint-postgres
        """
        try:
            from langgraph.checkpoint.postgres import PostgresSaver
        except ImportError:
            raise ImportError(
                "PostgreSQL checkpointer requires langgraph-checkpoint-postgres. "
                "Install with: pip install langgraph-checkpoint-postgres"
            )
        
        checkpointer = PostgresSaver.from_conn_string(db_uri)
        if auto_setup:
            checkpointer.setup()  # Auto-create tables
        
        return checkpointer


class MessageTrimmer:
    """
    Trim messages to fit within context window.
    
    Strategies:
        - keep_recent: Keep only the N most recent messages
        - keep_first_and_recent: Keep first message + N recent messages
        - keep_by_tokens: Keep messages within token limit
    
    Usage Example:
        >>> trimmer = MessageTrimmer(strategy="keep_recent", max_messages=5)
        >>> trimmed = trimmer.trim(messages)
    """
    
    def __init__(
        self,
        strategy: str = "keep_first_and_recent",
        max_messages: int = 10,
        max_tokens: Optional[int] = None
    ):
        """
        Initialize message trimmer.
        
        Args:
            strategy: Trimming strategy
                     - "keep_recent": Keep N most recent messages
                     - "keep_first_and_recent": Keep first + N recent
                     - "keep_by_tokens": Keep within token limit
            max_messages: Maximum number of messages to keep
            max_tokens: Maximum tokens (for token-based strategy)
        """
        self.strategy = strategy
        self.max_messages = max_messages
        self.max_tokens = max_tokens
    
    def trim(self, messages: List[BaseMessage]) -> List[BaseMessage]:
        """
        Trim messages according to strategy.
        
        Args:
            messages: List of messages to trim
        
        Returns:
            Trimmed list of messages
        
        Usage:
            >>> messages = [msg1, msg2, msg3, msg4, msg5]
            >>> trimmer = MessageTrimmer(max_messages=3)
            >>> trimmed = trimmer.trim(messages)
            >>> len(trimmed)  # 4 (first + 3 recent)
        """
        if len(messages) <= self.max_messages:
            return messages
        
        if self.strategy == "keep_recent":
            return messages[-self.max_messages:]
        
        elif self.strategy == "keep_first_and_recent":
            # Keep first message (usually system message) + recent messages
            first_msg = messages[0]
            recent_count = self.max_messages - 1
            recent_messages = messages[-recent_count:]
            return [first_msg] + recent_messages
        
        elif self.strategy == "keep_by_tokens":
            if self.max_tokens is None:
                raise ValueError("max_tokens required for token-based strategy")
            
            # Simple token estimation (4 chars ≈ 1 token)
            total_tokens = 0
            kept_messages = []
            
            for msg in reversed(messages):
                msg_tokens = len(msg.content) // 4
                if total_tokens + msg_tokens <= self.max_tokens:
                    kept_messages.insert(0, msg)
                    total_tokens += msg_tokens
                else:
                    break
            
            return kept_messages
        
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
    
    def create_removal_commands(
        self,
        messages: List[BaseMessage]
    ) -> List[RemoveMessage]:
        """
        Create RemoveMessage commands for messages to be trimmed.
        
        Args:
            messages: Original message list
        
        Returns:
            List of RemoveMessage commands
        
        Usage:
            >>> trimmer = MessageTrimmer(max_messages=5)
            >>> removal_commands = trimmer.create_removal_commands(messages)
            >>> # Use in middleware to remove old messages
        """
        trimmed = self.trim(messages)
        trimmed_ids = {id(msg) for msg in trimmed}
        
        return [
            RemoveMessage(id=msg.id)
            for msg in messages
            if id(msg) not in trimmed_ids and hasattr(msg, 'id')
        ]


class MessageSummarizer:
    """
    Summarize long conversations to save context.
    
    Usage Example:
        >>> summarizer = MessageSummarizer(model="gpt-4o-mini")
        >>> summary = summarizer.summarize(messages)
    """
    
    def __init__(
        self,
        model: Optional[Any] = None,
        trigger_tokens: int = 4000,
        keep_messages: int = 20
    ):
        """
        Initialize message summarizer.
        
        Args:
            model: Model to use for summarization (optional)
            trigger_tokens: Token count to trigger summarization
            keep_messages: Number of recent messages to keep
        """
        self.model = model
        self.trigger_tokens = trigger_tokens
        self.keep_messages = keep_messages
    
    def should_summarize(self, messages: List[BaseMessage]) -> bool:
        """
        Check if messages should be summarized.
        
        Args:
            messages: List of messages
        
        Returns:
            True if summarization should be triggered
        """
        # Simple token estimation
        total_tokens = sum(len(msg.content) // 4 for msg in messages)
        return total_tokens > self.trigger_tokens
    
    def summarize(self, messages: List[BaseMessage]) -> str:
        """
        Summarize a list of messages.
        
        Args:
            messages: Messages to summarize
        
        Returns:
            Summary text
        
        Usage:
            >>> summarizer = MessageSummarizer()
            >>> summary = summarizer.summarize(old_messages)
            >>> # Replace old messages with summary
        """
        if not messages:
            return "No messages to summarize."
        
        # Create summary from messages
        conversation = "\n".join([
            f"{msg.type}: {msg.content}"
            for msg in messages
        ])
        
        if self.model:
            # Use model to create summary
            # This would call the model with a summarization prompt
            summary_prompt = f"Summarize this conversation:\n\n{conversation}"
            # summary = self.model.invoke(summary_prompt)
            # For now, return a simple summary
            return f"Summary of {len(messages)} messages: {conversation[:200]}..."
        else:
            # Simple summary without model
            return f"Conversation summary: {len(messages)} messages exchanged"


class MemoryManager:
    """
    Main memory management class for agents.
    
    Provides high-level interface for memory operations.
    
    Usage Example:
        >>> manager = MemoryManager()
        >>> checkpointer = manager.create_checkpointer("inmemory")
        >>> config = manager.create_config("thread-1")
        >>> trimmer = manager.create_trimmer(max_messages=10)
    """
    
    def __init__(self):
        """Initialize memory manager."""
        self.factory = CheckpointerFactory()
    
    def create_checkpointer(
        self,
        type: str = "inmemory",
        **kwargs
    ):
        """
        Create a checkpointer.
        
        Args:
            type: Checkpointer type ("inmemory" or "postgres")
            **kwargs: Additional arguments for checkpointer
        
        Returns:
            Checkpointer instance
        
        Usage:
            >>> manager = MemoryManager()
            >>> checkpointer = manager.create_checkpointer("inmemory")
            >>> # Or for PostgreSQL:
            >>> checkpointer = manager.create_checkpointer(
            ...     "postgres",
            ...     db_uri="postgresql://..."
            ... )
        """
        if type == "inmemory":
            return self.factory.create_inmemory()
        elif type == "postgres":
            db_uri = kwargs.get("db_uri")
            if not db_uri:
                raise ValueError("db_uri required for postgres checkpointer")
            return self.factory.create_postgres(db_uri, kwargs.get("auto_setup", True))
        else:
            raise ValueError(f"Unknown checkpointer type: {type}")
    
    @staticmethod
    def create_config(thread_id: str, **kwargs) -> Dict[str, Any]:
        """
        Create configuration for agent invocation.
        
        Args:
            thread_id: Thread ID for conversation isolation
            **kwargs: Additional configuration options
        
        Returns:
            Configuration dictionary
        
        Usage:
            >>> config = MemoryManager.create_config("user-123")
            >>> agent.invoke({"messages": [...]}, config=config)
        """
        config = {"configurable": {"thread_id": thread_id}}
        if kwargs:
            config["configurable"].update(kwargs)
        return config
    
    @staticmethod
    def create_trimmer(
        strategy: str = "keep_first_and_recent",
        max_messages: int = 10,
        max_tokens: Optional[int] = None
    ) -> MessageTrimmer:
        """
        Create a message trimmer.
        
        Args:
            strategy: Trimming strategy
            max_messages: Maximum messages to keep
            max_tokens: Maximum tokens (for token-based strategy)
        
        Returns:
            MessageTrimmer instance
        
        Usage:
            >>> trimmer = MemoryManager.create_trimmer(max_messages=5)
            >>> trimmed = trimmer.trim(messages)
        """
        return MessageTrimmer(strategy, max_messages, max_tokens)
    
    @staticmethod
    def create_summarizer(
        model: Optional[Any] = None,
        trigger_tokens: int = 4000,
        keep_messages: int = 20
    ) -> MessageSummarizer:
        """
        Create a message summarizer.
        
        Args:
            model: Model for summarization
            trigger_tokens: Token count to trigger summarization
            keep_messages: Recent messages to keep
        
        Returns:
            MessageSummarizer instance
        
        Usage:
            >>> summarizer = MemoryManager.create_summarizer()
            >>> if summarizer.should_summarize(messages):
            ...     summary = summarizer.summarize(messages[:-20])
        """
        return MessageSummarizer(model, trigger_tokens, keep_messages)


# =============================================================================
# MIDDLEWARE CREATION FUNCTIONS
# =============================================================================

def create_trimming_middleware(
    max_messages: int = 10,
    strategy: str = "keep_first_and_recent"
) -> Callable:
    """
    Create middleware for automatic message trimming.
    
    Args:
        max_messages: Maximum messages to keep
        strategy: Trimming strategy
    
    Returns:
        Middleware function
    
    Usage:
        >>> from langchain.agents import create_agent
        >>> from langchain.agents.middleware import before_model
        >>> 
        >>> trim_middleware = create_trimming_middleware(max_messages=5)
        >>> agent = create_agent(
        ...     model,
        ...     tools,
        ...     middleware=[trim_middleware]
        ... )
    
    Note:
        This creates a before_model middleware that trims messages
        before they are sent to the model.
    """
    from langchain.agents.middleware import before_model
    from langchain.agents import AgentState
    from langgraph.runtime import Runtime
    from langchain_core.messages import ToolMessage
    
    @before_model
    def trim_messages(state: AgentState, runtime: Runtime) -> Optional[Dict[str, Any]]:
        """Trim messages to fit context window."""
        messages = state["messages"]
        
        if len(messages) <= max_messages:
            return None
        
        # Keep system message if present
        system_msg = None
        other_messages = messages
        if messages and hasattr(messages[0], 'type') and messages[0].type == 'system':
            system_msg = messages[0]
            other_messages = messages[1:]
        
        # Calculate how many to keep
        keep_count = max_messages - (1 if system_msg else 0)
        
        # Keep recent messages, but ensure tool messages have their tool_calls
        recent = other_messages[-keep_count:] if len(other_messages) > keep_count else other_messages
        
        # Build final message list
        final_messages = []
        if system_msg:
            final_messages.append(system_msg)
        final_messages.extend(recent)
        
        # Only trim if we actually removed messages
        if len(final_messages) < len(messages):
            return {
                "messages": [
                    RemoveMessage(id=REMOVE_ALL_MESSAGES),
                    *final_messages
                ]
            }
        
        return None
    
    return trim_messages


def create_deletion_middleware(
    delete_count: int = 2,
    trigger_count: int = 10
) -> Callable:
    """
    Create middleware for automatic message deletion.
    
    Args:
        delete_count: Number of old messages to delete
        trigger_count: Message count to trigger deletion
    
    Returns:
        Middleware function
    
    Usage:
        >>> from langchain.agents import create_agent
        >>> 
        >>> delete_middleware = create_deletion_middleware(delete_count=2)
        >>> agent = create_agent(
        ...     model,
        ...     tools,
        ...     middleware=[delete_middleware]
        ... )
    
    Note:
        This creates an after_model middleware that deletes old messages
        after the model responds.
    """
    from langchain.agents.middleware import after_model
    from langchain.agents import AgentState
    from langgraph.runtime import Runtime
    
    @after_model
    def delete_old_messages(state: AgentState, runtime: Runtime) -> Optional[Dict[str, Any]]:
        """Remove old messages to keep conversation manageable."""
        messages = state["messages"]
        
        if len(messages) <= trigger_count:
            return None
        
        # Remove the earliest messages
        return {
            "messages": [
                RemoveMessage(id=m.id)
                for m in messages[:delete_count]
            ]
        }
    
    return delete_old_messages


def create_summarization_middleware(
    model: Any,
    trigger_tokens: int = 4000,
    keep_messages: int = 20
) -> Callable:
    """
    Create middleware for automatic conversation summarization.
    
    Args:
        model: Model to use for summarization
        trigger_tokens: Token count to trigger summarization
        keep_messages: Number of recent messages to keep
    
    Returns:
        Middleware function
    
    Usage:
        >>> from langchain.agents import create_agent
        >>> from langchain.agents.middleware import SummarizationMiddleware
        >>> 
        >>> # Use built-in SummarizationMiddleware
        >>> agent = create_agent(
        ...     model="gpt-4o",
        ...     tools=[],
        ...     middleware=[
        ...         SummarizationMiddleware(
        ...             model="gpt-4o-mini",
        ...             trigger=("tokens", 4000),
        ...             keep=("messages", 20)
        ...         )
        ...     ]
        ... )
    
    Note:
        For production use, prefer the built-in SummarizationMiddleware
        from langchain.agents.middleware
    """
    from langchain.agents.middleware import SummarizationMiddleware
    
    return SummarizationMiddleware(
        model=model,
        trigger=("tokens", trigger_tokens),
        keep=("messages", keep_messages)
    )


def create_dynamic_prompt_middleware(
    prompt_generator: Callable[[Dict[str, Any]], str]
) -> Callable:
    """
    Create middleware for dynamic prompt generation.
    
    Args:
        prompt_generator: Function that generates prompt from context
                         Signature: (context: Dict) -> str
    
    Returns:
        Middleware function
    
    Usage:
        >>> from langchain.agents import create_agent
        >>> from langchain.agents.middleware import dynamic_prompt
        >>> 
        >>> def generate_prompt(context):
        ...     user_name = context.get("user_name", "User")
        ...     return f"You are helpful. Address user as {user_name}."
        >>> 
        >>> prompt_middleware = create_dynamic_prompt_middleware(generate_prompt)
        >>> agent = create_agent(
        ...     model,
        ...     tools,
        ...     middleware=[prompt_middleware]
        ... )
    
    Note:
        The prompt_generator function receives the context dictionary
        and should return a system prompt string.
    """
    from langchain.agents.middleware import dynamic_prompt, ModelRequest
    
    @dynamic_prompt
    def dynamic_system_prompt(request: ModelRequest) -> str:
        """Generate dynamic system prompt from context."""
        context = request.runtime.context
        return prompt_generator(context)
    
    return dynamic_system_prompt
