"""
Middleware Utilities - Common middleware patterns for agents

This module provides reusable middleware functions for LangChain agents.

Custom Middleware Functions:
    - create_dynamic_model_selector(): Select model based on complexity
    - create_tool_error_handler(): Handle tool execution errors
    - create_context_based_prompt(): Generate prompts from context

Built-in Middleware Wrappers:
    - create_summarization_middleware(): Conversation summarization
    - create_human_in_loop_middleware(): Human approval for tool calls
    - create_model_call_limit_middleware(): Limit model calls
    - create_tool_call_limit_middleware(): Limit tool calls
    - create_model_fallback_middleware(): Model fallback on failure
    - create_pii_middleware(): PII detection and handling
    - create_todo_list_middleware(): Task planning and tracking
    - create_llm_tool_selector_middleware(): Intelligent tool selection
    - create_tool_retry_middleware(): Retry failed tool calls
    - create_model_retry_middleware(): Retry failed model calls
    - create_llm_tool_emulator(): Emulate tools with LLM
    - create_context_editing_middleware(): Manage conversation context
    - create_shell_tool_middleware(): Persistent shell session
    - create_filesystem_search_middleware(): File search tools

Based on: examples/practice/13_agent_base.py, 20_middleware_buildint.py
LangChain Version: 1.0
Python Version: 3.11

Author: AI Toolkit Team
Version: 1.0.0
"""

from typing import Callable, Dict, Any, Optional, List
from langchain_core.language_models import BaseChatModel


def create_dynamic_model_selector(
    basic_model: BaseChatModel,
    advanced_model: BaseChatModel,
    threshold: int = 10
) -> Callable:
    """
    Create middleware for dynamic model selection based on conversation complexity.
    
    Automatically switches to advanced model for longer conversations.
    Based on examples/practice/13_agent_base.py
    
    Args:
        basic_model: Model for simple/short conversations
        advanced_model: Model for complex/long conversations
        threshold: Message count to trigger advanced model (default: 10)
    
    Returns:
        Middleware function for model selection
    
    Example:
        >>> from ai_toolkit.models import ModelManager
        >>> from langchain.agents import create_agent
        >>> 
        >>> manager = ModelManager()
        >>> basic = manager.create_model("deepseek", model="deepseek-chat")
        >>> advanced = manager.create_model("qwen", model="qwen-turbo")
        >>> 
        >>> model_selector = create_dynamic_model_selector(
        ...     basic_model=basic,
        ...     advanced_model=advanced,
        ...     threshold=10
        ... )
        >>> 
        >>> agent = create_agent(
        ...     model=basic,  # Default model
        ...     tools=[search_tool],
        ...     middleware=[model_selector]
        ... )
        >>> 
        >>> # First 10 messages use basic model
        >>> # After 10 messages, automatically switches to advanced model
    
    Note:
        - Counts all messages in conversation history
        - Threshold includes system, human, AI, and tool messages
        - Model switch is transparent to the user
    """
    from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
    
    @wrap_model_call
    def dynamic_model_selection(request: ModelRequest, handler) -> ModelResponse:
        """Choose model based on conversation complexity."""
        message_count = len(request.state["messages"])
        
        if message_count > threshold:
            # Use advanced model for longer conversations
            model = advanced_model
        else:
            # Use basic model for shorter conversations
            model = basic_model
        
        return handler(request.override(model=model))
    
    return dynamic_model_selection


def create_tool_error_handler(
    error_message_template: Optional[str] = None,
    log_errors: bool = True
) -> Callable:
    """
    Create middleware for handling tool execution errors gracefully.
    
    Catches tool errors and returns user-friendly error messages.
    Based on examples/practice/13_agent_base.py
    
    Args:
        error_message_template: Custom error message template
                               Use {error} placeholder for error details
        log_errors: Whether to log errors (default: True)
    
    Returns:
        Middleware function for error handling
    
    Example:
        >>> from langchain.agents import create_agent
        >>> 
        >>> error_handler = create_tool_error_handler(
        ...     error_message_template="Tool failed: {error}. Please try again.",
        ...     log_errors=True
        ... )
        >>> 
        >>> agent = create_agent(
        ...     model=model,
        ...     tools=[search_tool, calculator_tool],
        ...     middleware=[error_handler]
        ... )
        >>> 
        >>> # If a tool fails, agent receives friendly error message
        >>> # instead of crashing
    
    Note:
        - Prevents agent crashes from tool failures
        - Returns ToolMessage with error details
        - Allows agent to recover and try alternative approaches
    """
    from langchain.agents.middleware import wrap_tool_call
    from langchain.messages import ToolMessage
    import logging
    
    logger = logging.getLogger(__name__)
    
    # Default error message template
    if error_message_template is None:
        error_message_template = (
            "Tool error: Please check your input and try again. "
            "Error details: {error}"
        )
    
    @wrap_tool_call
    def handle_tool_errors(request, handler):
        """Handle tool execution errors with custom messages."""
        try:
            return handler(request)
        except Exception as e:
            # Log error if enabled
            if log_errors:
                logger.error(
                    f"Tool '{request.tool_call.get('name', 'unknown')}' failed: {e}",
                    exc_info=True
                )
            
            # Return custom error message to the model
            error_msg = error_message_template.format(error=str(e))
            return ToolMessage(
                content=error_msg,
                tool_call_id=request.tool_call["id"]
            )
    
    return handle_tool_errors


def create_context_based_prompt(
    prompt_generator: Callable[[Dict[str, Any]], str],
    context_key: Optional[str] = None
) -> Callable:
    """
    Create middleware for generating dynamic prompts based on context.
    
    Allows system prompt to adapt based on user context (role, preferences, etc.).
    Based on examples/practice/13_agent_base.py
    
    Args:
        prompt_generator: Function that generates prompt from context
                         Signature: (context: Dict) -> str
        context_key: Optional key to extract from context
                    If None, passes entire context to generator
    
    Returns:
        Middleware function for dynamic prompts
    
    Example:
        >>> from langchain.agents import create_agent
        >>> from typing import TypedDict
        >>> 
        >>> class UserContext(TypedDict):
        ...     user_role: str
        ...     expertise_level: str
        >>> 
        >>> def generate_prompt(context):
        ...     role = context.get("user_role", "user")
        ...     level = context.get("expertise_level", "beginner")
        ...     
        ...     if role == "expert":
        ...         return "Provide detailed technical responses."
        ...     elif level == "beginner":
        ...         return "Explain concepts simply, avoid jargon."
        ...     else:
        ...         return "You are a helpful assistant."
        >>> 
        >>> prompt_middleware = create_context_based_prompt(generate_prompt)
        >>> 
        >>> agent = create_agent(
        ...     model=model,
        ...     tools=[search_tool],
        ...     middleware=[prompt_middleware],
        ...     context_schema=UserContext
        ... )
        >>> 
        >>> # Invoke with context
        >>> result = agent.invoke(
        ...     {"messages": [{"role": "user", "content": "Explain ML"}]},
        ...     context={"user_role": "expert", "expertise_level": "advanced"}
        ... )
    
    Note:
        - Context must be provided in agent.invoke(context=...)
        - Prompt is generated dynamically for each request
        - Useful for personalization and role-based responses
    """
    from langchain.agents.middleware import dynamic_prompt, ModelRequest
    
    @dynamic_prompt
    def dynamic_system_prompt(request: ModelRequest) -> str:
        """Generate dynamic system prompt from context."""
        context = request.runtime.context
        
        # Extract specific key if provided
        if context_key:
            context_value = context.get(context_key, {})
            return prompt_generator(context_value)
        else:
            return prompt_generator(context)
    
    return dynamic_system_prompt



# =============================================================================
# BUILT-IN MIDDLEWARE WRAPPERS
# =============================================================================

def create_summarization_middleware(
    model: Any,
    trigger_tokens: int = 4000,
    keep_messages: int = 20
) -> Any:
    """
    Create middleware for automatic conversation summarization.
    
    Wrapper for LangChain's built-in SummarizationMiddleware.
    Based on examples/practice/20_middleware_buildint.py
    
    Args:
        model: Model to use for summarization (can be model name or instance)
        trigger_tokens: Token count to trigger summarization (default: 4000)
        keep_messages: Number of recent messages to keep (default: 20)
    
    Returns:
        SummarizationMiddleware instance
    
    Example:
        >>> from ai_toolkit.agents import create_agent_with_tools
        >>> from ai_toolkit.agents.middleware_utils import create_summarization_middleware
        >>> 
        >>> summarization = create_summarization_middleware(
        ...     model="gpt-4o-mini",
        ...     trigger_tokens=4000,
        ...     keep_messages=20
        ... )
        >>> 
        >>> agent = create_agent_with_tools(
        ...     model=model,
        ...     tools=[search_tool],
        ...     middleware=[summarization]
        ... )
    
    Note:
        - Automatically summarizes old messages when token limit reached
        - Keeps recent messages for context
        - Reduces token usage for long conversations
    """
    from langchain.agents.middleware import SummarizationMiddleware
    
    return SummarizationMiddleware(
        model=model,
        trigger=("tokens", trigger_tokens),
        keep=("messages", keep_messages)
    )


def create_human_in_loop_middleware(
    interrupt_on: Dict[str, Any],
    checkpointer_required: bool = True
) -> Any:
    """
    Create middleware for human-in-the-loop approval.
    
    Wrapper for LangChain's built-in HumanInTheLoopMiddleware.
    Based on examples/practice/20_middleware_buildint.py
    
    Args:
        interrupt_on: Dict mapping tool names to approval settings
                     - True: Require approval
                     - False: No approval needed
                     - Dict with "allowed_decisions": List of allowed actions
        checkpointer_required: Whether checkpointer is required (default: True)
    
    Returns:
        HumanInTheLoopMiddleware instance
    
    Example:
        >>> from ai_toolkit.agents.middleware_utils import create_human_in_loop_middleware
        >>> 
        >>> human_approval = create_human_in_loop_middleware(
        ...     interrupt_on={
        ...         "send_email_tool": {
        ...             "allowed_decisions": ["approve", "edit", "reject"]
        ...         },
        ...         "read_email_tool": False
        ...     }
        ... )
        >>> 
        >>> # Requires checkpointer
        >>> agent = create_agent_with_memory(
        ...     model=model,
        ...     tools=[send_email_tool, read_email_tool],
        ...     checkpointer_type="inmemory",
        ...     middleware=[human_approval]
        ... )
    
    Note:
        - Requires checkpointer for state management
        - Interrupts execution for human approval
        - Supports approve, edit, reject decisions
    """
    from langchain.agents.middleware import HumanInTheLoopMiddleware
    
    if checkpointer_required:
        # Note: Checkpointer must be provided when creating agent
        pass
    
    return HumanInTheLoopMiddleware(interrupt_on=interrupt_on)


def create_model_call_limit_middleware(
    thread_limit: Optional[int] = None,
    run_limit: Optional[int] = None,
    exit_behavior: str = "end"
) -> Any:
    """
    Create middleware to limit model calls.
    
    Wrapper for LangChain's built-in ModelCallLimitMiddleware.
    Based on examples/practice/20_middleware_buildint.py
    
    Args:
        thread_limit: Max model calls per thread (conversation)
        run_limit: Max model calls per run (single invocation)
        exit_behavior: Behavior when limit reached ("end" or "error")
    
    Returns:
        ModelCallLimitMiddleware instance
    
    Example:
        >>> from ai_toolkit.agents.middleware_utils import create_model_call_limit_middleware
        >>> 
        >>> call_limiter = create_model_call_limit_middleware(
        ...     thread_limit=10,
        ...     run_limit=5,
        ...     exit_behavior="end"
        ... )
        >>> 
        >>> agent = create_agent_with_memory(
        ...     model=model,
        ...     tools=[search_tool],
        ...     checkpointer_type="inmemory",
        ...     middleware=[call_limiter]
        ... )
    
    Note:
        - Prevents infinite loops
        - Requires checkpointer for thread limiting
        - exit_behavior="end" stops gracefully, "error" raises exception
    """
    from langchain.agents.middleware import ModelCallLimitMiddleware
    
    return ModelCallLimitMiddleware(
        thread_limit=thread_limit,
        run_limit=run_limit,
        exit_behavior=exit_behavior
    )


def create_tool_call_limit_middleware(
    tool_name: Optional[str] = None,
    thread_limit: Optional[int] = None,
    run_limit: Optional[int] = None
) -> Any:
    """
    Create middleware to limit tool calls.
    
    Wrapper for LangChain's built-in ToolCallLimitMiddleware.
    Based on examples/practice/20_middleware_buildint.py
    
    Args:
        tool_name: Specific tool to limit (None for global limit)
        thread_limit: Max tool calls per thread
        run_limit: Max tool calls per run
    
    Returns:
        ToolCallLimitMiddleware instance
    
    Example:
        >>> from ai_toolkit.agents.middleware_utils import create_tool_call_limit_middleware
        >>> 
        >>> # Global limit
        >>> global_limiter = create_tool_call_limit_middleware(
        ...     thread_limit=20,
        ...     run_limit=10
        ... )
        >>> 
        >>> # Tool-specific limit
        >>> search_limiter = create_tool_call_limit_middleware(
        ...     tool_name="search",
        ...     thread_limit=5,
        ...     run_limit=3
        ... )
        >>> 
        >>> agent = create_agent_with_tools(
        ...     model=model,
        ...     tools=[search_tool, database_tool],
        ...     middleware=[global_limiter, search_limiter]
        ... )
    
    Note:
        - Can set global limits or per-tool limits
        - Prevents excessive tool usage
        - Multiple instances can be combined
    """
    from langchain.agents.middleware import ToolCallLimitMiddleware
    
    return ToolCallLimitMiddleware(
        tool_name=tool_name,
        thread_limit=thread_limit,
        run_limit=run_limit
    )


def create_model_fallback_middleware(
    *fallback_models: str
) -> Any:
    """
    Create middleware for automatic model fallback.
    
    Wrapper for LangChain's built-in ModelFallbackMiddleware.
    Based on examples/practice/20_middleware_buildint.py
    
    Args:
        *fallback_models: Model names to fallback to in order
    
    Returns:
        ModelFallbackMiddleware instance
    
    Example:
        >>> from ai_toolkit.agents.middleware_utils import create_model_fallback_middleware
        >>> 
        >>> fallback = create_model_fallback_middleware(
        ...     "gpt-4o-mini",
        ...     "claude-3-5-sonnet-20241022"
        ... )
        >>> 
        >>> agent = create_agent_with_tools(
        ...     model="gpt-4o",
        ...     tools=[search_tool],
        ...     middleware=[fallback]
        ... )
    
    Note:
        - Automatically tries fallback models on failure
        - Models are tried in order
        - Improves reliability
    """
    from langchain.agents.middleware import ModelFallbackMiddleware
    
    return ModelFallbackMiddleware(*fallback_models)


def create_pii_middleware(
    pii_type: str,
    strategy: str = "redact",
    apply_to_input: bool = True,
    apply_to_output: bool = False,
    detector: Optional[Any] = None
) -> Any:
    """
    Create middleware for PII detection and handling.
    
    Wrapper for LangChain's built-in PIIMiddleware.
    Based on examples/practice/20_middleware_buildint.py
    
    Args:
        pii_type: Type of PII to detect (e.g., "email", "credit_card", "ssn")
        strategy: Handling strategy ("redact", "mask", "block", "hash")
        apply_to_input: Apply to user input (default: True)
        apply_to_output: Apply to model output (default: False)
        detector: Custom detector (regex pattern, compiled regex, or function)
    
    Returns:
        PIIMiddleware instance
    
    Example:
        >>> from ai_toolkit.agents.middleware_utils import create_pii_middleware
        >>> import re
        >>> 
        >>> # Built-in PII types
        >>> email_pii = create_pii_middleware("email", strategy="redact")
        >>> card_pii = create_pii_middleware("credit_card", strategy="mask")
        >>> 
        >>> # Custom regex pattern
        >>> api_key_pii = create_pii_middleware(
        ...     "api_key",
        ...     detector=r"sk-[a-zA-Z0-9]{32}",
        ...     strategy="block"
        ... )
        >>> 
        >>> # Custom detector function
        >>> def detect_ssn(content: str) -> list[dict]:
        ...     matches = []
        ...     pattern = r"\\d{3}-\\d{2}-\\d{4}"
        ...     for match in re.finditer(pattern, content):
        ...         matches.append({
        ...             "text": match.group(0),
        ...             "start": match.start(),
        ...             "end": match.end()
        ...         })
        ...     return matches
        >>> 
        >>> ssn_pii = create_pii_middleware(
        ...     "ssn",
        ...     detector=detect_ssn,
        ...     strategy="hash"
        ... )
        >>> 
        >>> agent = create_agent_with_tools(
        ...     model=model,
        ...     tools=[search_tool],
        ...     middleware=[email_pii, card_pii, api_key_pii]
        ... )
    
    Note:
        - Strategies: redact (remove), mask (***), block (reject), hash (SHA256)
        - Built-in types: email, credit_card, phone, ssn, etc.
        - Custom detectors: regex string, compiled regex, or function
    """
    from langchain.agents.middleware import PIIMiddleware
    
    kwargs = {
        "strategy": strategy,
        "apply_to_input": apply_to_input,
        "apply_to_output": apply_to_output
    }
    
    if detector is not None:
        kwargs["detector"] = detector
    
    return PIIMiddleware(pii_type, **kwargs)


def create_todo_list_middleware() -> Any:
    """
    Create middleware for task planning and tracking.
    
    Wrapper for LangChain's built-in TodoListMiddleware.
    Based on examples/practice/20_middleware_buildint.py
    
    Returns:
        TodoListMiddleware instance
    
    Example:
        >>> from ai_toolkit.agents.middleware_utils import create_todo_list_middleware
        >>> 
        >>> todo_list = create_todo_list_middleware()
        >>> 
        >>> agent = create_agent_with_tools(
        ...     model=model,
        ...     tools=[read_file, write_file, run_tests],
        ...     middleware=[todo_list]
        ... )
    
    Note:
        - Equips agents with task planning capabilities
        - Useful for complex multi-step tasks
        - Agent can create, track, and complete tasks
    """
    from langchain.agents.middleware import TodoListMiddleware
    
    return TodoListMiddleware()


def create_llm_tool_selector_middleware(
    model: Any,
    max_tools: int = 3,
    always_include: Optional[List[str]] = None
) -> Any:
    """
    Create middleware for intelligent tool selection.
    
    Wrapper for LangChain's built-in LLMToolSelectorMiddleware.
    Based on examples/practice/20_middleware_buildint.py
    
    Args:
        model: Model to use for tool selection (can be model name or instance)
        max_tools: Maximum tools to select (default: 3)
        always_include: Tool names to always include (default: None)
    
    Returns:
        LLMToolSelectorMiddleware instance
    
    Example:
        >>> from ai_toolkit.agents.middleware_utils import create_llm_tool_selector_middleware
        >>> 
        >>> tool_selector = create_llm_tool_selector_middleware(
        ...     model="gpt-4o-mini",
        ...     max_tools=3,
        ...     always_include=["search"]
        ... )
        >>> 
        >>> agent = create_agent_with_tools(
        ...     model=model,
        ...     tools=[tool1, tool2, tool3, tool4, tool5, tool6, tool7],
        ...     middleware=[tool_selector]
        ... )
    
    Note:
        - Best for agents with many tools (10+)
        - Uses LLM to select relevant tools per query
        - Reduces token usage and improves performance
    """
    from langchain.agents.middleware import LLMToolSelectorMiddleware
    
    return LLMToolSelectorMiddleware(
        model=model,
        max_tools=max_tools,
        always_include=always_include or []
    )


def create_tool_retry_middleware(
    max_retries: int = 3,
    backoff_factor: float = 2.0,
    initial_delay: float = 1.0
) -> Any:
    """
    Create middleware for automatic tool retry with backoff.
    
    Wrapper for LangChain's built-in ToolRetryMiddleware.
    Based on examples/practice/20_middleware_buildint.py
    
    Args:
        max_retries: Maximum retry attempts (default: 3)
        backoff_factor: Exponential backoff multiplier (default: 2.0)
        initial_delay: Initial delay in seconds (default: 1.0)
    
    Returns:
        ToolRetryMiddleware instance
    
    Example:
        >>> from ai_toolkit.agents.middleware_utils import create_tool_retry_middleware
        >>> 
        >>> tool_retry = create_tool_retry_middleware(
        ...     max_retries=3,
        ...     backoff_factor=2.0,
        ...     initial_delay=1.0
        ... )
        >>> 
        >>> agent = create_agent_with_tools(
        ...     model=model,
        ...     tools=[search_tool, database_tool],
        ...     middleware=[tool_retry]
        ... )
    
    Note:
        - Automatically retries failed tool calls
        - Exponential backoff: 1s, 2s, 4s, ...
        - Improves reliability for flaky tools
    """
    from langchain.agents.middleware import ToolRetryMiddleware
    
    return ToolRetryMiddleware(
        max_retries=max_retries,
        backoff_factor=backoff_factor,
        initial_delay=initial_delay
    )


def create_model_retry_middleware(
    max_retries: int = 3,
    backoff_factor: float = 2.0,
    initial_delay: float = 1.0
) -> Any:
    """
    Create middleware for automatic model retry with backoff.
    
    Wrapper for LangChain's built-in ModelRetryMiddleware.
    Based on examples/practice/20_middleware_buildint.py
    
    Args:
        max_retries: Maximum retry attempts (default: 3)
        backoff_factor: Exponential backoff multiplier (default: 2.0)
        initial_delay: Initial delay in seconds (default: 1.0)
    
    Returns:
        ModelRetryMiddleware instance
    
    Example:
        >>> from ai_toolkit.agents.middleware_utils import create_model_retry_middleware
        >>> 
        >>> model_retry = create_model_retry_middleware(
        ...     max_retries=3,
        ...     backoff_factor=2.0,
        ...     initial_delay=1.0
        ... )
        >>> 
        >>> agent = create_agent_with_tools(
        ...     model=model,
        ...     tools=[search_tool],
        ...     middleware=[model_retry]
        ... )
    
    Note:
        - Automatically retries failed model calls
        - Exponential backoff: 1s, 2s, 4s, ...
        - Improves reliability for API failures
    """
    from langchain.agents.middleware import ModelRetryMiddleware
    
    return ModelRetryMiddleware(
        max_retries=max_retries,
        backoff_factor=backoff_factor,
        initial_delay=initial_delay
    )


def create_llm_tool_emulator(
    model: Optional[Any] = None
) -> Any:
    """
    Create middleware to emulate tool execution with LLM.
    
    Wrapper for LangChain's built-in LLMToolEmulator.
    Based on examples/practice/20_middleware_buildint.py
    
    Args:
        model: Model to use for emulation (default: uses agent's model)
    
    Returns:
        LLMToolEmulator instance
    
    Example:
        >>> from ai_toolkit.agents.middleware_utils import create_llm_tool_emulator
        >>> 
        >>> tool_emulator = create_llm_tool_emulator()
        >>> 
        >>> agent = create_agent_with_tools(
        ...     model=model,
        ...     tools=[get_weather, search_database, send_email],
        ...     middleware=[tool_emulator]
        ... )
    
    Note:
        - For testing purposes only
        - Replaces actual tool calls with AI-generated responses
        - Useful for development without real tool access
    """
    from langchain.agents.middleware import LLMToolEmulator
    
    if model is not None:
        return LLMToolEmulator(model=model)
    return LLMToolEmulator()


def create_context_editing_middleware(
    trigger_tokens: int = 100000,
    keep_tool_uses: int = 3
) -> Any:
    """
    Create middleware for managing conversation context.
    
    Wrapper for LangChain's built-in ContextEditingMiddleware with ClearToolUsesEdit.
    Based on examples/practice/20_middleware_buildint.py
    
    Args:
        trigger_tokens: Token count to trigger context editing (default: 100000)
        keep_tool_uses: Number of recent tool uses to keep (default: 3)
    
    Returns:
        ContextEditingMiddleware instance
    
    Example:
        >>> from ai_toolkit.agents.middleware_utils import create_context_editing_middleware
        >>> 
        >>> context_editor = create_context_editing_middleware(
        ...     trigger_tokens=100000,
        ...     keep_tool_uses=3
        ... )
        >>> 
        >>> agent = create_agent_with_tools(
        ...     model=model,
        ...     tools=[search_tool],
        ...     middleware=[context_editor]
        ... )
    
    Note:
        - Clears older tool call outputs when token limit reached
        - Preserves recent tool results
        - Helps manage long conversations
    """
    from langchain.agents.middleware import ContextEditingMiddleware, ClearToolUsesEdit
    
    return ContextEditingMiddleware(
        edits=[
            ClearToolUsesEdit(
                trigger=trigger_tokens,
                keep=keep_tool_uses
            )
        ]
    )


def create_shell_tool_middleware(
    workspace_root: str,
    execution_policy: Optional[Any] = None
) -> Any:
    """
    Create middleware for persistent shell session.
    
    Wrapper for LangChain's built-in ShellToolMiddleware.
    Based on examples/practice/20_middleware_buildint.py
    
    Args:
        workspace_root: Root directory for shell operations
        execution_policy: Execution policy (default: HostExecutionPolicy)
    
    Returns:
        ShellToolMiddleware instance
    
    Example:
        >>> from ai_toolkit.agents.middleware_utils import create_shell_tool_middleware
        >>> 
        >>> shell_tool = create_shell_tool_middleware(
        ...     workspace_root="/workspace"
        ... )
        >>> 
        >>> agent = create_agent_with_tools(
        ...     model=model,
        ...     tools=[search_tool],
        ...     middleware=[shell_tool]
        ... )
    
    Warning:
        - Provides shell access to the agent
        - Use with caution in production
        - Consider security implications
    
    Note:
        - Exposes persistent shell session to agent
        - Agent can execute shell commands
        - Useful for development and automation tasks
    """
    from langchain.agents.middleware import ShellToolMiddleware, HostExecutionPolicy
    
    if execution_policy is None:
        execution_policy = HostExecutionPolicy()
    
    return ShellToolMiddleware(
        workspace_root=workspace_root,
        execution_policy=execution_policy
    )


def create_filesystem_search_middleware(
    root_path: str,
    use_ripgrep: bool = True
) -> Any:
    """
    Create middleware for filesystem search tools.
    
    Wrapper for LangChain's built-in FilesystemFileSearchMiddleware.
    Based on examples/practice/20_middleware_buildint.py
    
    Args:
        root_path: Root directory for file searches
        use_ripgrep: Use ripgrep for faster searches (default: True)
    
    Returns:
        FilesystemFileSearchMiddleware instance
    
    Example:
        >>> from ai_toolkit.agents.middleware_utils import create_filesystem_search_middleware
        >>> 
        >>> file_search = create_filesystem_search_middleware(
        ...     root_path="/workspace",
        ...     use_ripgrep=True
        ... )
        >>> 
        >>> agent = create_agent_with_tools(
        ...     model=model,
        ...     tools=[],
        ...     middleware=[file_search]
        ... )
    
    Note:
        - Provides Glob and Grep search tools
        - ripgrep is faster than standard grep
        - Agent can search files in workspace
    """
    from langchain.agents.middleware import FilesystemFileSearchMiddleware
    
    return FilesystemFileSearchMiddleware(
        root_path=root_path,
        use_ripgrep=use_ripgrep
    )
