# Message Toolkit

Comprehensive utilities for working with LangChain message types.

## Overview

The Message Toolkit provides four main components for managing LangChain messages:

1. **MessageBuilder** - Fluent interface for building message lists
2. **MessageFormatter** - Display and format messages
3. **MessageValidator** - Validate message structures
4. **MessageConverter** - Convert between formats

## Message Types

LangChain supports four message types:

- **SystemMessage**: Sets agent behavior and personality (use at start)
- **HumanMessage**: Represents user input (every user message)
- **AIMessage**: Model output (usually auto-generated)
- **ToolMessage**: Tool execution results (with tool_call_id)

**Official Documentation:** https://docs.langchain.com/oss/python/langchain/messages

## Quick Start

```python
from ai_toolkit.messages import MessageBuilder, MessageFormatter

# Build messages with fluent interface
builder = MessageBuilder()
messages = (builder
    .add_system("You are a helpful assistant")
    .add_human("Hello!")
    .add_ai("Hi! How can I help?")
    .build())

# Display as conversation
MessageFormatter.print_messages(messages, style='conversation')
```

## Components

### 1. MessageBuilder

Fluent interface for constructing message lists.

```python
from ai_toolkit.messages import MessageBuilder

# Basic usage
builder = MessageBuilder()
messages = (builder
    .add_system("You are helpful")
    .add_human("What is 2+2?")
    .add_ai("2+2 equals 4")
    .build())

# Add conversation history
builder.add_conversation([
    ("Hello!", "Hi! How can I help?"),
    ("Thanks", "You're welcome!")
])

# Add tool messages
builder.add_tool(
    content="Result: 42",
    tool_call_id="call_123",
    name="calculator"
)

# Utility methods
count = builder.count()
last_msg = builder.get_last()
builder.clear()
```

### 2. MessageFormatter

Display and format messages in various styles.

```python
from ai_toolkit.messages import MessageFormatter

# Format as conversation (with emoji)
print(MessageFormatter.format_conversation(messages))
# Output:
# 🔧 System: You are helpful
# 👤 User: Hello!
# 🤖 AI: Hi! How can I help?

# Format with numbers
print(MessageFormatter.format_messages(messages, numbered=True))

# Get statistics
stats = MessageFormatter.get_message_stats(messages)
print(f"Total: {stats['total']}, Human: {stats['human']}")

# Format as dictionaries
dicts = MessageFormatter.format_as_dicts(messages)

# Print to console
MessageFormatter.print_messages(messages, style='conversation')
```

### 3. MessageValidator

Validate message structures and flows.

```python
from ai_toolkit.messages import MessageValidator

# Validate single message
is_valid, error = MessageValidator.validate_message(message)
if not is_valid:
    print(f"Error: {error}")

# Validate message list
is_valid, error = MessageValidator.validate_messages(
    messages,
    require_system=True,
    require_human=True
)

# Validate conversation flow
is_valid, error = MessageValidator.validate_conversation_flow(messages)

# Count message types
counts = MessageValidator.count_message_types(messages)
print(f"Human: {counts['human']}, AI: {counts['ai']}")

# Filter by type
human_msgs = MessageValidator.filter_by_type(messages, HumanMessage)

# Get first/last of type
first_human = MessageValidator.get_first_message_of_type(messages, HumanMessage)
last_ai = MessageValidator.get_last_message_of_type(messages, AIMessage)
```

### 4. MessageConverter

Convert between different message formats.

```python
from ai_toolkit.messages import MessageConverter

# Convert to dictionary
dict_msg = MessageConverter.to_dict(message)
# {'role': 'user', 'content': 'Hello!'}

# Convert from dictionary
data = {'role': 'user', 'content': 'Hello!'}
message = MessageConverter.from_dict(data)

# Convert to OpenAI format
openai_msgs = MessageConverter.to_openai_format(messages)

# Convert from OpenAI format
messages = MessageConverter.from_openai_format(openai_msgs)

# Extract content only
contents = MessageConverter.extract_content(messages)

# Merge message lists
merged = MessageConverter.merge_messages(history, new_messages)

# Clone messages
cloned = MessageConverter.clone_message(message)
cloned_list = MessageConverter.clone_messages(messages)
```

## Examples

### Example 1: Building Agent Conversation

```python
from ai_toolkit.messages import MessageBuilder

builder = MessageBuilder()
conversation = (builder
    .add_system("You are a math tutor")
    .add_human("What is 123 * 456?")
    .add_ai("Let me calculate that.")
    .add_tool("Result: 56088", tool_call_id="call_001", name="calculator")
    .add_ai("123 * 456 = 56,088")
    .build())
```

### Example 2: Validating User Input

```python
from ai_toolkit.messages import MessageValidator, MessageFormatter

# Validate messages
is_valid, error = MessageValidator.validate_messages(user_messages)

if is_valid:
    MessageFormatter.print_messages(user_messages, style='conversation')
else:
    print(f"Validation error: {error}")
```

### Example 3: Converting for API Calls

```python
from ai_toolkit.messages import MessageConverter

# Convert to OpenAI format
api_messages = [
    SystemMessage(content="You are helpful"),
    HumanMessage(content="Hello!")
]

openai_format = MessageConverter.to_openai_format(api_messages)
# Send to API...
```

### Example 4: Analyzing Conversations

```python
from ai_toolkit.messages import MessageFormatter, MessageValidator

# Get statistics
stats = MessageFormatter.get_message_stats(messages)
counts = MessageValidator.count_message_types(messages)

print(f"Total messages: {stats['total']}")
print(f"Human messages: {counts['human']}")
print(f"AI messages: {counts['ai']}")
print(f"Average length: {stats['total_chars'] // stats['total']} chars")
```

## Complete Example

See `examples/4.message_toolkit_guide.py` for a comprehensive demonstration of all features.

```bash
python examples/4.message_toolkit_guide.py
```

## API Reference

### MessageBuilder

- `add_system(content, name=None, **kwargs)` - Add SystemMessage
- `add_human(content, name=None, **kwargs)` - Add HumanMessage
- `add_ai(content, name=None, **kwargs)` - Add AIMessage
- `add_tool(content, tool_call_id, name=None, **kwargs)` - Add ToolMessage
- `add_message(message)` - Add pre-constructed message
- `add_messages(messages)` - Add multiple messages
- `add_conversation(exchanges)` - Add human-AI pairs
- `build()` - Return message list
- `clear()` - Clear all messages
- `count()` - Get message count
- `get_last()` - Get last message

### MessageFormatter

- `format_message(message, ...)` - Format single message
- `format_messages(messages, ...)` - Format multiple messages
- `format_conversation(messages, ...)` - Format as conversation
- `format_as_dict(message)` - Convert to dictionary
- `format_as_dicts(messages)` - Convert multiple to dictionaries
- `get_message_stats(messages)` - Get statistics
- `print_messages(messages, style)` - Print to console

### MessageValidator

- `validate_message(message)` - Validate single message
- `validate_messages(messages, ...)` - Validate message list
- `validate_conversation_flow(messages)` - Validate flow
- `has_message_type(messages, type)` - Check for type
- `count_message_types(messages)` - Count by type
- `get_first_message_of_type(messages, type)` - Get first of type
- `get_last_message_of_type(messages, type)` - Get last of type
- `filter_by_type(messages, type)` - Filter by type

### MessageConverter

- `to_dict(message)` - Convert to dictionary
- `to_dicts(messages)` - Convert multiple to dictionaries
- `from_dict(data)` - Convert from dictionary
- `from_dicts(data_list)` - Convert multiple from dictionaries
- `to_openai_format(messages)` - Convert to OpenAI format
- `from_openai_format(data_list)` - Convert from OpenAI format
- `to_string(messages, separator)` - Convert to string
- `extract_content(messages)` - Extract content only
- `merge_messages(*message_lists)` - Merge lists
- `clone_message(message)` - Clone message
- `clone_messages(messages)` - Clone multiple messages

## Benefits

- **Clean Code**: Fluent interface for readable message construction
- **Type Safety**: Validation ensures message quality
- **Flexibility**: Convert between formats easily
- **Debugging**: Format messages for better visibility
- **Reusability**: Standardized utilities across projects

## Official Documentation

- **LangChain Messages**: https://docs.langchain.com/oss/python/langchain/messages

## License

Part of AI Toolkit - See main project LICENSE
