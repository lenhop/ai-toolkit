# AI Toolkit - Complete Modules Overview

This document provides a comprehensive overview of all modules in the AI Toolkit, organized by directory.

## Table of Contents

1. [Configuration Management (`ai_toolkit/config`)](#configuration-management)
2. [Model Management (`ai_toolkit/models`)](#model-management)
3. [Prompt Management (`ai_toolkit/prompts`)](#prompt-management)
4. [Output Parsing (`ai_toolkit/parsers`)](#output-parsing)
5. [Streaming Processing (`ai_toolkit/streaming`)](#streaming-processing)
6. [Error Handling (`ai_toolkit/errors`)](#error-handling)
7. [Token Management (`ai_toolkit/tokens`)](#token-management)
8. [Utilities (`ai_toolkit/utils`)](#utilities)

---

## Configuration Management

**Directory:** `ai_toolkit/config/`

### Overview
The Configuration Management toolkit provides comprehensive configuration handling including file loading, environment variable management, and validation.

### Modules

#### 1. `config_manager.py`
**Purpose:** Load and manage YAML/JSON configuration files with environment variable substitution.

**Key Classes:**
- `ConfigManager`: Central configuration manager
  - `load_config(config_path)`: Load configuration from file
  - `save_config(config_path, config)`: Save configuration to file
  - `get_config(key, default)`: Get configuration value with dot notation
  - `set_config(key, value)`: Set configuration value
  - `merge_configs(*config_paths)`: Merge multiple configuration files
  - `to_dict()`: Export configuration as dictionary
  - `from_dict(config)`: Load configuration from dictionary

**Features:**
- YAML and JSON support
- Environment variable substitution (`${VAR_NAME}` or `${VAR_NAME:default}`)
- Nested key access with dot notation
- Configuration merging
- Reset and clear operations


#### 2. `config_validator.py`
**Purpose:** Validate configuration values against flexible rules.

**Key Classes:**
- `ValidationRule`: Base class for validation rules
- `RequiredRule`: Check if value is required (not None/empty)
- `TypeRule`: Validate value type
- `RangeRule`: Validate numeric ranges
- `PatternRule`: Validate string patterns (regex)
- `ChoiceRule`: Validate value is in allowed choices
- `CustomRule`: Custom validation function
- `ConfigValidator`: Main validator class
  - `add_rule(key, rule)`: Add validation rule
  - `validate(config)`: Validate configuration
  - `get_errors()`: Get validation errors
  - `validate_model_config(config)`: Specialized model validation
  - `validate_api_keys(api_keys)`: Validate API keys
  - `validate_file_path(path)`: Validate file paths
  - `validate_url(url)`: Validate URLs

**Features:**
- Flexible rule-based validation
- Nested key validation with dot notation
- Multiple validation rules per key
- Specialized validators for common use cases

#### 3. `env_loader.py`
**Purpose:** Load and manage environment variables with type safety.

**Key Classes:**
- `EnvLoader`: Environment variable loader
  - `load_env_file(env_file)`: Load .env file
  - `get(key, default, cast, required)`: Get variable with type casting
  - `get_str/get_int/get_float/get_bool/get_list()`: Type-specific getters
  - `get_api_key(provider)`: Get API key with multiple naming conventions
  - `set(key, value)`: Set environment variable
  - `validate_required_vars(vars)`: Validate required variables
  - `create_env_file(variables)`: Create .env file

**Features:**
- Type-safe variable access
- Multiple naming convention support for API keys
- Batch loading with mappings
- .env file creation and management

---

## Model Management

**Directory:** `ai_toolkit/models/`

### Overview
The Model Management toolkit provides unified interfaces for multiple AI model providers including DeepSeek, Qwen, and GLM.

### Modules

#### 1. `model_config.py`
**Purpose:** Pydantic-based configuration classes for AI models.

**Key Classes:**
- `ModelConfig`: Model configuration with validation
  - Fields: `api_key`, `base_url`, `model`, `temperature`, `max_tokens`, `top_p`, `frequency_penalty`, `presence_penalty`, `timeout`, `max_retries`
  - Validators for API keys, URLs, and model names
- `ProviderConfig`: Provider-specific configuration
  - Fields: `name`, `class_path`, `supported_models`, `default_model`, `pricing`

**Functions:**
- `load_config_from_env(provider)`: Load configuration from environment variables
- `validate_config(config)`: Validate model configuration

**Features:**
- Pydantic validation
- Environment variable loading
- Provider-specific defaults
- Type safety

#### 2. `model_providers.py`
**Purpose:** Provider implementations for different AI services.

**Key Classes:**
- `BaseModelProvider`: Abstract base class
  - `create_model()`: Create model instance (abstract)
  - `validate_config()`: Validate configuration (abstract)
  - `get_model()`: Get or create model (lazy loading)
- `DeepSeekProvider`: DeepSeek implementation
  - Supports: deepseek-chat, deepseek-coder
- `QwenProvider`: Qwen (Alibaba) implementation
  - Supports: qwen-turbo, qwen-plus, qwen-max, qwen-long
- `GLMProvider`: GLM (Zhipu AI) implementation
  - Supports: glm-4.6, glm-4, glm-3-turbo
- `GLMChatModel`: LangChain-compatible GLM wrapper

**Functions:**
- `create_provider(provider_name, config)`: Factory function
- `get_provider_class(provider_name)`: Get provider class

**Features:**
- Unified provider interface
- LangChain compatibility
- Provider-specific validation
- Model caching

#### 3. `model_manager.py`
**Purpose:** Centralized management for AI models and providers.

**Key Classes:**
- `ModelManager`: Central model manager
  - `load_config(config_path)`: Load configuration from file
  - `create_model(provider_name, model_name, **kwargs)`: Create model instance
  - `get_model(provider_name, model_name)`: Get cached model
  - `list_models()`: List all available models
  - `list_providers()`: List all providers
  - `remove_model(provider_name, model_name)`: Remove cached model
  - `get_model_info(provider_name, model_name)`: Get model information

**Features:**
- Model caching
- Configuration management
- Multiple provider support
- Environment variable fallback

---

## Prompt Management

**Directory:** `ai_toolkit/prompts/`

### Overview
The Prompt Management toolkit provides template management, loading, and rendering for various prompt types.

### Modules

#### 1. `prompt_templates.py`
**Purpose:** Template classes for different prompt types.

**Key Classes:**
- `BasePromptTemplate`: Abstract base class
  - `render(**kwargs)`: Render template (abstract)
  - `get_variables()`: Get required variables
  - `validate_variables(**kwargs)`: Validate provided variables
  - `to_langchain()`: Convert to LangChain template
- `SimplePromptTemplate`: Basic string template
- `ChatPromptTemplate`: Multi-turn conversation template
  - Fields: `system_message`, `human_message`, `ai_message`
- `SystemPromptTemplate`: System instruction template
  - Fields: `instructions`, `constraints`, `examples`
- `FewShotPromptTemplate`: Few-shot learning template
  - Fields: `examples`, `example_template`, `prefix`, `suffix`

**Functions:**
- `create_template(template_type, **kwargs)`: Factory function
- `detect_template_type(template_data)`: Auto-detect template type

**Features:**
- Multiple template types
- Variable validation
- LangChain integration
- Pydantic validation

#### 2. `prompt_loader.py`
**Purpose:** Load templates from files and directories.

**Key Classes:**
- `PromptLoader`: Template file loader
  - `load_from_file(file_path)`: Load single template
  - `load_from_dir(dir_path, recursive)`: Load directory of templates
  - `save_template(template, file_path)`: Save template to file
  - `find_templates(pattern)`: Find templates by pattern
  - `get_template_info(file_path)`: Get template metadata
  - `get_cached_template(name)`: Get cached template
  - `clear_cache()`: Clear template cache

**Features:**
- YAML, JSON, TXT, MD support
- Recursive directory loading
- Template caching
- Metadata extraction

#### 3. `prompt_manager.py`
**Purpose:** Centralized prompt template management.

**Key Classes:**
- `PromptManager`: Central template manager
  - `load_templates(paths)`: Load templates from paths
  - `load_template(name, template_data)`: Load from data
  - `get_template(name)`: Get template by name
  - `render_template(name, **kwargs)`: Render template
  - `create_template(name, type, **kwargs)`: Create new template
  - `list_templates()`: List all templates
  - `search_templates(query)`: Search templates
  - `clone_template(source, target)`: Clone template
  - `export_templates(output_path)`: Export all templates

**Features:**
- Template lifecycle management
- Search and filtering
- Template cloning
- Batch operations

---

## Output Parsing

**Directory:** `ai_toolkit/parsers/`

### Overview
The Output Parsing toolkit converts AI model outputs into structured data formats.

### Modules

#### 1. `output_parser.py`
**Purpose:** Parser classes for different output formats.

**Key Classes:**
- `BaseOutputParser`: Abstract base class
  - `parse(text)`: Parse text (abstract)
  - `get_format_instructions()`: Get format instructions (abstract)
  - `to_langchain()`: Convert to LangChain parser
- `StrOutputParser`: String parser with cleaning
- `JsonOutputParser`: JSON parser with error recovery
  - Features: JSON extraction, format fixing, schema validation
- `PydanticOutputParser`: Pydantic model parser
- `ListOutputParser`: List parser with custom separators
- `RegexOutputParser`: Regex-based extraction

**Functions:**
- `create_parser(parser_type, **kwargs)`: Factory function

**Features:**
- Multiple parser types
- Error recovery
- LangChain compatibility
- Schema validation

#### 2. `parser_manager.py`
**Purpose:** Centralized parser management.

**Key Classes:**
- `ParserManager`: Central parser manager
  - `create_parser(name, parser_type, **kwargs)`: Create parser
  - `get_parser(name)`: Get parser by name
  - `parse(parser_name, text)`: Parse with error handling
  - `parse_with_fallback(parser_names, text)`: Try multiple parsers
  - `validate_output(parser_name, output)`: Validate output
  - `batch_parse(parser_name, texts)`: Parse multiple texts
  - `test_parser(parser_name, test_text)`: Test parser
  - `clone_parser(source, target)`: Clone parser

**Features:**
- Parser caching
- Error recovery
- Fallback parsing
- Batch processing

#### 3. `error_handler.py`
**Purpose:** Error recovery for parsing failures.

**Key Classes:**
- `OutputErrorHandler`: Parsing error handler
  - `handle_error(text, error_msg, parser)`: Handle parsing error
  - `retry_parse(parser, text, max_attempts)`: Retry with recovery
  - `fix_json(json_text)`: Fix malformed JSON
  - `get_recovery_suggestions(text, parser_type)`: Get suggestions

**Features:**
- Multiple recovery strategies
- JSON fixing
- Error analysis
- Retry mechanisms

---

## Streaming Processing

**Directory:** `ai_toolkit/streaming/`

### Overview
The Streaming Processing toolkit handles real-time streaming output from AI models.

### Modules

#### 1. `stream_handler.py`
**Purpose:** Handle and aggregate streaming chunks.

**Key Classes:**
- `StreamChunk`: Single streaming chunk
  - Fields: `content`, `timestamp`, `metadata`, `chunk_id`, `is_final`
- `StreamSession`: Streaming session data
  - Fields: `session_id`, `chunks`, `start_time`, `end_time`, `total_content`
  - Properties: `duration`, `chunk_count`, `is_complete`
- `StreamHandler`: Stream processing handler
  - `start_session(session_id)`: Start streaming session
  - `end_session(session_id)`: End session
  - `handle_stream(chunk, session_id)`: Process chunk
  - `format_chunk(chunk)`: Format raw chunk
  - `aggregate_stream(session_id)`: Get aggregated content
  - `get_statistics(session_id)`: Get streaming statistics
  - `stream_iterator(session_id)`: Iterate over chunks

**Features:**
- Session management
- Chunk aggregation
- Statistics tracking
- Iterator support

#### 2. `stream_callback.py`
**Purpose:** LangChain-compatible streaming callbacks.

**Key Classes:**
- `StreamCallback`: Base streaming callback
  - `on_llm_start()`: Called when LLM starts
  - `on_llm_new_token(token)`: Called for each token
  - `on_llm_end(response)`: Called when LLM ends
  - `on_llm_error(error)`: Called on error
  - `get_accumulated_content()`: Get accumulated output
  - `get_session_statistics()`: Get session stats
- `MultiStreamCallback`: Multiple callback handler
- `BufferedStreamCallback`: Buffered streaming with batch processing
  - `flush_buffer()`: Flush buffered tokens

**Features:**
- LangChain integration
- Token-by-token processing
- Buffering support
- Error handling

---

## Error Handling

**Directory:** `ai_toolkit/errors/`

### Overview
The Error Handling toolkit provides comprehensive error management with custom exceptions and retry mechanisms.

### Modules

#### 1. `exception_types.py`
**Purpose:** Custom exception hierarchy for AI operations.

**Key Classes:**
- `ErrorSeverity`: Enum (LOW, MEDIUM, HIGH, CRITICAL)
- `ErrorCategory`: Enum (MODEL, PARSING, CONFIG, API, etc.)
- `AIException`: Base exception class
  - Fields: `message`, `error_code`, `severity`, `category`, `context`, `suggestions`
  - `to_dict()`: Convert to dictionary
- `ModelError`: Model-related errors
- `ParseError`: Parsing errors
- `ConfigError`: Configuration errors
- `APIError`: API communication errors
- `TimeoutError`: Timeout errors
- `RateLimitError`: Rate limiting errors
- `AuthenticationError`: Authentication errors
- `ValidationError`: Validation errors

**Functions:**
- `create_model_error()`: Create model error with suggestions
- `create_parse_error()`: Create parse error with suggestions
- `create_api_error()`: Create API error with suggestions

**Features:**
- Structured error information
- Severity levels
- Error categorization
- Suggested solutions


#### 2. `error_handler.py`
**Purpose:** Unified error handling with configurable rules.

**Key Classes:**
- `ErrorAction`: Enum (IGNORE, LOG, RETRY, FALLBACK, RAISE, NOTIFY)
- `ErrorRule`: Error handling rule
  - Fields: `error_type`, `action`, `max_retries`, `retry_delay`, `fallback_handler`
- `ErrorContext`: Error context information
  - Fields: `operation`, `timestamp`, `attempt`, `max_attempts`, `metadata`
- `ErrorHandler`: Central error handler
  - `add_rule(rule)`: Add error handling rule
  - `handle_error(error, context)`: Handle error
  - `handle_api_error(error, endpoint, status_code)`: Handle API errors
  - `handle_parse_error(error, parser_type, input_data)`: Handle parse errors
  - `handle_timeout(error, operation, timeout_duration)`: Handle timeouts
  - `get_error_statistics()`: Get error statistics
  - `get_recent_errors(limit)`: Get recent errors
  - `create_error_report()`: Create formatted report

**Functions:**
- `with_error_handling(handler, operation)`: Decorator for error handling
- `safe_execute(func, handler)`: Safely execute function

**Features:**
- Rule-based error handling
- Multiple actions (retry, fallback, etc.)
- Error statistics
- Custom handlers

#### 3. `retry_manager.py`
**Purpose:** Intelligent retry mechanisms with multiple strategies.

**Key Classes:**
- `RetryStrategy`: Enum (FIXED_DELAY, EXPONENTIAL_BACKOFF, LINEAR_BACKOFF, FIBONACCI_BACKOFF)
- `CircuitState`: Enum (CLOSED, OPEN, HALF_OPEN)
- `RetryConfig`: Retry configuration
  - Fields: `max_attempts`, `strategy`, `base_delay`, `max_delay`, `jitter`, `retryable_exceptions`
- `CircuitBreakerConfig`: Circuit breaker configuration
  - Fields: `failure_threshold`, `recovery_timeout`, `success_threshold`
- `RetryAttempt`: Retry attempt information
- `RetryManager`: Retry manager
  - `retry(func, config)`: Execute with retry
  - `retry_with_backoff(func, max_attempts)`: Exponential backoff retry
  - `retry_with_condition(func, condition)`: Conditional retry
  - `should_retry(exception, attempt)`: Check if should retry
  - `get_statistics()`: Get retry statistics
  - `reset_circuit_breaker()`: Reset circuit breaker

**Functions:**
- `retry(max_attempts, strategy)`: Decorator for retry
- `with_exponential_backoff(func)`: Execute with backoff
- `with_circuit_breaker(func)`: Execute with circuit breaker

**Features:**
- Multiple retry strategies
- Circuit breaker pattern
- Jitter support
- Conditional retry logic

---

## Token Management

**Directory:** `ai_toolkit/tokens/`

### Overview
The Token Management toolkit provides token counting, cost estimation, and optimization.

### Modules

#### 1. `token_counter.py`
**Purpose:** Count tokens and estimate costs for AI models.

**Key Classes:**
- `ModelType`: Enum for supported models (GPT-4, Claude, DeepSeek, Qwen, GLM, etc.)
- `TokenUsage`: Token usage information
  - Fields: `prompt_tokens`, `completion_tokens`, `total_tokens`
  - Properties: `input_tokens`, `output_tokens`
- `CostEstimate`: Cost estimation
  - Fields: `prompt_cost`, `completion_cost`, `total_cost`, `currency`, `model`
- `ModelPricing`: Model pricing information
  - Fields: `model`, `prompt_price_per_1k`, `completion_price_per_1k`
  - `calculate_cost(usage)`: Calculate cost from usage
- `TokenCounter`: Token counter
  - `count_tokens(text)`: Count tokens in text
  - `count_message_tokens(message)`: Count tokens in message
  - `count_messages_tokens(messages)`: Count tokens in message list
  - `estimate_completion_tokens(prompt)`: Estimate completion tokens
  - `estimate_cost(prompt_tokens, completion_tokens)`: Estimate cost
  - `analyze_text(text)`: Analyze text patterns
  - `compare_models(prompt_tokens, completion_tokens)`: Compare costs
  - `batch_count_tokens(texts)`: Count multiple texts
  - `calculate_conversation_cost(messages)`: Calculate conversation cost

**Features:**
- Multiple model support
- Tiktoken integration
- Cost estimation
- Batch processing

#### 2. `token_optimizer.py`
**Purpose:** Reduce token usage and optimize costs.

**Key Classes:**
- `OptimizationStrategy`: Enum (TRUNCATE_START, TRUNCATE_END, TRUNCATE_MIDDLE, SUMMARIZE, COMPRESS)
- `OptimizationResult`: Optimization result
  - Fields: `original_text`, `optimized_text`, `original_tokens`, `optimized_tokens`, `tokens_saved`, `compression_ratio`
  - Property: `savings_percentage`
- `TokenOptimizer`: Token optimizer
  - `truncate_text(text, max_tokens, strategy)`: Truncate text
  - `compress_text(text)`: Compress by removing redundancy
  - `optimize_messages(messages, max_tokens)`: Optimize message list
  - `summarize_context(text, target_tokens)`: Summarize text
  - `optimize_for_cost(text, max_cost)`: Optimize for cost limit
  - `batch_optimize(texts, max_tokens_per_text)`: Optimize multiple texts
  - `get_optimization_stats(results)`: Get optimization statistics

**Features:**
- Multiple optimization strategies
- Text compression
- Message optimization
- Cost-based optimization

---

## Utilities

**Directory:** `ai_toolkit/utils/`

### Overview
The Utilities toolkit provides common helper functions for logging, file operations, and general tasks.

### Modules

#### 1. `logger.py`
**Purpose:** Logging setup and utilities.

**Functions:**
- `setup_logger(name, level, log_file, log_format)`: Setup logger
  - Supports console and file logging
  - Rotating file handlers
  - Custom formats
- `get_logger(name)`: Get existing logger
- `log_info/log_error/log_warning/log_debug/log_critical(message)`: Convenience logging
- `set_log_level(level)`: Set logging level
- `disable_logging()/enable_logging()`: Toggle logging

**Features:**
- Console and file logging
- Log rotation (size and time-based)
- Custom formatting
- Multiple log levels

#### 2. `file_utils.py`
**Purpose:** File operation utilities.

**Functions:**
- `read_file(file_path)`: Read file content
- `write_file(file_path, content)`: Write content to file
- `read_lines(file_path)`: Read file as lines
- `write_lines(file_path, lines)`: Write lines to file
- `read_json(file_path)`: Read JSON file
- `write_json(file_path, data)`: Write JSON file
- `ensure_dir(dir_path)`: Ensure directory exists
- `delete_file(file_path)`: Delete file
- `delete_dir(dir_path, recursive)`: Delete directory
- `copy_file(src, dst)`: Copy file
- `copy_dir(src, dst)`: Copy directory
- `move_file(src, dst)`: Move file
- `file_exists(file_path)`: Check file existence
- `get_file_size(file_path)`: Get file size
- `list_files(dir_path, pattern, recursive)`: List files
- `list_dirs(dir_path)`: List directories
- `get_file_extension(file_path)`: Get file extension
- `get_file_name(file_path)`: Get file name
- `create_temp_file()/create_temp_dir()`: Create temporary files/dirs

**Features:**
- Comprehensive file operations
- JSON support
- Directory management
- Temporary file creation

#### 3. `utils.py`
**Purpose:** General utility functions.

**Functions:**
- `format_messages(messages)`: Format messages for display
- `validate_input(value, type, min, max, allowed)`: Validate input
- `sanitize_text(text)`: Clean text content
- `format_response(response, format_type)`: Format response
- `truncate_text(text, max_length)`: Truncate text
- `chunk_text(text, chunk_size, overlap)`: Split text into chunks
- `merge_dicts(*dicts, deep)`: Merge dictionaries
- `flatten_dict(d)`: Flatten nested dictionary
- `unflatten_dict(d)`: Unflatten dictionary
- `get_timestamp(format)`: Get current timestamp
- `parse_timestamp(timestamp)`: Parse timestamp
- `calculate_hash(text, algorithm)`: Calculate hash
- `retry_on_failure(func, max_attempts)`: Retry function
- `batch_process(items, batch_size, processor)`: Process in batches
- `filter_dict(d, keys, exclude_keys, predicate)`: Filter dictionary
- `safe_get(d, key, default)`: Safely get nested value
- `safe_set(d, key, value)`: Safely set nested value
- `is_empty(value)`: Check if empty
- `coalesce(*values)`: Return first non-None value

**Features:**
- Text processing
- Dictionary operations
- Timestamp handling
- Batch processing
- Safe operations

---

## Module Relationships

### Configuration Flow
```
EnvLoader → os.environ → ConfigManager → ConfigValidator → Application
```

### Model Creation Flow
```
ModelConfig → ModelProvider → ModelManager → LangChain Model
```

### Prompt Processing Flow
```
PromptLoader → PromptManager → PromptTemplate → Rendered Prompt
```

### Output Processing Flow
```
Model Output → OutputParser → ParserManager → Structured Data
```

### Streaming Flow
```
Model Stream → StreamCallback → StreamHandler → Aggregated Output
```

### Error Handling Flow
```
Exception → ErrorHandler → RetryManager → Recovery/Raise
```

### Token Management Flow
```
Text → TokenCounter → TokenOptimizer → Optimized Text
```

---

## Best Practices

### 1. Configuration Management
- Use environment variables for sensitive data
- Validate configurations before use
- Merge configurations for different environments

### 2. Model Management
- Cache model instances for reuse
- Use appropriate providers for your needs
- Configure timeouts and retries

### 3. Prompt Management
- Organize prompts by category
- Use templates for reusable prompts
- Validate variables before rendering

### 4. Output Parsing
- Use appropriate parser for output format
- Implement fallback parsers
- Handle parsing errors gracefully

### 5. Streaming
- Use buffering for high-frequency streams
- Track session statistics
- Handle stream interruptions

### 6. Error Handling
- Define error handling rules
- Use circuit breakers for external services
- Log errors with context

### 7. Token Management
- Count tokens before API calls
- Optimize for cost when possible
- Monitor token usage

### 8. Utilities
- Use logging consistently
- Handle file operations safely
- Validate inputs

---

## Quick Reference

### Import Patterns

```python
# Configuration
from ai_toolkit.config import ConfigManager, ConfigValidator, EnvLoader

# Models
from ai_toolkit.models import ModelManager, ModelConfig

# Prompts
from ai_toolkit.prompts import PromptManager, ChatPromptTemplate

# Parsers
from ai_toolkit.parsers import ParserManager, JsonOutputParser

# Streaming
from ai_toolkit.streaming import StreamHandler, StreamCallback

# Errors
from ai_toolkit.errors import ErrorHandler, RetryManager, AIException

# Tokens
from ai_toolkit.tokens import TokenCounter, TokenOptimizer

# Utils
from ai_toolkit.utils import setup_logger, read_json, merge_dicts
```

### Common Workflows

#### 1. Setup Configuration
```python
env_loader = EnvLoader('.env')
config_manager = ConfigManager('config/config.yaml')
validator = ConfigValidator()
validator.validate(config_manager.to_dict())
```

#### 2. Create Model
```python
model_manager = ModelManager('config/models.yaml')
model = model_manager.create_model('deepseek', 'deepseek-chat')
```

#### 3. Render Prompt
```python
prompt_manager = PromptManager('config/prompts')
rendered = prompt_manager.render_template('chat_template', user_input="Hello")
```

#### 4. Parse Output
```python
parser_manager = ParserManager()
parser_manager.create_json_parser('json_parser')
result = parser_manager.parse('json_parser', output_text)
```

#### 5. Handle Streaming
```python
stream_handler = StreamHandler()
callback = StreamCallback(stream_handler=stream_handler)
# Use callback with LangChain model
```

#### 6. Handle Errors
```python
error_handler = ErrorHandler()
retry_manager = RetryManager()
result = retry_manager.retry(risky_function)
```

#### 7. Manage Tokens
```python
counter = TokenCounter()
tokens = counter.count_tokens(text)
optimizer = TokenOptimizer(counter)
optimized = optimizer.compress_text(text)
```

---

## Summary

The AI Toolkit provides 8 comprehensive modules organized into logical directories:

1. **Configuration Management** - Load, validate, and manage configurations
2. **Model Management** - Unified interface for multiple AI providers
3. **Prompt Management** - Template-based prompt handling
4. **Output Parsing** - Convert outputs to structured data
5. **Streaming Processing** - Handle real-time streaming
6. **Error Handling** - Comprehensive error management
7. **Token Management** - Count, estimate, and optimize tokens
8. **Utilities** - Common helper functions

Each module is designed to work independently or as part of an integrated system, providing flexibility and power for AI application development.
