# Configuration Management Toolkit Workflow

## Overview

The Configuration Management Toolkit consists of three interconnected components that work together to provide a complete configuration management solution:

1. **EnvLoader** - Loads environment variables from .env files
2. **ConfigManager** - Loads and manages YAML/JSON configuration files
3. **ConfigValidator** - Validates configuration values against rules

## Component Relationships

```
┌─────────────────────────────────────────────────────────────────┐
│                    Configuration Pipeline                        │
└─────────────────────────────────────────────────────────────────┘

Step 1: Load Environment Variables
┌──────────────────┐
│   EnvLoader      │
│                  │
│  .env file       │──┐
│  ├─ API_KEY=xxx  │  │
│  ├─ MODEL=gpt4   │  │  load_dotenv()
│  └─ DEBUG=true   │  │
└──────────────────┘  │
                      ▼
                ┌──────────────┐
                │ os.environ   │
                │ (System Env) │
                └──────────────┘
                      │
                      │ Environment variables available
                      ▼

Step 2: Load Configuration Files
┌──────────────────┐
│ ConfigManager    │
│                  │
│  config.yaml     │──┐
│  ├─ api_key:     │  │
│  │   ${API_KEY}  │  │  load_config()
│  ├─ model:       │  │  + substitute env vars
│  │   ${MODEL}    │  │
│  └─ debug:       │  │
│      ${DEBUG}    │  │
└──────────────────┘  │
                      ▼
                ┌──────────────┐
                │ Config Dict  │
                │ {            │
                │  api_key: xxx│
                │  model: gpt4 │
                │  debug: true │
                │ }            │
                └──────────────┘
                      │
                      │ Configuration loaded
                      ▼

Step 3: Validate Configuration
┌──────────────────┐
│ ConfigValidator  │
│                  │
│  Validation      │──┐
│  Rules:          │  │
│  ├─ api_key      │  │  validate()
│  │   required    │  │  + check rules
│  ├─ model        │  │
│  │   type: str   │  │
│  └─ debug        │  │
│      type: bool  │  │
└──────────────────┘  │
                      ▼
                ┌──────────────┐
                │ Validation   │
                │ Result       │
                │ ✓ All valid  │
                │ or           │
                │ ✗ Errors     │
                └──────────────┘
                      │
                      │ Ready to use
                      ▼
                ┌──────────────┐
                │ Application  │
                │ Uses Config  │
                └──────────────┘
```

## Detailed Data Flow

### 1. EnvLoader → os.environ

```
.env file                    EnvLoader                   os.environ
─────────                    ─────────                   ──────────
API_KEY=sk-xxx    ──load──>  load_env_file()  ──set──>  API_KEY=sk-xxx
MODEL=gpt-4                  (python-dotenv)             MODEL=gpt-4
DEBUG=true                                               DEBUG=true
```

**Methods Used:**
- `load_env_file()` - Loads .env file into environment
- `get()`, `get_str()`, `get_int()`, etc. - Type-safe access
- `get_api_key()` - Specialized API key retrieval

### 2. ConfigManager → Configuration Dictionary

```
config.yaml                  ConfigManager               Config Dict
───────────                  ─────────────               ───────────
api_key: ${API_KEY}  ──>    load_config()      ──>     api_key: sk-xxx
model: ${MODEL}              ├─ Parse YAML              model: gpt-4
debug: ${DEBUG}              └─ Substitute env vars     debug: true
```

**Methods Used:**
- `load_config()` - Loads YAML/JSON files
- `_substitute_env_vars()` - Replaces ${VAR} with env values
- `get_config()` - Retrieves config values with dot notation
- `set_config()` - Updates config values
- `merge_configs()` - Combines multiple config files

### 3. ConfigValidator → Validation Result

```
Config Dict                  ConfigValidator             Validation
───────────                  ───────────────             ──────────
api_key: sk-xxx    ──>      validate()         ──>     ✓ api_key valid
model: gpt-4                 ├─ Check rules             ✓ model valid
debug: true                  └─ Collect errors          ✓ debug valid
```

**Methods Used:**
- `add_rule()` - Adds validation rules
- `validate()` - Validates entire config
- `validate_model_config()` - Specialized model validation
- `get_errors()` - Retrieves validation errors

## Integration Patterns

### Pattern 1: Complete Pipeline (Recommended)

```python
from ai_toolkit.config import EnvLoader, ConfigManager, ConfigValidator

# Step 1: Load environment variables
env_loader = EnvLoader('.env')

# Step 2: Load configuration with env substitution
config_manager = ConfigManager('config/config.yaml')

# Step 3: Validate configuration
validator = ConfigValidator()
validator.add_rule('api_key', RequiredRule())
validator.add_rule('model', TypeRule(str))

if validator.validate(config_manager.to_dict()):
    print("Configuration is valid!")
    config = config_manager.get_config()
else:
    print("Validation errors:", validator.get_errors())
```

**Data Flow:**
```
.env → EnvLoader → os.environ → ConfigManager → Config Dict → ConfigValidator → ✓/✗
```

### Pattern 2: Direct Environment Access

```python
# Load environment variables only
env_loader = EnvLoader('.env')

# Access directly without config files
api_key = env_loader.get_api_key('deepseek')
model = env_loader.get_str('MODEL', default='gpt-4')
debug = env_loader.get_bool('DEBUG', default=False)
```

**Data Flow:**
```
.env → EnvLoader → Application
```

### Pattern 3: Config File Only (No .env)

```python
# Load config with hardcoded values
config_manager = ConfigManager('config/config.yaml')

# Validate
validator = ConfigValidator()
validator.validate_model_config(config_manager.to_dict())
```

**Data Flow:**
```
config.yaml → ConfigManager → ConfigValidator → Application
```

### Pattern 4: Runtime Validation

```python
# Load config
config_manager = ConfigManager('config/config.yaml')

# Add dynamic validation rules
validator = ConfigValidator()
validator.add_rule('temperature', RangeRule(0.0, 2.0))
validator.add_rule('max_tokens', RangeRule(1, 100000))

# Validate before each API call
if validator.validate(config_manager.to_dict()):
    # Make API call
    pass
```

**Data Flow:**
```
Config Dict → ConfigValidator (runtime) → ✓/✗ → API Call
```

## Component Independence

While these components work together, they can also be used independently:

### EnvLoader (Standalone)
```python
env = EnvLoader('.env')
api_key = env.get_api_key('deepseek')
# No ConfigManager or ConfigValidator needed
```

### ConfigManager (Standalone)
```python
config = ConfigManager('config.yaml')
value = config.get_config('models.deepseek.api_key')
# No EnvLoader or ConfigValidator needed
```

### ConfigValidator (Standalone)
```python
validator = ConfigValidator()
validator.add_rule('api_key', RequiredRule())
is_valid = validator.validate({'api_key': 'sk-xxx'})
# No EnvLoader or ConfigManager needed
```

## Complete Example

```python
from ai_toolkit.config import EnvLoader, ConfigManager, ConfigValidator
from ai_toolkit.config.config_validator import RequiredRule, TypeRule, RangeRule

# 1. Load environment variables from .env
env_loader = EnvLoader('.env', auto_load=True)

# 2. Load configuration file (with env var substitution)
config_manager = ConfigManager('config/models.yaml', auto_load=True)

# 3. Set up validation rules
validator = ConfigValidator()
validator.add_rule('deepseek.api_key', RequiredRule())
validator.add_rule('deepseek.model', TypeRule(str))
validator.add_rule('deepseek.temperature', RangeRule(0.0, 2.0))
validator.add_rule('deepseek.max_tokens', RangeRule(1, 100000))

# 4. Validate configuration
config_dict = config_manager.to_dict()
if validator.validate(config_dict):
    print("✓ Configuration is valid!")
    
    # 5. Use configuration
    api_key = config_manager.get_config('deepseek.api_key')
    model = config_manager.get_config('deepseek.model')
    
    # Make API calls with validated config
    print(f"Using model: {model}")
else:
    print("✗ Configuration errors:")
    for error in validator.get_errors():
        print(f"  - {error}")
```

## Key Features

### EnvLoader Features
- ✓ Loads .env files using python-dotenv
- ✓ Type-safe variable access (str, int, float, bool, list)
- ✓ API key retrieval with multiple naming conventions
- ✓ Fallback key support
- ✓ Required variable validation

### ConfigManager Features
- ✓ Loads YAML and JSON files
- ✓ Environment variable substitution (${VAR})
- ✓ Nested key access with dot notation
- ✓ Merge multiple config files
- ✓ Save/update/delete operations
- ✓ Reset to original state

### ConfigValidator Features
- ✓ Flexible validation rules (Required, Type, Range, Pattern, Choice, Custom)
- ✓ Nested key validation
- ✓ Error collection and reporting
- ✓ Specialized validators (model config, API keys, file paths, URLs)
- ✓ Pydantic model validation

## Summary

The three components form a complete configuration management pipeline:

1. **EnvLoader** loads sensitive data from .env files into environment
2. **ConfigManager** loads config files and substitutes environment variables
3. **ConfigValidator** validates the final configuration against rules

They can be used together for a complete solution or independently based on your needs.
