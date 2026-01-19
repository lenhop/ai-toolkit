# AI工具包项目执行方案

## 项目概述

基于 `ai_agent_project_prompt_final.md` 项目1的设计要求，本方案详细规划AI工具包项目的具体实现，包括工具包结构、功能模块、代码组织、执行步骤等。

**项目目标**：构建可复用的Python AI工具库，封装LangChain核心功能，为后续AI Agent项目提供基础能力支撑。

**技术栈**：Python 3.11、LangChain、Pydantic、macOS系统

**项目周期**：3周

---

## 工具包功能设计

### 工具包分类清单

#### 1. 模型管理工具包

**基础工具包（必须）**

| 工具文件 | 类/函数 | 功能说明 |
|---------|---------|---------|
| `model_manager.py` | `ModelManager` | 统一管理多模型实例，支持模型切换和配置管理 |
| | `create_model()` | 创建模型实例（DeepSeek、Qwen、GLM等） |
| | `get_model()` | 获取指定模型实例 |
| | `list_models()` | 列出所有可用模型 |
| `model_providers.py` | `BaseModelProvider` | 模型提供者抽象基类 |
| | `DeepSeekProvider` | DeepSeek模型提供者 |
| | `QwenProvider` | 通义千问模型提供者 |
| | `GLMProvider` | GLM模型提供者 |
| `model_config.py` | `ModelConfig` | 模型配置数据类（Pydantic） |
| | `load_config()` | 从配置文件加载模型配置 |
| | `validate_config()` | 验证模型配置有效性 |

**高级工具包（可选）**

| 工具文件 | 类/函数 | 功能说明 |
|---------|---------|---------|
| `model_factory.py` | `ModelFactory` | 模型工厂类，支持动态创建和缓存 |
| | `create_with_cache()` | 创建带缓存的模型实例 |
| | `get_or_create()` | 获取或创建模型实例 |
| `model_monitor.py` | `ModelMonitor` | 模型调用监控，统计调用次数、延迟、错误率 |
| | `record_call()` | 记录模型调用信息 |
| | `get_statistics()` | 获取模型调用统计信息 |

---

#### 2. Prompt管理工具包

**基础工具包（必须）**

| 工具文件 | 类/函数 | 功能说明 |
|---------|---------|---------|
| `prompt_manager.py` | `PromptManager` | Prompt模板管理器 |
| | `load_template()` | 加载Prompt模板文件 |
| | `get_template()` | 获取指定模板 |
| | `render_template()` | 渲染模板（变量替换） |
| | `save_template()` | 保存模板到文件 |
| `prompt_templates.py` | `BasePromptTemplate` | Prompt模板基类 |
| | `ChatPromptTemplate` | 聊天Prompt模板 |
| | `SystemPromptTemplate` | 系统Prompt模板 |
| | `FewShotPromptTemplate` | 少样本Prompt模板 |
| `prompt_loader.py` | `PromptLoader` | 从文件系统加载Prompt模板 |
| | `load_from_file()` | 从文件加载模板 |
| | `load_from_dir()` | 从目录批量加载模板 |
| | `validate_template()` | 验证模板格式 |

**高级工具包（可选）**

| 工具文件 | 类/函数 | 功能说明 |
|---------|---------|---------|
| `prompt_version.py` | `PromptVersionManager` | Prompt版本管理 |
| | `create_version()` | 创建模板版本 |
| | `get_version()` | 获取指定版本 |
| | `list_versions()` | 列出所有版本 |
| `prompt_optimizer.py` | `PromptOptimizer` | Prompt优化工具 |
| | `optimize_length()` | 优化Prompt长度 |
| | `add_examples()` | 添加示例到Prompt |
| | `test_prompt()` | 测试Prompt效果 |

---

#### 3. 输出解析工具包

**基础工具包（必须）**

| 工具文件 | 类/函数 | 功能说明 |
|---------|---------|---------|
| `output_parser.py` | `BaseOutputParser` | 输出解析器基类 |
| | `PydanticOutputParser` | Pydantic模型解析器 |
| | `JsonOutputParser` | JSON解析器 |
| | `StrOutputParser` | 字符串解析器 |
| `parser_manager.py` | `ParserManager` | 解析器管理器 |
| | `create_parser()` | 创建解析器实例 |
| | `parse()` | 解析输出内容 |
| | `validate_output()` | 验证输出格式 |
| `error_handler.py` | `OutputErrorHandler` | 输出解析错误处理 |
| | `fix_json()` | 修复JSON格式错误 |
| | `retry_parse()` | 重试解析 |
| | `handle_error()` | 处理解析错误 |

**高级工具包（可选）**

| 工具文件 | 类/函数 | 功能说明 |
|---------|---------|---------|
| `parser_chain.py` | `ParserChain` | 解析器链，支持多级解析 |
| | `add_parser()` | 添加解析器到链 |
| | `parse_chain()` | 链式解析 |
| `output_validator.py` | `OutputValidator` | 输出验证器 |
| | `validate_schema()` | 验证输出是否符合Schema |
| | `validate_content()` | 验证输出内容质量 |

---

#### 4. 流式处理工具包

**基础工具包（必须）**

| 工具文件 | 类/函数 | 功能说明 |
|---------|---------|---------|
| `stream_handler.py` | `StreamHandler` | 流式输出处理器 |
| | `handle_stream()` | 处理流式输出 |
| | `format_chunk()` | 格式化流式数据块 |
| | `aggregate_stream()` | 聚合流式输出 |

**注意**: 流式回调请直接使用 LangChain 的 `BaseCallbackHandler`

**高级工具包（可选）**

| 工具文件 | 类/函数 | 功能说明 |
|---------|---------|---------|
| `stream_buffer.py` | `StreamBuffer` | 流式数据缓冲区 |
| | `add_chunk()` | 添加数据块到缓冲区 |
| | `get_buffer()` | 获取缓冲区内容 |
| | `clear_buffer()` | 清空缓冲区 |
| `stream_analyzer.py` | `StreamAnalyzer` | 流式输出分析器 |
| | `analyze_speed()` | 分析流式输出速度 |
| | `analyze_quality()` | 分析流式输出质量 |

---

#### 5. 错误处理工具包

**基础工具包（必须）**

| 工具文件 | 类/函数 | 功能说明 |
|---------|---------|---------|
| `error_handler.py` | `ErrorHandler` | 统一错误处理器 |
| | `handle_api_error()` | 处理API调用错误 |
| | `handle_parse_error()` | 处理解析错误 |
| | `handle_timeout()` | 处理超时错误 |
| `retry_manager.py` | `RetryManager` | 重试管理器 |
| | `retry_with_backoff()` | 指数退避重试 |
| | `retry_with_condition()` | 条件重试 |
| | `should_retry()` | 判断是否应该重试 |
| `exception_types.py` | `AIException` | AI相关异常基类 |
| | `ModelError` | 模型调用异常 |
| | `ParseError` | 解析异常 |
| | `ConfigError` | 配置异常 |

**高级工具包（可选）**

| 工具文件 | 类/函数 | 功能说明 |
|---------|---------|---------|
| `error_recovery.py` | `ErrorRecovery` | 错误恢复策略 |
| | `recover_from_error()` | 从错误中恢复 |
| | `fallback_strategy()` | 降级策略 |
| `error_logger.py` | `ErrorLogger` | 错误日志记录器 |
| | `log_error()` | 记录错误日志 |
| | `get_error_stats()` | 获取错误统计信息 |

---

#### 6. Token工具包

**基础工具包（必须）**

| 工具文件 | 类/函数 | 功能说明 |
|---------|---------|---------|
| `token_counter.py` | `TokenCounter` | Token计数器 |
| | `count_tokens()` | 统计文本Token数量 |
| | `count_messages()` | 统计消息列表Token数量 |
| | `estimate_cost()` | 估算API调用成本 |
| `token_optimizer.py` | `TokenOptimizer` | Token优化器 |
| | `truncate_text()` | 截断文本以控制Token数 |
| | `compress_messages()` | 压缩消息列表 |
| | `summarize_context()` | 总结上下文以减少Token |

**高级工具包（可选）**

| 工具文件 | 类/函数 | 功能说明 |
|---------|---------|---------|
| `token_analyzer.py` | `TokenAnalyzer` | Token分析器 |
| | `analyze_distribution()` | 分析Token分布 |
| | `find_expensive_tokens()` | 找出消耗Token较多的内容 |
| `token_cache.py` | `TokenCache` | Token缓存管理器 |
| | `cache_result()` | 缓存Token计算结果 |
| | `get_cached_count()` | 获取缓存的Token数量 |

---

#### 7. 配置管理工具包

**基础工具包（必须）**

| 工具文件 | 类/函数 | 功能说明 |
|---------|---------|---------|
| `config_manager.py` | `ConfigManager` | 配置管理器 |
| | `load_config()` | 加载配置文件（YAML/JSON） |
| | `get_config()` | 获取配置项 |
| | `set_config()` | 设置配置项 |
| | `save_config()` | 保存配置到文件 |
| `config_validator.py` | `ConfigValidator` | 配置验证器 |
| | `validate_model_config()` | 验证模型配置 |
| | `validate_api_keys()` | 验证API密钥 |
| `env_loader.py` | `EnvLoader` | 环境变量加载器 |
| | `load_from_env()` | 从环境变量加载配置 |
| | `get_api_key()` | 获取API密钥（优先环境变量） |

---

#### 8. 工具类工具包

**基础工具包（必须）**

| 工具文件 | 类/函数 | 功能说明 |
|---------|---------|---------|
| `utils.py` | `format_messages()` | 格式化消息列表 |
| | `validate_input()` | 验证输入参数 |
| | `sanitize_text()` | 清理文本内容 |
| | `format_response()` | 格式化响应内容 |
| `logger.py` | `setup_logger()` | 设置日志记录器 |
| | `log_info()` | 记录信息日志 |
| | `log_error()` | 记录错误日志 |
| | `log_debug()` | 记录调试日志 |
| `file_utils.py` | `read_file()` | 读取文件内容 |
| | `write_file()` | 写入文件内容 |
| | `ensure_dir()` | 确保目录存在 |

---

## 项目目录结构

```
/Users/hzz/KMS/ai-toolkit/
├── README.md                    # 项目说明文档
├── requirements.txt             # Python依赖
├── setup.py                     # 安装配置
├── pyproject.toml               # 项目配置
├── .env.example                 # 环境变量示例
├── config/
│   ├── __init__.py
│   ├── config.yaml              # 配置文件
│   └── models.yaml              # 模型配置
├── ai_toolkit/
│   ├── __init__.py
│   ├── models/                  # 模型管理模块
│   │   ├── __init__.py
│   │   ├── model_manager.py
│   │   ├── model_providers.py
│   │   ├── model_config.py
│   │   ├── model_factory.py      # 高级
│   │   └── model_monitor.py      # 高级
│   ├── prompts/                 # Prompt管理模块
│   │   ├── __init__.py
│   │   ├── prompt_manager.py
│   │   ├── prompt_templates.py
│   │   ├── prompt_loader.py
│   │   ├── prompt_version.py     # 高级
│   │   └── prompt_optimizer.py   # 高级
│   ├── parsers/                 # 输出解析模块
│   │   ├── __init__.py
│   │   ├── output_parser.py
│   │   ├── parser_manager.py
│   │   ├── error_handler.py
│   │   ├── parser_chain.py       # 高级
│   │   └── output_validator.py  # 高级
│   ├── streaming/               # 流式处理模块
│   │   ├── __init__.py
│   │   ├── stream_handler.py
│   │   ├── stream_callback.py
│   │   ├── stream_buffer.py      # 高级
│   │   └── stream_analyzer.py   # 高级
│   ├── errors/                  # 错误处理模块
│   │   ├── __init__.py
│   │   ├── error_handler.py
│   │   ├── retry_manager.py
│   │   ├── exception_types.py
│   │   ├── error_recovery.py     # 高级
│   │   └── error_logger.py      # 高级
│   ├── tokens/                  # Token工具模块
│   │   ├── __init__.py
│   │   ├── token_counter.py
│   │   ├── token_optimizer.py
│   │   ├── token_analyzer.py     # 高级
│   │   └── token_cache.py        # 高级
│   ├── config/                  # 配置管理模块
│   │   ├── __init__.py
│   │   ├── config_manager.py
│   │   ├── config_validator.py
│   │   └── env_loader.py
│   └── utils/                   # 工具类模块
│       ├── __init__.py
│       ├── utils.py
│       ├── logger.py
│       └── file_utils.py
├── tests/                       # 测试目录
│   ├── __init__.py
│   ├── test_models/
│   ├── test_prompts/
│   ├── test_parsers/
│   └── test_utils/
├── examples/                    # 示例代码
│   ├── basic_usage.py
│   ├── advanced_usage.py
│   └── integration_example.py
└── docs/                        # 文档目录
    ├── api_reference.md
    └── usage_guide.md
```

---

## 详细执行步骤

### 第1周：项目搭建和基础模块实现

#### Day 1: 项目初始化

**任务清单**：
1. 创建项目目录结构
2. 初始化Python虚拟环境
3. 创建requirements.txt和配置文件
4. 设置Git仓库

**具体执行**：

```bash
# 1. 创建项目目录
mkdir -p ai_toolkit/{ai_toolkit/{models,prompts,parsers,streaming,errors,tokens,config,utils},tests/{test_models,test_prompts,test_parsers,test_utils},examples,docs,config}

# 2. 创建Python虚拟环境（macOS）
cd /Users/hzz/KMS/ai-toolkit
conda activate py311

# 3. 创建requirements.txt
cat > requirements.txt << EOF
langchain>=0.1.0
langchain-core>=0.1.0
langchain-community>=0.0.20
pydantic>=2.0.0
pyyaml>=6.0
python-dotenv>=1.0.0
tiktoken>=0.5.0
openai>=1.0.0
anthropic>=0.7.0
zhipuai>=2.0.0
pytest>=7.4.0
pytest-cov>=4.1.0
black>=23.0.0
flake8>=6.0.0
mypy>=1.5.0
EOF

# 4. 安装依赖
pip install -r requirements.txt

# 5. 初始化Git仓库
git init
git add .
git commit -m "Initial project setup"
```

**验收标准**：
- [ ] 项目目录结构完整
- [ ] 虚拟环境创建成功
- [ ] 依赖包安装成功
- [ ] Git仓库初始化完成

---

#### Day 2-3: 实现配置管理模块

**任务清单**：
1. 实现ConfigManager类
2. 实现ConfigValidator类
3. 实现EnvLoader类
4. 创建配置文件模板

**具体执行**：

```bash
# 1. 创建配置管理模块文件
touch ai_toolkit/config/{__init__.py,config_manager.py,config_validator.py,env_loader.py}

# 2. 实现config_manager.py
# 实现load_config(), get_config(), set_config(), save_config()方法

# 3. 实现config_validator.py
# 实现validate_model_config(), validate_api_keys()方法

# 4. 实现env_loader.py
# 实现load_from_env(), get_api_key()方法

# 5. 创建配置文件模板
cat > config/config.yaml << EOF
models:
  deepseek:
    api_key: ${DEEPSEEK_API_KEY}
    base_url: https://api.deepseek.com
    model: deepseek-chat
  qwen:
    api_key: ${QWEN_API_KEY}
    base_url: https://dashscope.aliyuncs.com
    model: qwen-turbo
  glm:
    api_key: ${GLM_API_KEY}
    base_url: https://open.bigmodel.cn
    model: glm-4
EOF

# 6. 创建.env.example
cat > .env.example << EOF
DEEPSEEK_API_KEY=your_deepseek_api_key
QWEN_API_KEY=your_qwen_api_key
GLM_API_KEY=your_glm_api_key
EOF
```

**验收标准**：
- [ ] ConfigManager类实现完成
- [ ] 配置文件加载功能正常
- [ ] 环境变量加载功能正常
- [ ] 配置验证功能正常

---

#### Day 4-5: 实现模型管理模块（基础）

**任务清单**：
1. 实现ModelConfig数据类
2. 实现BaseModelProvider抽象类
3. 实现DeepSeekProvider、QwenProvider、GLMProvider
4. 实现ModelManager类

**具体执行**：

```bash
# 1. 创建模型管理模块文件
touch ai_toolkit/models/{__init__.py,model_config.py,model_providers.py,model_manager.py}

# 2. 实现model_config.py
# 使用Pydantic定义ModelConfig数据类
# 包含api_key, base_url, model, temperature等字段

# 3. 实现model_providers.py
# 实现BaseModelProvider抽象基类
# 实现create_model()抽象方法
# 实现DeepSeekProvider（使用langchain_community.chat_models）
# 实现QwenProvider（使用langchain_community.chat_models）
# 实现GLMProvider（使用langchain_community.chat_models）

# 4. 实现model_manager.py
# 实现ModelManager类
# 实现create_model()方法：根据provider名称创建模型实例
# 实现get_model()方法：获取已创建的模型实例
# 实现list_models()方法：列出所有可用模型

# 5. 编写单元测试
touch tests/test_models/test_model_manager.py
# 测试模型创建、获取、列表功能
```

**验收标准**：
- [ ] ModelConfig数据类定义完成
- [ ] 三个模型提供者实现完成
- [ ] ModelManager类实现完成
- [ ] 单元测试通过

---

#### Day 6-7: 实现Prompt管理模块（基础）

**任务清单**：
1. 实现PromptLoader类
2. 实现Prompt模板类（BasePromptTemplate、ChatPromptTemplate等）
3. 实现PromptManager类
4. 创建示例Prompt模板文件

**具体执行**：

```bash
# 1. 创建Prompt管理模块文件
touch ai_toolkit/prompts/{__init__.py,prompt_loader.py,prompt_templates.py,prompt_manager.py}

# 2. 实现prompt_loader.py
# 实现load_from_file()：从文件加载模板
# 实现load_from_dir()：从目录批量加载
# 实现validate_template()：验证模板格式

# 3. 实现prompt_templates.py
# 实现BasePromptTemplate基类
# 实现ChatPromptTemplate：封装langchain的ChatPromptTemplate
# 实现SystemPromptTemplate：系统提示模板
# 实现FewShotPromptTemplate：少样本模板

# 4. 实现prompt_manager.py
# 实现load_template()：加载模板文件
# 实现get_template()：获取指定模板
# 实现render_template()：渲染模板（变量替换）
# 实现save_template()：保存模板到文件

# 5. 创建示例模板文件
mkdir -p config/prompts
cat > config/prompts/system_chat.yaml << EOF
name: system_chat
description: 通用聊天系统提示
template: |
  你是一个有用的AI助手。请根据用户的问题提供准确、有帮助的回答。
  
  用户问题：{user_input}
EOF

# 6. 编写单元测试
touch tests/test_prompts/test_prompt_manager.py
```

**验收标准**：
- [ ] PromptLoader类实现完成
- [ ] Prompt模板类实现完成
- [ ] PromptManager类实现完成
- [ ] 模板加载和渲染功能正常
- [ ] 单元测试通过

---

### 第2周：核心功能实现

#### Day 8-9: 实现输出解析模块（基础）

**任务清单**：
1. 实现BaseOutputParser抽象类
2. 实现PydanticOutputParser、JsonOutputParser、StrOutputParser
3. 实现ParserManager类
4. 实现OutputErrorHandler类

**具体执行**：

```bash
# 1. 创建输出解析模块文件
touch ai_toolkit/parsers/{__init__.py,output_parser.py,parser_manager.py,error_handler.py}

# 2. 实现output_parser.py
# 实现BaseOutputParser抽象基类
# 实现PydanticOutputParser：使用langchain的PydanticOutputParser
# 实现JsonOutputParser：使用langchain的JsonOutputParser
# 实现StrOutputParser：字符串解析器

# 3. 实现parser_manager.py
# 实现create_parser()：根据类型创建解析器
# 实现parse()：解析输出内容
# 实现validate_output()：验证输出格式

# 4. 实现error_handler.py
# 实现fix_json()：修复JSON格式错误（缺失引号、尾随逗号等）
# 实现retry_parse()：重试解析
# 实现handle_error()：统一错误处理

# 5. 编写单元测试
touch tests/test_parsers/test_output_parser.py
# 测试各种解析器功能
# 测试错误处理和修复功能
```

**验收标准**：
- [ ] 输出解析器类实现完成
- [ ] ParserManager类实现完成
- [ ] 错误处理功能实现完成
- [ ] JSON修复功能正常
- [ ] 单元测试通过

---

#### Day 10-11: 实现流式处理模块（基础）

**任务清单**：
1. 实现StreamHandler类
2. 集成LangChain流式输出功能

**具体执行**：

```bash
# 1. 创建流式处理模块文件
touch ai_toolkit/streaming/{__init__.py,stream_handler.py}

# 2. 实现stream_handler.py
# 实现handle_stream()：处理流式输出
# 实现format_chunk()：格式化数据块
# 实现aggregate_stream()：聚合流式输出为完整文本

# 3. 流式回调使用 LangChain 的 BaseCallbackHandler
# 用户可以直接继承 langchain_core.callbacks.BaseCallbackHandler

# 4. 编写示例代码
cat > examples/streaming_example.py << EOF
from ai_toolkit.models import ModelManager
from ai_toolkit.streaming import StreamHandler
from langchain_core.callbacks import BaseCallbackHandler

manager = ModelManager()
model = manager.create_model("deepseek")

class PrintCallback(BaseCallbackHandler):
    def on_llm_new_token(self, token: str, **kwargs):
        print(token, end='', flush=True)

callback = PrintCallback()
model.invoke("你好", config={"callbacks": [callback]})
EOF

# 5. 编写单元测试
touch tests/test_streaming/test_stream_handler.py
```

**验收标准**：
- [ ] StreamHandler类实现完成
- [ ] 流式输出处理功能正常
- [ ] 示例代码运行成功
- [ ] 单元测试通过

---

#### Day 12-13: 实现错误处理模块（基础）

**任务清单**：
1. 实现异常类型定义
2. 实现ErrorHandler类
3. 实现RetryManager类

**具体执行**：

```bash
# 1. 创建错误处理模块文件
touch ai_toolkit/errors/{__init__.py,exception_types.py,error_handler.py,retry_manager.py}

# 2. 实现exception_types.py
# 定义AIException基类
# 定义ModelError：模型调用异常
# 定义ParseError：解析异常
# 定义ConfigError：配置异常

# 3. 实现error_handler.py
# 实现handle_api_error()：处理API调用错误（认证失败、频率限制、超时等）
# 实现handle_parse_error()：处理解析错误
# 实现handle_timeout()：处理超时错误

# 4. 实现retry_manager.py
# 实现retry_with_backoff()：指数退避重试（1s, 2s, 4s...）
# 实现retry_with_condition()：条件重试（根据错误类型决定是否重试）
# 实现should_retry()：判断是否应该重试

# 5. 编写单元测试
touch tests/test_errors/test_error_handler.py
# 测试各种错误处理场景
# 测试重试机制
```

**验收标准**：
- [ ] 异常类型定义完成
- [ ] ErrorHandler类实现完成
- [ ] RetryManager类实现完成
- [ ] 错误处理功能正常
- [ ] 重试机制工作正常
- [ ] 单元测试通过

---

#### Day 14: 实现Token工具模块（基础）

**任务清单**：
1. 实现TokenCounter类
2. 实现TokenOptimizer类
3. 集成tiktoken库

**具体执行**：

```bash
# 1. 创建Token工具模块文件
touch ai_toolkit/tokens/{__init__.py,token_counter.py,token_optimizer.py}

# 2. 实现token_counter.py
# 实现count_tokens()：使用tiktoken统计Token数量
# 实现count_messages()：统计消息列表Token数量
# 实现estimate_cost()：根据Token数量和模型价格估算成本

# 3. 实现token_optimizer.py
# 实现truncate_text()：截断文本以控制Token数
# 实现compress_messages()：压缩消息列表（保留重要消息）
# 实现summarize_context()：总结上下文以减少Token

# 4. 编写单元测试
touch tests/test_tokens/test_token_counter.py
# 测试Token统计功能
# 测试Token优化功能
```

**验收标准**：
- [ ] TokenCounter类实现完成
- [ ] TokenOptimizer类实现完成
- [ ] Token统计功能正常
- [ ] Token优化功能正常
- [ ] 单元测试通过

---

### 第3周：完善和优化

#### Day 15-16: 实现工具类模块和集成测试

**任务清单**：
1. 实现utils.py工具函数
2. 实现logger.py日志工具
3. 实现file_utils.py文件工具
4. 编写集成测试

**具体执行**：

```bash
# 1. 创建工具类模块文件
touch ai_toolkit/utils/{__init__.py,utils.py,logger.py,file_utils.py}

# 2. 实现utils.py
# 实现format_messages()：格式化消息列表
# 实现validate_input()：验证输入参数
# 实现sanitize_text()：清理文本内容
# 实现format_response()：格式化响应内容

# 3. 实现logger.py
# 实现setup_logger()：设置日志记录器（使用logging模块）
# 实现log_info(), log_error(), log_debug()：日志记录函数

# 4. 实现file_utils.py
# 实现read_file()：读取文件内容
# 实现write_file()：写入文件内容
# 实现ensure_dir()：确保目录存在

# 5. 编写集成测试
touch tests/test_integration.py
# 测试完整流程：模型调用 -> Prompt渲染 -> 输出解析
# 测试错误处理和重试机制
# 测试流式输出处理
```

**验收标准**：
- [ ] 工具类模块实现完成
- [ ] 日志功能正常
- [ ] 文件操作功能正常
- [ ] 集成测试通过

---

#### Day 17-18: 实现高级功能模块（可选）

**任务清单**：
1. 实现ModelFactory和ModelMonitor
2. 实现PromptVersionManager
3. 实现ParserChain和OutputValidator
4. 实现StreamBuffer和StreamAnalyzer

**具体执行**：

```bash
# 1. 实现高级模型管理功能
touch ai_toolkit/models/{model_factory.py,model_monitor.py}
# 实现ModelFactory：模型工厂和缓存
# 实现ModelMonitor：模型调用监控

# 2. 实现高级Prompt功能
touch ai_toolkit/prompts/{prompt_version.py,prompt_optimizer.py}
# 实现PromptVersionManager：版本管理
# 实现PromptOptimizer：Prompt优化

# 3. 实现高级解析功能
touch ai_toolkit/parsers/{parser_chain.py,output_validator.py}
# 实现ParserChain：解析器链
# 实现OutputValidator：输出验证器

# 4. 实现高级流式处理功能
touch ai_toolkit/streaming/{stream_buffer.py,stream_analyzer.py}
# 实现StreamBuffer：流式数据缓冲区
# 实现StreamAnalyzer：流式输出分析器
```

**验收标准**：
- [ ] 高级功能模块实现完成（可选）
- [ ] 功能测试通过

---

#### Day 19-20: 编写文档和示例代码

**任务清单**：
1. 编写README.md
2. 编写API参考文档
3. 编写使用指南
4. 编写示例代码

**具体执行**：

```bash
# 1. 编写README.md
cat > README.md << EOF
# AI工具包项目

## 简介
可复用的Python AI工具库，封装LangChain核心功能。

## 安装
pip install -e .

## 快速开始
[示例代码]
EOF

# 2. 编写API参考文档
touch docs/api_reference.md
# 列出所有类和函数的API文档

# 3. 编写使用指南
touch docs/usage_guide.md
# 编写详细的使用示例

# 4. 编写示例代码
cat > examples/basic_usage.py << EOF
# 基础使用示例
from ai_toolkit import ModelManager, PromptManager, ParserManager

# 创建模型
manager = ModelManager()
model = manager.create_model("deepseek")

# 使用Prompt模板
prompt_mgr = PromptManager()
template = prompt_mgr.get_template("system_chat")
prompt = template.render(user_input="你好")

# 调用模型
response = model.invoke(prompt)

# 解析输出
parser_mgr = ParserManager()
parser = parser_mgr.create_parser("json")
result = parser.parse(response)
EOF
```

**验收标准**：
- [ ] README.md编写完成
- [ ] API文档编写完成
- [ ] 使用指南编写完成
- [ ] 示例代码编写完成

---

#### Day 21: 代码优化和最终测试

**任务清单**：
1. 代码格式化（black）
2. 代码检查（flake8）
3. 类型检查（mypy）
4. 运行所有测试
5. 性能优化

**具体执行**：

```bash
# 1. 代码格式化
black ai_toolkit/ tests/

# 2. 代码检查
flake8 ai_toolkit/ tests/

# 3. 类型检查
mypy ai_toolkit/

# 4. 运行所有测试
pytest tests/ -v --cov=ai_toolkit --cov-report=html

# 5. 检查测试覆盖率
# 目标：基础模块覆盖率 > 80%

# 6. 性能测试
# 测试模型调用性能
# 测试Token统计性能
# 测试解析性能
```

**验收标准**：
- [ ] 代码格式化完成
- [ ] 代码检查通过
- [ ] 类型检查通过
- [ ] 所有测试通过
- [ ] 测试覆盖率达标
- [ ] 性能满足要求

---

## 项目交付清单

### 代码交付
- [ ] 所有基础工具包模块实现完成
- [ ] 所有高级工具包模块实现完成（可选）
- [ ] 单元测试覆盖率 > 80%
- [ ] 集成测试通过
- [ ] 代码符合PEP8规范

### 文档交付
- [ ] README.md：项目说明和快速开始
- [ ] API参考文档：所有类和函数的API文档
- [ ] 使用指南：详细的使用示例和最佳实践
- [ ] 示例代码：基础使用和高级使用示例

### 配置交付
- [ ] requirements.txt：Python依赖列表
- [ ] setup.py：安装配置
- [ ] config.yaml：配置文件模板
- [ ] .env.example：环境变量示例

---

## 技术要点

### Python 3.11特性使用
- 使用类型提示（Type Hints）提高代码可读性
- 使用dataclass和Pydantic定义数据模型
- 使用async/await处理异步操作（如需要）

### macOS系统注意事项
- 使用Python 3.11虚拟环境
- 注意文件路径分隔符（使用pathlib处理）
- 注意权限问题（API密钥文件权限）

### LangChain最佳实践
- 使用langchain-core的统一接口
- 合理使用回调处理器
- 正确处理流式输出
- 实现错误处理和重试机制

---

## 常见问题

### Q1: 如何处理API密钥安全？
**A**: 使用环境变量存储API密钥，不要提交到Git仓库。使用.env文件管理本地开发环境。

### Q2: 如何选择基础工具包还是高级工具包？
**A**: 基础工具包必须实现，高级工具包根据实际需求和时间安排决定是否实现。

### Q3: 如何测试模型调用功能？
**A**: 使用mock对象模拟API调用，避免实际调用API产生费用。

### Q4: 如何处理不同模型的接口差异？
**A**: 使用LangChain的统一接口封装，在Provider层处理差异。

---

*本方案设计时间：2025年1月*  
*预计完成时间：3周（21个工作日）*
