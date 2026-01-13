# AI工具包项目 (AI Toolkit)

## 简介

可复用的Python AI工具库，封装LangChain核心功能，为AI Agent项目提供基础能力支撑。

## 特性

- 🤖 **模型管理**: 统一管理多个AI模型（DeepSeek、Qwen、GLM等）
- 📝 **Prompt管理**: 模板化Prompt管理和渲染
- 🔄 **输出解析**: 结构化输出解析和验证
- 🌊 **流式处理**: 流式输出处理和回调
- 🛡️ **错误处理**: 完善的错误处理和重试机制
- 🎯 **Token管理**: Token统计和优化工具
- ⚙️ **配置管理**: 灵活的配置管理系统

## 安装

### 环境要求

- Python 3.11+
- macOS (推荐) / Linux / Windows

### 安装步骤

1. 克隆项目
```bash
git clone <repository-url>
cd ai-toolkit
```

2. 创建虚拟环境
```bash
conda create -n ai-toolkit python=3.11
conda activate ai-toolkit
```

3. 安装依赖
```bash
pip install -r requirements.txt
```

4. 开发模式安装
```bash
pip install -e .
```

5. 配置环境变量
```bash
cp .env.example .env
# 编辑 .env 文件，填入你的API密钥
```

## 快速开始

```python
from ai_toolkit import ModelManager, PromptManager, ParserManager

# 创建模型管理器
model_manager = ModelManager()
model = model_manager.create_model("deepseek")

# 使用Prompt模板
prompt_manager = PromptManager()
template = prompt_manager.get_template("system_chat")
prompt = template.render(user_input="你好，请介绍一下自己")

# 调用模型
response = model.invoke(prompt)

# 解析输出
parser_manager = ParserManager()
parser = parser_manager.create_parser("str")
result = parser.parse(response)

print(result)
```

## 项目结构

```
ai-toolkit/
├── ai_toolkit/           # 主要代码
│   ├── models/          # 模型管理
│   ├── prompts/         # Prompt管理
│   ├── parsers/         # 输出解析
│   ├── streaming/       # 流式处理
│   ├── errors/          # 错误处理
│   ├── tokens/          # Token工具
│   ├── config/          # 配置管理
│   └── utils/           # 工具类
├── tests/               # 测试代码
├── examples/            # 示例代码
├── docs/                # 文档
└── config/              # 配置文件
```

## 开发

### 运行测试

```bash
pytest tests/ -v --cov=ai_toolkit
```

### 代码格式化

```bash
black ai_toolkit/ tests/
```

### 代码检查

```bash
flake8 ai_toolkit/ tests/
mypy ai_toolkit/
```

## 许可证

MIT License

## 贡献

欢迎提交Issue和Pull Request！

## 更新日志

### v0.1.0 (开发中)

- 初始版本
- 基础模块实现
- 核心功能开发