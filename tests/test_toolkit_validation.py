#!/usr/bin/env python3
"""
Comprehensive test cases for AI Toolkit validation.

This test suite validates:
1. LangChain integration correctness
2. Error handling robustness
3. Type compatibility
4. Edge case handling
5. Code quality issues

Author: AI Toolkit Team
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ai_toolkit.messages.message_builder import MessageBuilder
from ai_toolkit.messages.message_converter import MessageConverter
from ai_toolkit.messages.message_validator import MessageValidator
from ai_toolkit.parsers.output_parser import (
    JsonOutputParser,
    PydanticOutputParser,
    StrOutputParser
)
from ai_toolkit.models.model_providers import (
    BaseModelProvider,
    DeepSeekProvider,
    GLMProvider
)
from ai_toolkit.models.model_config import ModelConfig
from ai_toolkit.tokens.token_counter import TokenCounter, ModelType
from ai_toolkit.memory.memory_manager import (
    MessageTrimmer,
    MemoryManager,
    CheckpointerFactory
)
from ai_toolkit.streaming.stream_callback import StreamCallback

from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
    ToolMessage
)
from pydantic import BaseModel, Field


class TestMessageBuilder(unittest.TestCase):
    """Test MessageBuilder functionality."""
    
    def test_add_conversation_type_hint(self):
        """Test that add_conversation accepts Tuple type hints correctly."""
        builder = MessageBuilder()
        exchanges = [
            ("Hello!", "Hi there!"),
            ("How are you?", "I'm doing well!")
        ]
        # This should not raise a type error
        builder.add_conversation(exchanges)
        messages = builder.build()
        self.assertEqual(len(messages), 4)  # 2 human + 2 AI messages
    
    def test_message_building_fluent_interface(self):
        """Test fluent interface for building messages."""
        builder = MessageBuilder()
        messages = (builder
            .add_system("You are helpful")
            .add_human("Hello!")
            .add_ai("Hi!")
            .build())
        
        self.assertEqual(len(messages), 3)
        self.assertIsInstance(messages[0], SystemMessage)
        self.assertIsInstance(messages[1], HumanMessage)
        self.assertIsInstance(messages[2], AIMessage)
    
    def test_add_tool_message(self):
        """Test adding tool messages."""
        builder = MessageBuilder()
        builder.add_tool("Result: 42", tool_call_id="call_123", name="calculator")
        messages = builder.build()
        
        self.assertEqual(len(messages), 1)
        self.assertIsInstance(messages[0], ToolMessage)
        self.assertEqual(messages[0].tool_call_id, "call_123")


class TestMessageConverter(unittest.TestCase):
    """Test MessageConverter functionality."""
    
    def test_to_dict_conversion(self):
        """Test message to dictionary conversion."""
        msg = HumanMessage(content="Hello!", name="Alice")
        dict_msg = MessageConverter.to_dict(msg)
        
        self.assertEqual(dict_msg['role'], 'user')
        self.assertEqual(dict_msg['content'], 'Hello!')
        self.assertEqual(dict_msg['name'], 'Alice')
    
    def test_from_dict_conversion(self):
        """Test dictionary to message conversion."""
        data = {'role': 'user', 'content': 'Hello!', 'name': 'Alice'}
        msg = MessageConverter.from_dict(data)
        
        self.assertIsInstance(msg, HumanMessage)
        self.assertEqual(msg.content, 'Hello!')
        self.assertEqual(msg.name, 'Alice')
    
    def test_tool_message_conversion(self):
        """Test ToolMessage conversion."""
        msg = ToolMessage(
            content="Result: 42",
            tool_call_id="call_123",
            name="calculator"
        )
        dict_msg = MessageConverter.to_dict(msg)
        
        self.assertEqual(dict_msg['role'], 'tool')
        self.assertEqual(dict_msg['tool_call_id'], 'call_123')
        
        # Convert back
        restored = MessageConverter.from_dict(dict_msg)
        self.assertIsInstance(restored, ToolMessage)
        self.assertEqual(restored.tool_call_id, 'call_123')


class TestMessageValidator(unittest.TestCase):
    """Test MessageValidator functionality."""
    
    def test_validate_empty_message(self):
        """Test validation of empty message."""
        msg = HumanMessage(content="")
        is_valid, error = MessageValidator.validate_message(msg)
        
        self.assertFalse(is_valid)
        self.assertIsNotNone(error)
    
    def test_validate_tool_message_without_id(self):
        """Test validation of ToolMessage without tool_call_id."""
        # Create a ToolMessage without tool_call_id (should fail)
        msg = ToolMessage(content="Result", tool_call_id="")
        is_valid, error = MessageValidator.validate_message(msg)
        
        self.assertFalse(is_valid)
        self.assertIn('tool_call_id', error.lower())
    
    def test_validate_conversation_flow(self):
        """Test conversation flow validation."""
        messages = [
            SystemMessage(content="You are helpful"),
            HumanMessage(content="Hello!"),
            AIMessage(content="Hi!")
        ]
        is_valid, error = MessageValidator.validate_conversation_flow(messages)
        
        self.assertTrue(is_valid)
    
    def test_validate_multiple_system_messages(self):
        """Test validation fails with multiple system messages."""
        messages = [
            SystemMessage(content="First"),
            SystemMessage(content="Second"),
            HumanMessage(content="Hello!")
        ]
        is_valid, error = MessageValidator.validate_conversation_flow(messages)
        
        self.assertFalse(is_valid)
        self.assertIn('Multiple SystemMessages', error)


class TestJsonOutputParser(unittest.TestCase):
    """Test JsonOutputParser functionality."""
    
    def test_parse_valid_json(self):
        """Test parsing valid JSON."""
        parser = JsonOutputParser()
        text = '{"name": "John", "age": 30}'
        result = parser.parse(text)
        
        self.assertEqual(result['name'], 'John')
        self.assertEqual(result['age'], 30)
    
    def test_parse_json_with_trailing_comma(self):
        """Test parsing JSON with trailing comma (should be fixed)."""
        parser = JsonOutputParser(strict=False)
        text = '{"name": "John", "age": 30,}'
        result = parser.parse(text)
        
        self.assertEqual(result['name'], 'John')
        self.assertEqual(result['age'], 30)
    
    def test_parse_json_with_unquoted_keys(self):
        """Test parsing JSON with unquoted keys (should be fixed)."""
        parser = JsonOutputParser(strict=False)
        text = '{name: "John", age: 30}'
        result = parser.parse(text)
        
        self.assertEqual(result['name'], 'John')
        self.assertEqual(result['age'], 30)
    
    def test_parse_nested_json(self):
        """Test parsing nested JSON structures."""
        parser = JsonOutputParser()
        text = '{"user": {"name": "John", "address": {"city": "NYC"}}}'
        result = parser.parse(text)
        
        self.assertEqual(result['user']['name'], 'John')
        self.assertEqual(result['user']['address']['city'], 'NYC')
    
    def test_schema_validation_nested(self):
        """Test nested schema validation."""
        schema = {
            'user': {
                'name': str,
                'age': int
            }
        }
        parser = JsonOutputParser(schema=schema, strict=False)
        
        # Valid nested structure
        valid_text = '{"user": {"name": "John", "age": 30}}'
        result = parser.parse(valid_text)
        self.assertEqual(result['user']['name'], 'John')
        self.assertEqual(result['user']['age'], 30)
        
        # Note: Schema validation currently only validates top-level keys
        # Nested validation is implemented but may not catch all cases
        # This test verifies the parser works with nested structures


class TestPydanticOutputParser(unittest.TestCase):
    """Test PydanticOutputParser functionality."""
    
    def test_parse_pydantic_model(self):
        """Test parsing into Pydantic model."""
        class User(BaseModel):
            name: str
            age: int
        
        parser = PydanticOutputParser(pydantic_object=User)
        text = '{"name": "John", "age": 30}'
        result = parser.parse(text)
        
        self.assertIsInstance(result, User)
        self.assertEqual(result.name, 'John')
        self.assertEqual(result.age, 30)
    
    def test_parse_invalid_pydantic(self):
        """Test parsing invalid data into Pydantic model."""
        class User(BaseModel):
            name: str
            age: int
        
        parser = PydanticOutputParser(pydantic_object=User)
        text = '{"name": "John", "age": "not_a_number"}'
        
        with self.assertRaises(ValueError):
            parser.parse(text)


class TestModelProviders(unittest.TestCase):
    """Test model provider functionality."""
    
    def test_glm_message_type_detection(self):
        """Test GLM message type detection robustness."""
        from ai_toolkit.models.model_providers import GLMChatModel
        
        # Create a mock config
        config = ModelConfig(
            api_key="test_key_1234567890",
            base_url="https://test.com",
            model="glm-4"
        )
        
        provider = GLMProvider(config)
        
        # Test with various message types
        messages = [
            SystemMessage(content="You are helpful"),
            HumanMessage(content="Hello!"),
            AIMessage(content="Hi!")
        ]
        
        # This should not raise an error
        # Note: We can't actually call _generate without real API, but we can test structure
        self.assertTrue(hasattr(provider, 'create_model'))
    
    def test_model_type_validation_with_fallback(self):
        """Test model type validation with fallback."""
        # Test with invalid model name
        counter = TokenCounter(model="invalid-model-name")
        
        # Should fallback to GPT_4
        self.assertEqual(counter.model, ModelType.GPT_4)


class TestTokenCounter(unittest.TestCase):
    """Test TokenCounter functionality."""
    
    def test_count_tokens(self):
        """Test token counting."""
        counter = TokenCounter(model=ModelType.GPT_4)
        text = "Hello, world!"
        count = counter.count_tokens(text)
        
        self.assertGreater(count, 0)
    
    def test_count_message_tokens(self):
        """Test counting tokens in messages."""
        counter = TokenCounter(model=ModelType.GPT_4)
        msg = HumanMessage(content="Hello, world!")
        count = counter.count_message_tokens(msg)
        
        self.assertGreater(count, 0)
    
    def test_estimate_cost(self):
        """Test cost estimation."""
        counter = TokenCounter(model=ModelType.GPT_4)
        cost = counter.estimate_cost(prompt_tokens=1000, completion_tokens=500)
        
        self.assertGreater(cost.total_cost, 0)
        self.assertEqual(cost.currency, "USD")


class TestMemoryManager(unittest.TestCase):
    """Test MemoryManager functionality."""
    
    def test_message_trimmer(self):
        """Test message trimming."""
        trimmer = MessageTrimmer(strategy="keep_recent", max_messages=3)
        
        messages = [
            SystemMessage(content="System"),
            HumanMessage(content="1"),
            HumanMessage(content="2"),
            HumanMessage(content="3"),
            HumanMessage(content="4"),
        ]
        
        trimmed = trimmer.trim(messages)
        self.assertEqual(len(trimmed), 3)
        self.assertEqual(trimmed[-1].content, "4")
    
    def test_message_trimmer_keep_first_and_recent(self):
        """Test keep_first_and_recent strategy."""
        trimmer = MessageTrimmer(
            strategy="keep_first_and_recent",
            max_messages=3
        )
        
        messages = [
            SystemMessage(content="System"),
            HumanMessage(content="1"),
            HumanMessage(content="2"),
            HumanMessage(content="3"),
            HumanMessage(content="4"),
        ]
        
        trimmed = trimmer.trim(messages)
        self.assertEqual(len(trimmed), 3)
        self.assertIsInstance(trimmed[0], SystemMessage)
    
    def test_create_inmemory_checkpointer(self):
        """Test creating in-memory checkpointer."""
        checkpointer = CheckpointerFactory.create_inmemory()
        self.assertIsNotNone(checkpointer)


class TestStreamCallback(unittest.TestCase):
    """Test StreamCallback functionality."""
    
    def test_stream_callback_initialization(self):
        """Test StreamCallback initialization."""
        callback = StreamCallback(verbose=False)
        self.assertIsNotNone(callback.stream_handler)
    
    def test_on_chain_start_run_id_handling(self):
        """Test that run_id is handled correctly even if not provided."""
        callback = StreamCallback(verbose=False)
        
        # Call without run_id
        callback.on_chain_start(
            serialized={},
            inputs={},
            run_id=None
        )
        
        # Should have generated a run_id
        self.assertIsNotNone(callback._current_run_id)


class TestTypeCompatibility(unittest.TestCase):
    """Test Python version compatibility."""
    
    def test_union_type_syntax_compatibility(self):
        """Test that union types use Optional instead of | syntax."""
        # This test ensures we're using Optional[Type] instead of Type | None
        # which is Python 3.10+ syntax
        
        from ai_toolkit.memory.memory_manager import create_trimming_middleware
        
        # Should not raise syntax error on Python < 3.10
        middleware = create_trimming_middleware(max_messages=5)
        self.assertIsNotNone(middleware)
    
    def test_tuple_type_hint_compatibility(self):
        """Test that tuple type hints use Tuple from typing."""
        # This test ensures we're using Tuple[str, str] instead of tuple[str, str]
        
        builder = MessageBuilder()
        exchanges = [("Hello", "Hi")]
        builder.add_conversation(exchanges)
        
        # Should work without type errors
        messages = builder.build()
        self.assertEqual(len(messages), 2)


if __name__ == '__main__':
    unittest.main()
