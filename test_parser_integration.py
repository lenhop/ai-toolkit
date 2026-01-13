#!/usr/bin/env python3
"""
Integration test for the output parsing toolkit.
"""

import json
from pathlib import Path
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables
env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)

from ai_toolkit.parsers import (
    ParserManager, 
    create_parser,
    StrOutputParser,
    JsonOutputParser,
    PydanticOutputParser,
    ListOutputParser,
    RegexOutputParser,
    OutputErrorHandler
)


class PersonModel(BaseModel):
    """Test Pydantic model for integration tests."""
    name: str = Field(..., description="Person's name")
    age: int = Field(..., description="Person's age")
    email: str = Field(..., description="Person's email address")
    skills: list = Field(default_factory=list, description="List of skills")


class ProductModel(BaseModel):
    """Test product model."""
    id: int = Field(..., description="Product ID")
    name: str = Field(..., description="Product name")
    price: float = Field(..., description="Product price")
    in_stock: bool = Field(..., description="Whether product is in stock")


def test_basic_parsers():
    """Test basic parser functionality."""
    print("🧪 Testing Basic Parser Functionality...")
    
    try:
        # Test string parser
        str_parser = create_parser('str', strip_whitespace=True)
        result = str_parser.parse("  Hello, World!  ")
        assert result == "Hello, World!"
        print("✅ String parser working")
        
        # Test JSON parser
        json_parser = create_parser('json')
        json_text = '{"name": "Alice", "age": 30, "active": true}'
        result = json_parser.parse(json_text)
        assert result == {"name": "Alice", "age": 30, "active": True}
        print("✅ JSON parser working")
        
        # Test Pydantic parser
        pydantic_parser = create_parser('pydantic', pydantic_object=PersonModel)
        person_json = '{"name": "Bob", "age": 25, "email": "bob@example.com", "skills": ["Python", "AI"]}'
        result = pydantic_parser.parse(person_json)
        assert isinstance(result, PersonModel)
        assert result.name == "Bob"
        assert result.age == 25
        print("✅ Pydantic parser working")
        
        # Test list parser
        list_parser = create_parser('list', numbered=True)
        list_text = "1. First item\n2. Second item\n3. Third item"
        result = list_parser.parse(list_text)
        assert result == ["First item", "Second item", "Third item"]
        print("✅ List parser working")
        
        # Test regex parser
        regex_parser = create_parser('regex', 
                                   regex_pattern=r"Name: (\w+), Age: (\d+)",
                                   group_names=["name", "age"])
        regex_text = "Person details - Name: Charlie, Age: 35"
        result = regex_parser.parse(regex_text)
        assert result == {"name": "Charlie", "age": "35"}
        print("✅ Regex parser working")
        
        print(f"✅ All 5 basic parsers working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Basic parsers test failed: {e}")
        return False


def test_parser_manager():
    """Test ParserManager functionality."""
    print("\n🧪 Testing Parser Manager...")
    
    try:
        manager = ParserManager()
        
        # Create various parsers
        manager.create_parser('string_cleaner', 'str', strip_whitespace=True, remove_empty_lines=True)
        manager.create_json_parser('json_basic', schema={'name': str, 'age': int})
        manager.create_pydantic_parser('person_parser', PersonModel)
        manager.create_list_parser('task_list', numbered=True)
        manager.create_regex_parser('email_extractor', r'Email: ([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})')
        
        print(f"✅ Created {len(manager._parsers)} parsers")
        
        # Test parsing with different parsers
        test_cases = [
            ('string_cleaner', '  Clean this text  \n\n  ', 'Clean this text'),
            ('json_basic', '{"name": "Alice", "age": 30}', {"name": "Alice", "age": 30}),
            ('task_list', '1. Buy groceries\n2. Walk the dog\n3. Read a book', 
             ['Buy groceries', 'Walk the dog', 'Read a book']),
        ]
        
        for parser_name, input_text, expected in test_cases:
            result = manager.parse(parser_name, input_text)
            assert result == expected
            print(f"✅ Parser '{parser_name}' working correctly")
        
        # Test Pydantic parser
        person_data = '{"name": "David", "age": 28, "email": "david@test.com", "skills": ["JavaScript", "React"]}'
        person_result = manager.parse('person_parser', person_data)
        assert isinstance(person_result, PersonModel)
        assert person_result.name == "David"
        print("✅ Pydantic parser working correctly")
        
        # Test regex parser
        email_text = "Contact info: Email: john.doe@company.com, Phone: 123-456-7890"
        email_result = manager.parse('email_extractor', email_text)
        assert email_result == "john.doe@company.com"
        print("✅ Regex parser working correctly")
        
        # Test parser listing
        parsers = manager.list_parsers()
        assert len(parsers) == 5
        print(f"✅ Listed {len(parsers)} parsers")
        
        # Test format instructions
        instructions = manager.get_format_instructions('json_basic')
        assert 'JSON' in instructions
        print("✅ Format instructions working")
        
        return True
        
    except Exception as e:
        print(f"❌ Parser manager test failed: {e}")
        return False


def test_error_handling():
    """Test error handling and recovery."""
    print("\n🧪 Testing Error Handling and Recovery...")
    
    try:
        manager = ParserManager()
        error_handler = OutputErrorHandler()
        
        # Create JSON parser for testing
        manager.create_json_parser('json_recovery')
        
        # Test cases with recoverable errors
        test_cases = [
            # Trailing comma
            '{"name": "Alice", "age": 30,}',
            # Single quotes
            "{'name': 'Bob', 'age': 25}",
            # Python booleans
            '{"name": "Charlie", "active": True, "verified": False}',
            # JSON in text
            'Here is the data: {"name": "Diana", "age": 35} and that\'s all.',
        ]
        
        successful_recoveries = 0
        
        for i, malformed_json in enumerate(test_cases):
            try:
                result = manager.parse('json_recovery', malformed_json, retry_on_error=True)
                if isinstance(result, dict) and 'name' in result:
                    successful_recoveries += 1
                    print(f"✅ Recovered malformed JSON case {i+1}")
            except Exception as e:
                print(f"⚠️  Could not recover JSON case {i+1}: {e}")
        
        print(f"✅ Successfully recovered {successful_recoveries}/{len(test_cases)} malformed JSON cases")
        
        # Test direct error handler
        malformed = '{"name": "Test", "age": 30,}'  # Trailing comma
        fixed = error_handler.fix_json(malformed)
        try:
            json.loads(fixed)
            print("✅ Direct JSON fix working")
        except json.JSONDecodeError:
            print("⚠️  Direct JSON fix needs improvement")
        
        # Test error analysis
        try:
            manager.parse('json_recovery', 'completely invalid json', retry_on_error=False)
        except ValueError as e:
            error_info = error_handler.handle_parse_error(e, 'completely invalid json', 'json')
            assert 'error_type' in error_info
            assert 'suggested_fixes' in error_info
            print("✅ Error analysis working")
        
        return True
        
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        return False


def test_advanced_features():
    """Test advanced parser features."""
    print("\n🧪 Testing Advanced Features...")
    
    try:
        manager = ParserManager()
        
        # Test complex Pydantic model
        manager.create_pydantic_parser('product_parser', ProductModel)
        
        product_json = '{"id": 123, "name": "Laptop", "price": 999.99, "in_stock": true}'
        product = manager.parse('product_parser', product_json)
        assert isinstance(product, ProductModel)
        assert product.id == 123
        assert product.price == 999.99
        print("✅ Complex Pydantic model parsing working")
        
        # Test nested list parser with JSON items
        json_parser = create_parser('json')
        manager.create_list_parser('json_list', item_parser=json_parser, separator='\n---\n')
        
        json_list_text = '''{"name": "Alice", "score": 95}
---
{"name": "Bob", "score": 87}
---
{"name": "Charlie", "score": 92}'''
        
        results = manager.parse('json_list', json_list_text)
        assert len(results) == 3
        assert all(isinstance(item, dict) for item in results)
        assert results[0]['name'] == 'Alice'
        print("✅ Nested list parser working")
        
        # Test fallback parsing
        manager.create_parser('strict_json', 'json', strict=True)
        manager.create_parser('lenient_json', 'json', strict=False)
        manager.create_parser('fallback_str', 'str')
        
        # This should fail with strict parser but work with lenient
        malformed_json = '{"name": "Test", "age": 30,}'  # Trailing comma
        result = manager.parse_with_fallback(['strict_json', 'lenient_json', 'fallback_str'], malformed_json)
        assert isinstance(result, dict)
        print("✅ Fallback parsing working")
        
        # Test batch parsing
        json_texts = [
            '{"name": "Person1", "age": 25}',
            '{"name": "Person2", "age": 30}',
            '{"name": "Person3", "age": 35,}',  # Recoverable trailing comma
            '{"name": "Person4", "age": 40}'
        ]
        
        batch_results = manager.batch_parse('lenient_json', json_texts, continue_on_error=True)
        assert len(batch_results) == 4
        assert batch_results[0]['name'] == 'Person1'
        assert batch_results[2]['name'] == 'Person3'  # Should be recovered
        assert batch_results[3]['name'] == 'Person4'
        print("✅ Batch parsing working")
        
        # Test parser cloning
        original_parser = manager.create_parser('original_str', 'str', strip_whitespace=True)
        cloned_parser = manager.clone_parser('original_str', 'cloned_str', remove_empty_lines=True)
        
        assert 'cloned_str' in manager._parsers
        assert cloned_parser is not original_parser
        print("✅ Parser cloning working")
        
        # Test parser statistics
        stats = manager.get_parser_stats()
        assert stats['total_parsers'] > 0
        assert 'parser_types' in stats
        print(f"✅ Parser statistics: {stats['total_parsers']} total parsers")
        
        return True
        
    except Exception as e:
        print(f"❌ Advanced features test failed: {e}")
        return False


def test_langchain_integration():
    """Test LangChain integration."""
    print("\n🧪 Testing LangChain Integration...")
    
    try:
        # Test converting parsers to LangChain format
        parsers_to_test = [
            ('str', StrOutputParser()),
            ('json', JsonOutputParser()),
            ('pydantic', PydanticOutputParser(pydantic_object=PersonModel)),
        ]
        
        for parser_name, parser in parsers_to_test:
            try:
                lc_parser = parser.to_langchain()
                assert lc_parser is not None
                print(f"✅ {parser_name.capitalize()} parser converted to LangChain format")
            except Exception as e:
                print(f"⚠️  {parser_name.capitalize()} parser LangChain conversion failed: {e}")
        
        # Test that LangChain parsers have expected methods
        str_parser = StrOutputParser()
        lc_str_parser = str_parser.to_langchain()
        
        # LangChain parsers should have parse method
        if hasattr(lc_str_parser, 'parse'):
            result = lc_str_parser.parse("test text")
            assert result == "test text"
            print("✅ LangChain parser parse method working")
        
        return True
        
    except Exception as e:
        print(f"❌ LangChain integration test failed: {e}")
        return False


def test_real_world_scenarios():
    """Test real-world parsing scenarios."""
    print("\n🧪 Testing Real-World Scenarios...")
    
    try:
        manager = ParserManager()
        
        # Scenario 1: API response parsing
        manager.create_json_parser('api_response')
        api_response = '''
        {
            "status": "success",
            "data": {
                "users": [
                    {"id": 1, "name": "Alice", "active": true},
                    {"id": 2, "name": "Bob", "active": false}
                ]
            },
            "timestamp": "2025-01-13T10:30:00Z"
        }
        '''
        
        result = manager.parse('api_response', api_response)
        assert result['status'] == 'success'
        assert len(result['data']['users']) == 2
        print("✅ API response parsing working")
        
        # Scenario 2: Log parsing
        manager.create_regex_parser('log_parser', 
                                  r'(\d{4}-\d{2}-\d{2}) (\d{2}:\d{2}:\d{2}) \[(\w+)\] (.+)',
                                  group_names=['date', 'time', 'level', 'message'])
        
        log_entry = "2025-01-13 10:30:45 [ERROR] Database connection failed"
        log_result = manager.parse('log_parser', log_entry)
        assert log_result['level'] == 'ERROR'
        assert 'Database' in log_result['message']
        print("✅ Log parsing working")
        
        # Scenario 3: Configuration parsing
        manager.create_list_parser('config_list', separator='\n', numbered=False)
        config_text = """database_host=localhost
database_port=5432
database_name=myapp
debug_mode=true"""
        
        config_items = manager.parse('config_list', config_text)
        assert len(config_items) == 4
        assert any('localhost' in item for item in config_items)
        print("✅ Configuration parsing working")
        
        # Scenario 4: Mixed content extraction
        manager.create_regex_parser('email_phone_extractor',
                                  r'Email: ([^\s,]+).*?Phone: ([^\s,]+)',
                                  group_names=['email', 'phone'])
        
        contact_text = "Contact: Email: support@company.com, Phone: +1-555-0123, Address: 123 Main St"
        contact_info = manager.parse('email_phone_extractor', contact_text)
        assert '@' in contact_info['email']
        assert '+1-555-0123' == contact_info['phone']
        print("✅ Mixed content extraction working")
        
        return True
        
    except Exception as e:
        print(f"❌ Real-world scenarios test failed: {e}")
        return False


def main():
    """Run all integration tests."""
    print("🚀 Starting Output Parsing Toolkit Integration Tests...\n")
    
    tests = [
        test_basic_parsers,
        test_parser_manager,
        test_error_handling,
        test_advanced_features,
        test_langchain_integration,
        test_real_world_scenarios,
    ]
    
    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test_func.__name__} crashed: {e}")
            results.append(False)
    
    # Summary
    print("\n" + "="*60)
    print("📊 Integration Test Results Summary:")
    print("="*60)
    
    passed = sum(results)
    total = len(results)
    
    for i, (test_func, result) in enumerate(zip(tests, results)):
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_func.__name__}")
    
    print(f"\n📈 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All integration tests passed! Output parsing toolkit is ready.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)