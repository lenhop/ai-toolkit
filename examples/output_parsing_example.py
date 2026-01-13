#!/usr/bin/env python3
"""
Example usage of the AI Toolkit Output Parsing system.
"""

import json
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List

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


# Example Pydantic models for demonstration
class PersonModel(BaseModel):
    """Example person model."""
    name: str = Field(..., description="Person's full name")
    age: int = Field(..., description="Person's age")
    email: str = Field(..., description="Person's email address")
    skills: List[str] = Field(default_factory=list, description="List of skills")


class ProductModel(BaseModel):
    """Example product model."""
    id: int = Field(..., description="Product ID")
    name: str = Field(..., description="Product name")
    price: float = Field(..., description="Product price")
    category: str = Field(..., description="Product category")
    in_stock: bool = Field(..., description="Whether product is in stock")


def basic_parser_examples():
    """Demonstrate basic parser functionality."""
    print("🚀 Basic Parser Examples")
    print("=" * 50)
    
    # 1. String Parser
    print("1. String Parser:")
    str_parser = create_parser('str', strip_whitespace=True, remove_empty_lines=True)
    
    messy_text = """  
    Hello, World!  
    
    This is a test.
    
    
    End of text.  
    """
    
    cleaned = str_parser.parse(messy_text)
    print(f"   Input: {repr(messy_text[:30])}...")
    print(f"   Output: {repr(cleaned)}")
    print(f"   Instructions: {str_parser.get_format_instructions()}")
    
    # 2. JSON Parser
    print("\n2. JSON Parser:")
    json_parser = create_parser('json')
    
    json_examples = [
        '{"name": "Alice", "age": 30, "active": true}',
        'Here is the data: {"name": "Bob", "age": 25} and that\'s all.',
        '{"name": "Charlie", "age": 35,}',  # Trailing comma - will be fixed
    ]
    
    for i, json_text in enumerate(json_examples):
        try:
            result = json_parser.parse(json_text)
            print(f"   Example {i+1}: {json_text[:40]}...")
            print(f"   Parsed: {result}")
        except Exception as e:
            print(f"   Example {i+1} failed: {e}")
    
    # 3. Pydantic Parser
    print("\n3. Pydantic Parser:")
    pydantic_parser = create_parser('pydantic', pydantic_object=PersonModel)
    
    person_json = '{"name": "David Wilson", "age": 28, "email": "david@example.com", "skills": ["Python", "AI", "Machine Learning"]}'
    person = pydantic_parser.parse(person_json)
    
    print(f"   Input: {person_json[:50]}...")
    print(f"   Parsed: {person}")
    print(f"   Type: {type(person)}")
    print(f"   Name: {person.name}, Age: {person.age}")
    
    # 4. List Parser
    print("\n4. List Parser:")
    list_parser = create_parser('list', numbered=True, separator='\n')
    
    list_text = """1. Buy groceries
2. Walk the dog
3. Read a book
4. Call mom
5. Finish project"""
    
    items = list_parser.parse(list_text)
    print(f"   Input: {list_text.replace(chr(10), ' | ')}")
    print(f"   Parsed: {items}")
    
    # 5. Regex Parser
    print("\n5. Regex Parser:")
    regex_parser = create_parser('regex', 
                                regex_pattern=r'Name: (\w+\s+\w+), Age: (\d+), Email: ([^\s,]+)',
                                group_names=['name', 'age', 'email'])
    
    contact_text = "Contact info - Name: John Doe, Age: 32, Email: john.doe@company.com"
    contact_info = regex_parser.parse(contact_text)
    
    print(f"   Input: {contact_text}")
    print(f"   Parsed: {contact_info}")
    
    print()


def parser_manager_examples():
    """Demonstrate ParserManager functionality."""
    print("⚙️  Parser Manager Examples")
    print("=" * 50)
    
    manager = ParserManager()
    
    # Create various parsers
    print("Creating parsers...")
    
    # Basic parsers
    manager.create_parser('text_cleaner', 'str', strip_whitespace=True, remove_empty_lines=True)
    manager.create_json_parser('api_response', schema={'status': str, 'data': dict})
    manager.create_pydantic_parser('person_data', PersonModel)
    manager.create_pydantic_parser('product_data', ProductModel)
    manager.create_list_parser('todo_list', numbered=True)
    manager.create_regex_parser('log_parser', 
                               r'(\d{4}-\d{2}-\d{2}) (\d{2}:\d{2}:\d{2}) \[(\w+)\] (.+)',
                               group_names=['date', 'time', 'level', 'message'])
    
    print(f"✅ Created {len(manager._parsers)} parsers")
    
    # List all parsers
    print("\nAvailable parsers:")
    parsers = manager.list_parsers()
    for parser_info in parsers:
        print(f"   - {parser_info['name']} ({parser_info['type']})")
    
    # Test parsing with different parsers
    print("\nTesting parsers:")
    
    # Test API response parser
    api_response = '{"status": "success", "data": {"users": [{"id": 1, "name": "Alice"}]}}'
    result = manager.parse('api_response', api_response)
    print(f"✅ API Response: {result['status']} with {len(result['data']['users'])} users")
    
    # Test person parser
    person_data = '{"name": "Emma Thompson", "age": 34, "email": "emma@test.com", "skills": ["JavaScript", "React", "Node.js"]}'
    person = manager.parse('person_data', person_data)
    print(f"✅ Person: {person.name} ({person.age} years old) - {len(person.skills)} skills")
    
    # Test log parser
    log_entry = "2025-01-13 14:30:45 [ERROR] Database connection timeout"
    log_info = manager.parse('log_parser', log_entry)
    print(f"✅ Log: {log_info['date']} {log_info['time']} [{log_info['level']}] {log_info['message']}")
    
    # Test fallback parsing
    print("\nTesting fallback parsing:")
    malformed_json = '{"name": "Test", "age": 30,}'  # Trailing comma
    result = manager.parse_with_fallback(['api_response', 'text_cleaner'], malformed_json)
    print(f"✅ Fallback result: {result}")
    
    print()


def error_handling_examples():
    """Demonstrate error handling and recovery."""
    print("🛡️  Error Handling Examples")
    print("=" * 50)
    
    manager = ParserManager()
    error_handler = OutputErrorHandler()
    
    # Create parsers for testing
    manager.create_json_parser('json_recovery')
    manager.create_pydantic_parser('strict_person', PersonModel)
    
    print("Testing JSON error recovery:")
    
    # Test cases with recoverable errors
    malformed_examples = [
        ('Trailing comma', '{"name": "Alice", "age": 30,}'),
        ('Single quotes', "{'name': 'Bob', 'age': 25}"),
        ('Python booleans', '{"name": "Charlie", "active": True, "verified": False}'),
        ('Unquoted keys', '{name: "Diana", age: 35}'),
        ('JSON in text', 'The result is: {"name": "Eve", "age": 28} - that\'s all.'),
    ]
    
    for description, malformed_json in malformed_examples:
        try:
            result = manager.parse('json_recovery', malformed_json, retry_on_error=True)
            print(f"   ✅ {description}: Successfully recovered")
            print(f"      Result: {result}")
        except Exception as e:
            print(f"   ❌ {description}: Could not recover - {e}")
    
    print("\nTesting error analysis:")
    
    # Test error analysis
    try:
        manager.parse('strict_person', '{"name": "Test"}', retry_on_error=False)  # Missing required fields
    except Exception as e:
        error_info = error_handler.handle_parse_error(e, '{"name": "Test"}', 'pydantic')
        print(f"   Error type: {error_info['error_type']}")
        print(f"   Error message: {error_info['error_message']}")
        print(f"   Suggested fixes: {error_info['suggested_fixes']}")
    
    # Test direct JSON fixing
    print("\nTesting direct JSON fixing:")
    broken_json = '{"items": ["item1", "item2",], "count": 2,}'
    fixed_json = error_handler.fix_json(broken_json)
    print(f"   Original: {broken_json}")
    print(f"   Fixed: {fixed_json}")
    
    try:
        json.loads(fixed_json)
        print("   ✅ Fixed JSON is valid")
    except json.JSONDecodeError:
        print("   ❌ Fixed JSON is still invalid")
    
    print()


def advanced_features_examples():
    """Demonstrate advanced features."""
    print("🎯 Advanced Features Examples")
    print("=" * 50)
    
    manager = ParserManager()
    
    # Complex nested parsing
    print("1. Complex Nested Parsing:")
    
    # Create a list parser that parses JSON items
    json_parser = create_parser('json')
    manager.create_list_parser('json_list', item_parser=json_parser, separator='\n---\n')
    
    json_list_text = '''{"name": "Alice", "score": 95, "grade": "A"}
---
{"name": "Bob", "score": 87, "grade": "B"}
---
{"name": "Charlie", "score": 92, "grade": "A"}'''
    
    results = manager.parse('json_list', json_list_text)
    print(f"   Parsed {len(results)} JSON objects from list:")
    for i, item in enumerate(results):
        print(f"      {i+1}. {item['name']}: {item['score']} ({item['grade']})")
    
    # Batch parsing
    print("\n2. Batch Parsing:")
    
    manager.create_json_parser('batch_json')
    
    json_texts = [
        '{"product": "Laptop", "price": 999.99}',
        '{"product": "Mouse", "price": 29.99}',
        '{"product": "Keyboard", "price": 79.99,}',  # Recoverable error
        '{"product": "Monitor", "price": 299.99}'
    ]
    
    batch_results = manager.batch_parse('batch_json', json_texts, continue_on_error=True)
    print(f"   Batch parsed {len(batch_results)} items:")
    for i, result in enumerate(batch_results):
        if result:
            print(f"      {i+1}. {result['product']}: ${result['price']}")
        else:
            print(f"      {i+1}. Failed to parse")
    
    # Parser testing
    print("\n3. Parser Testing:")
    
    manager.create_regex_parser('email_extractor', r'([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})')
    
    test_cases = [
        "Contact: john.doe@company.com",
        "Email me at alice@example.org",
        "No email here",
        "Multiple emails: bob@test.com and charlie@demo.net"
    ]
    
    for test_text in test_cases:
        result = manager.test_parser('email_extractor', test_text)
        status = "✅" if result['success'] else "❌"
        print(f"   {status} '{test_text[:30]}...' -> {result.get('output', result.get('error'))}")
    
    # Parser cloning
    print("\n4. Parser Cloning:")
    
    original = manager.create_parser('original_str', 'str', strip_whitespace=True)
    cloned = manager.clone_parser('original_str', 'enhanced_str', remove_empty_lines=True)
    
    test_text = "  Line 1  \n\n  Line 2  \n\n"
    original_result = manager.parse('original_str', test_text)
    cloned_result = manager.parse('enhanced_str', test_text)
    
    print(f"   Original result: {repr(original_result)}")
    print(f"   Enhanced result: {repr(cloned_result)}")
    
    # Statistics
    print("\n5. Parser Statistics:")
    stats = manager.get_parser_stats()
    print(f"   Total parsers: {stats['total_parsers']}")
    print(f"   Parser types: {stats['parser_types']}")
    
    print()


def real_world_scenarios():
    """Demonstrate real-world parsing scenarios."""
    print("🌍 Real-World Scenarios")
    print("=" * 50)
    
    manager = ParserManager()
    
    # Scenario 1: API Response Processing
    print("1. API Response Processing:")
    
    manager.create_json_parser('api_processor')
    
    api_responses = [
        '{"status": "success", "data": {"users": 150, "active": 120}, "timestamp": "2025-01-13T10:30:00Z"}',
        '{"status": "error", "error": {"code": 404, "message": "Not found"}, "timestamp": "2025-01-13T10:31:00Z"}',
        '{"status": "success", "data": {"orders": [{"id": 1, "total": 99.99}, {"id": 2, "total": 149.99}]}, "timestamp": "2025-01-13T10:32:00Z"}'
    ]
    
    for i, response in enumerate(api_responses):
        result = manager.parse('api_processor', response)
        print(f"   Response {i+1}: {result['status']} at {result['timestamp']}")
        if result['status'] == 'success':
            if 'users' in result['data']:
                print(f"      Users: {result['data']['users']} total, {result['data']['active']} active")
            elif 'orders' in result['data']:
                total_value = sum(order['total'] for order in result['data']['orders'])
                print(f"      Orders: {len(result['data']['orders'])} orders, total value: ${total_value}")
        else:
            print(f"      Error {result['error']['code']}: {result['error']['message']}")
    
    # Scenario 2: Log File Processing
    print("\n2. Log File Processing:")
    
    manager.create_regex_parser('access_log', 
                               r'(\d+\.\d+\.\d+\.\d+) - - \[([^\]]+)\] "(\w+) ([^"]+)" (\d+) (\d+)',
                               group_names=['ip', 'timestamp', 'method', 'path', 'status', 'size'])
    
    log_entries = [
        '192.168.1.100 - - [13/Jan/2025:10:30:45 +0000] "GET /api/users" 200 1234',
        '192.168.1.101 - - [13/Jan/2025:10:31:12 +0000] "POST /api/login" 401 567',
        '192.168.1.102 - - [13/Jan/2025:10:31:45 +0000] "GET /api/products" 200 2345'
    ]
    
    for entry in log_entries:
        try:
            log_data = manager.parse('access_log', entry)
            status_emoji = "✅" if log_data['status'].startswith('2') else "❌"
            print(f"   {status_emoji} {log_data['ip']} {log_data['method']} {log_data['path']} -> {log_data['status']} ({log_data['size']} bytes)")
        except Exception as e:
            print(f"   ❌ Failed to parse log entry: {e}")
    
    # Scenario 3: Configuration File Processing
    print("\n3. Configuration File Processing:")
    
    manager.create_list_parser('config_parser', separator='\n')
    
    config_text = """# Database Configuration
database.host=localhost
database.port=5432
database.name=myapp
database.user=admin

# Cache Configuration  
cache.enabled=true
cache.ttl=3600
cache.max_size=1000

# API Configuration
api.rate_limit=100
api.timeout=30"""
    
    config_lines = manager.parse('config_parser', config_text)
    config_dict = {}
    
    for line in config_lines:
        if '=' in line and not line.startswith('#'):
            key, value = line.split('=', 1)
            config_dict[key.strip()] = value.strip()
    
    print(f"   Parsed {len(config_dict)} configuration items:")
    for key, value in sorted(config_dict.items()):
        print(f"      {key} = {value}")
    
    # Scenario 4: Data Extraction from Mixed Content
    print("\n4. Data Extraction from Mixed Content:")
    
    manager.create_regex_parser('contact_extractor',
                               r'Name:\s*([^,\n]+),?\s*Email:\s*([^\s,\n]+),?\s*Phone:\s*([^\s,\n]+)',
                               group_names=['name', 'email', 'phone'])
    
    mixed_content = """
    Here are the contact details:
    
    Name: John Smith, Email: john.smith@company.com, Phone: +1-555-0123
    
    Please reach out if you have any questions.
    """
    
    try:
        contact = manager.parse('contact_extractor', mixed_content)
        print(f"   Extracted contact:")
        print(f"      Name: {contact['name']}")
        print(f"      Email: {contact['email']}")
        print(f"      Phone: {contact['phone']}")
    except Exception as e:
        print(f"   Failed to extract contact: {e}")
    
    print()


def langchain_integration_examples():
    """Demonstrate LangChain integration."""
    print("🔗 LangChain Integration Examples")
    print("=" * 50)
    
    # Create parsers and convert to LangChain format
    parsers_to_test = [
        ('String Parser', StrOutputParser()),
        ('JSON Parser', JsonOutputParser()),
        ('Pydantic Parser', PydanticOutputParser(pydantic_object=PersonModel)),
        ('List Parser', ListOutputParser(numbered=True)),
        ('Regex Parser', RegexOutputParser(regex_pattern=r'Result: (\w+)', group_names=['result']))
    ]
    
    print("Converting parsers to LangChain format:")
    
    for name, parser in parsers_to_test:
        try:
            lc_parser = parser.to_langchain()
            print(f"   ✅ {name}: {type(lc_parser).__name__}")
            
            # Test basic functionality
            if hasattr(lc_parser, 'get_format_instructions'):
                instructions = lc_parser.get_format_instructions()
                print(f"      Instructions: {instructions[:60]}...")
            
        except Exception as e:
            print(f"   ❌ {name}: Failed to convert - {e}")
    
    # Test LangChain parser usage
    print("\nTesting LangChain parser usage:")
    
    str_parser = StrOutputParser()
    lc_str_parser = str_parser.to_langchain()
    
    if hasattr(lc_str_parser, 'parse'):
        result = lc_str_parser.parse("  Hello from LangChain!  ")
        print(f"   LangChain string parser result: '{result}'")
    
    json_parser = JsonOutputParser()
    lc_json_parser = json_parser.to_langchain()
    
    if hasattr(lc_json_parser, 'parse'):
        try:
            result = lc_json_parser.parse('{"message": "Hello from LangChain!", "success": true}')
            print(f"   LangChain JSON parser result: {result}")
        except Exception as e:
            print(f"   LangChain JSON parser error: {e}")
    
    print()


def main():
    """Run all examples."""
    print("🎓 AI Toolkit Output Parsing Examples")
    print("=" * 60)
    print()
    
    examples = [
        basic_parser_examples,
        parser_manager_examples,
        error_handling_examples,
        advanced_features_examples,
        real_world_scenarios,
        langchain_integration_examples,
    ]
    
    for example in examples:
        try:
            example()
        except Exception as e:
            print(f"❌ Example {example.__name__} failed: {e}")
            print()
    
    print("🎉 Output Parsing Examples Complete!")
    print()
    print("💡 Key Features Demonstrated:")
    print("   ✅ 5 parser types (String, JSON, Pydantic, List, Regex)")
    print("   ✅ Comprehensive error handling and recovery")
    print("   ✅ Parser management and organization")
    print("   ✅ Batch processing and fallback parsing")
    print("   ✅ LangChain integration")
    print("   ✅ Real-world parsing scenarios")
    print("   ✅ Advanced features (cloning, testing, statistics)")


if __name__ == "__main__":
    main()