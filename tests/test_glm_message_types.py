"""
Test GLM provider message type handling.

This test verifies that the GLM provider correctly imports and uses
message types (AIMessage, SystemMessage, HumanMessage).
"""

import pytest
from ai_toolkit.models.model_providers import GLMChatModel


def test_glm_imports():
    """Test that GLM provider has all required imports."""
    # This test will fail if imports are missing
    from ai_toolkit.models.model_providers import AIMessage, SystemMessage, HumanMessage
    assert AIMessage is not None
    assert SystemMessage is not None
    assert HumanMessage is not None
    print("✅ All message type imports are available in GLM provider")


def test_glm_message_classes_accessible():
    """Test that message classes can be used with isinstance checks."""
    from ai_toolkit.models.model_providers import AIMessage, SystemMessage, HumanMessage
    from langchain_core.messages import HumanMessage as LCHumanMessage
    
    # Create a message
    msg = LCHumanMessage(content="test")
    
    # Verify isinstance works (this is what GLM provider uses)
    assert isinstance(msg, HumanMessage)
    print("✅ isinstance checks work correctly with imported message types")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
