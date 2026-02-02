



"""
Quick Start Example - Simplified Model Usage

Shows the easiest way to get started with ai_toolkit.
"""

import sys
import os
import time
from dotenv import load_dotenv, dotenv_values
from ai_toolkit.models import ModelManager

load_dotenv()
# Create model manager
manager = ModelManager()

# Create a model (auto-loads API key from environment - recommended)
model = manager.create_model(
    api_key=os.environ.get('DEEPSEEK_API_KEY'),
    provider="deepseek",           # Provider: deepseek, qwen, or glm
    model="deepseek-chat",         # Model variant (optional)
    temperature=0.7,               # Randomness: 0.0-2.0
    max_tokens=2000                # Max response length
)


# 1.bind_tools
from langchain.tools import tool

@tool
def get_weather(location: str) -> str:
    """Get the weather at a location."""
    return f"It's sunny in {location}."


model_with_tools = model.bind_tools([get_weather])  

response = model_with_tools.invoke("What's the weather like in Boston?")
for tool_call in response.tool_calls:
    # View tool calls made by the model
    print(f"Tool: {tool_call['name']}")
    print(f"Args: {tool_call['args']}")


# 2.construct output
# DeepSeek doesn't support the response_format parameter used by with_structured_output(). 
from pydantic import BaseModel, Field

class Movie(BaseModel):
    """A movie with details."""
    title: str = Field(..., description="The title of the movie")
    year: int = Field(..., description="The year the movie was released")
    director: str = Field(..., description="The director of the movie")
    rating: float = Field(..., description="The movie's rating out of 10")

model_with_structure = model.with_structured_output(Movie)
response = model_with_structure.invoke("Provide details about the movie Inception")
print(response)  # Movie(title="Inception", year=2010, director="Christopher Nolan", rating=8.8)


# 2.2 Use PydanticOutputParser with Prompt Engineering (Recommended), for DeepSeek
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage

class Movie(BaseModel):
    """A movie with details."""
    title: str = Field(..., description="The title of the movie")
    year: int = Field(..., description="The year the movie was released")
    director: str = Field(..., description="The director of the movie")
    rating: float = Field(..., description="The movie's rating out of 10")

# Create parser
parser = PydanticOutputParser(pydantic_object=Movie)

# Create prompt with format instructions
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Output valid JSON only.\n{format_instructions}"),
    ("human", "{query}")
])

# Create chain
chain = prompt | model | parser

# Invoke
response = chain.invoke({
    "format_instructions": parser.get_format_instructions(),
    "query": "Provide details about the movie Inception"
})

print(response)  # Movie(title="Inception", year=2010, director="Christopher Nolan", rating=8.8)


# 3.LangChain chat models can expose a dictionary of supported features and capabilities through a .profile
model.profile
# {
#   "max_input_tokens": 400000,
#   "image_inputs": True,
#   "reasoning_output": True,
#   "tool_calling": True,
#   ...
# }
