
import os
import openai
import dspy
from dotenv import load_dotenv
import json
from pydantic import BaseModel

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
lm = dspy.LM('openai/gpt-4.1', api_key=openai.api_key)
dspy.configure(lm=lm)
print(openai.api_key)

with open("../data/tools.json", "r") as f:
    tools = json.load(f)

tools_flat_list = []
for category, tool_list in tools.items():
    for tool in tool_list:
        tools_flat_list.append(tool["description"])

print(tools_flat_list)

from pydantic import BaseModel

class Tools(BaseModel):
    app_name: str
    tools: list[str]
    tool_descriptions: list[str]

class Integration(dspy.Signature):
    tools: Tools = dspy.InputField(description='The tools that the integration provides along with their descriptions')
    integration_description: str = dspy.OutputField(description='The description of the integration and the functionality that is supported by the integration.')

generate_integration = dspy.ChainOfThought(Integration)    

tools_pydantic_list: list[Tools] = []

for category, tool_dict in tools.items():
    tool_list = []
    tool_description_list = []
    for tool in tool_dict:
        tool_list.append(tool["name"])
        tool_description_list.append(tool["description"])
    tools_pydantic_list.append(Tools(app_name=category, tools=tool_list, tool_descriptions=tool_description_list))

print(tools_pydantic_list)

integration_descriptions = {}
for tool in tools_pydantic_list:
    result = generate_integration(tools=tool)
    print(result.integration_description)
    print(tool.model_dump())
    integration_descriptions[tool.app_name] = result.integration_description

print(integration_descriptions)

with open("../data/tool_descriptions.json", "w") as f:
    json.dump(integration_descriptions, f)
