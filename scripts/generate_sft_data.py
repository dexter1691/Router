
import json
import os
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Tuple, Optional
import random
import argparse
import multiprocessing as mp
import uuid

load_dotenv()

client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
)

TOOLS_JSON_PATH = "data/tools.json"

with open(TOOLS_JSON_PATH, "r") as f:
    tools = json.load(f)

tools_flat_list = []
tools_map = []
tools_to_integration_map = {}
for category, tools in tools.items():
    for tool in tools:
        tools_flat_list.append(f'{tool["name"]} - {tool["description"]}')
        tools_map.append(tool["name"])
        tools_to_integration_map[tool["name"]] = category

print('Length of tools_flat_list', len(tools_flat_list))

with open("data/tool_descriptions.json", "r") as f:
    integration_descriptions = json.load(f)

print('Length of integration_descriptions', len(integration_descriptions))

class ResponseFormat(BaseModel):
    index: int
    query: str
    tools_required: List[str]
    
class ResposneFormatList(BaseModel):
    queries: List[ResponseFormat]
    
def generate_diverse_queries_for_tools(selected_tools_list: List[str], temperature: float = 0.9) -> Optional[List[ResponseFormat]]:
    """
    Generate diverse queries for a random selection of k tools using GPT-4.
    
    Args:
        tools_flat_list: List of tool descriptions
        temperature: Temperature for GPT-4 (higher = more diverse)
    
    Returns:
        Tuple of (list of 10 diverse queries, list of selected tools)
    """
    # Create the prompt
    tools_text = "\n".join([f"- {tool}" for tool in selected_tools_list])
    
    prompt = f"""## Instruction: 
You are a helpful assistant that generates diverse user queries for tool usage scenarios.

## Input:
Given the following {len(selected_tools_list)} set of tools:
{tools_text}

## Output:
Generate exactly 50 diverse and realistic user queries that someone might ask to accomplish tasks using all these tools.

## Guidelines:
1. Make the queries varied in complexity, style, and use case. Include both simple and complex scenarios.
2. Be creative and diverse in your responses. 
3. You should make sure to use **all** the tools in the input.

# Output Format:
The response should be in the following format:
[
    {{
        "index": 0,
        "query": "query",
        "tools_required": ["tool1", "tool2", "tool3"]
    }},
    {{
        "index": 1,
        "query": "query", 
        "tools_required": ["tool1", "tool2", "tool3"]
    }},
    ...
    {{
        "index": 50,
        "query": "query",
        "tools_required": ["tool1", "tool2", "tool3"]
    }}
]

Return your answer in this format and no other text:"""

    try:
        response = client.responses.parse(
            model="gpt-4.1",
            input=[
                {"role": "system", "content": "You are a helpful assistant that generates diverse user queries for tool usage scenarios."},
                {"role": "user", "content": prompt}
            ],
            text_format=ResposneFormatList,
            temperature=temperature
        )
        return response.output_parsed.queries
        
    except Exception as e:
        print(f"Error generating queries: {e}")
        return 

def generate_queries(selected_tools: List[str], temperature: float = 0.5):
    response = generate_diverse_queries_for_tools(selected_tools, temperature=0.5)
    assert response is not None, "Response is None"

    uuid_id = str(uuid.uuid4())
    if not os.path.exists(f"data/sft_data"):
        os.makedirs(f"data/sft_data")

    with open(f"data/sft_data/{uuid_id}.json", "w") as f:
        responses = []
        for query in response:
            responses.append(query.model_dump())
        json.dump(responses, f)
    
    print(f"Generated {len(response)} queries for {selected_tools} and saved to {uuid_id}.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no_of_tools", type=int, default=5)
    parser.add_argument("--no_of_queries", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--no_of_processes", type=int, default=2)

    args = parser.parse_args()
    no_of_tools = args.no_of_tools
    no_of_queries = args.no_of_queries
    
    list_of_selected_tools = []
    for k in range(1, no_of_tools + 1):
        # Randomly select k tools
        for i in range(no_of_queries):
            selected_tools = random.sample(tools_flat_list, min(k, len(tools_flat_list)))
            list_of_selected_tools.append(selected_tools)

    pool = mp.Pool(processes=args.no_of_processes)
    pool.map(generate_queries, list_of_selected_tools)
    pool.close()
    pool.join()
