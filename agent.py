from pydantic_ai import Agent
from pydantic_ai.models.groq import GroqModel
from pydantic_ai.providers.groq import GroqProvider
from pydantic_ai.mcp import MCPServerStdio
from dotenv import load_dotenv
import os
import logfire   

load_dotenv()
groq_key = os.getenv("groq_key")
logfire_key = os.getenv("logfire_key")   

logfire.configure(
    send_to_logfire="if-token-present",
    token=logfire_key
)

server = MCPServerStdio(
    'python',           
    args=['mcp-server.py']
)

prompt = """
/no_think
Math Agent using MCP tools. RULES:
1. NEVER calculate mentally - always use MCP math tools
2. ONLY math questions - else respond: "This question is outside my math context."
3. Convert string numbers to numeric types before calling tools
4. Show step-by-step results in plain text only, no $ signs or LaTeX formatting
"""

model = GroqModel(
    'qwen/qwen3-32b',
    provider=GroqProvider(api_key=groq_key)
)

agent = Agent(
    system_prompt=prompt,
    model=model,
    toolsets=[server]
)

print("Math Agent ready! Type 'quit' to exit.\n")

while True:
    user_input = input("Enter a math question: ")
    if user_input.lower() in ['quit', 'exit']:
        break
    try:
        output = agent.run_sync(user_input)
        result_text = output.output.replace("$", "").strip()
        print(f"Result: {result_text}\n")

        logfire.info("Math query succeeded", query=user_input, result=result_text)

    except Exception as e:
        print(f"Error: {e}\n")

        logfire.error("Math query failed", query=user_input, error=str(e))
