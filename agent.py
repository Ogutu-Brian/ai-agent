from langchain_community.tools.tavily_search import TavilySearchResults
import getpass
import os
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

search = TavilySearchResults(max_results=2)
search_results = search.run("What is the weather in Nairobi Kenya?")
print(search_results)

tools = [search]

if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI API key: ")
    
model = init_chat_model(
  model="gpt-4.1",
  model_provider="openai"
)
response = model.invoke(
  input=[
    HumanMessage(
    content="hi!"
    )
  ]
)
print(response.content)

model_with_tools = model.bind_tools(tools)
response = model_with_tools.invoke(
  input=[
    HumanMessage(
    content="Hi!"
    )
  ]
)
print(f"ContentString: {response.content}")
print(f"ToolCalls: {response.tool_calls}")

agent_executor = create_react_agent(
  model=model,
  tools=tools
)
response = agent_executor.invoke(
  input={
    "messages": [
      HumanMessage(
        content="hi!"
      )
    ],
  }
)
print(response["messages"])

response = agent_executor.invoke(
  input={
    "messages": [
      HumanMessage(
        content="whats the weather in kenya?"
      )
    ],
  }
)
print(response["messages"])

for step in agent_executor.stream(
  input={
    "messages": [
      HumanMessage(
        content="whats the weather in kenya?"
      )
    ],
  }
):
  step["messages"][-1].pretty_print()

for step, metadata in agent_executor.stream(
  input={
    "messages": [
      HumanMessage(
        content="whats the weather in kenya?"
      )
    ],
  },
  return_only_outputs=True,
):
  if metadata["lang_graph_node"] == "agent" and (text := step.text()):
    print(text, end="|")


memory = MemorySaver()
agent_executor = create_react_agent(
  model=model,
  tools=tools,
  checkpointer=memory
)
config = {
  "configurable": {
    "thread_id": "abc123"
  }
}
for chunk in agent_executor.stream(
  input={
    "messages": [
      HumanMessage(
        content="hi bob!"
      )
    ],
  },
  config=config,
):
  print(chunk)
  print("___")