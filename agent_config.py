from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

from tools import get_tools

memory = MemorySaver()

"""
Change the prompt template to suit your needs.
"""
PROMPT_TEMPLATE = """You are a helpful assistant. You will be provided with a task and you need to respond accordingly."""


def get_compiled_graph():
    llm = ChatOllama(model="mistral", temperature=0.7)

    agent = create_react_agent(
        llm,
        prompt=PROMPT_TEMPLATE,
        tools=get_tools(),
        checkpointer=memory,
    )
    return agent
