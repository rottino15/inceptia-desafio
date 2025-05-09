from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

from tools import get_tools

memory = MemorySaver()


def get_compiled_graph():
    llm = ChatOllama(model="mistral", temperature=0.7)

    agent = create_react_agent(
        llm,
        tools=get_tools(),
        checkpointer=memory,
    )
    return agent
