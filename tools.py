from langchain.tools import tool

"""
Define tools for the agent to use.
After defining a tool, add it to get_tools() to make it available to the agent.
"""


@tool
def example_tool(text: str) -> str:
    """Returns True if example_tool is called."""
    return True


def get_tools():
    return [example_tool]
