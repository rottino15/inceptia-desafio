from langchain.tools import tool


@tool
def example_tool(text: str) -> str:
    """Returns True if example_tool is called."""
    return True


def get_tools():
    return [example_tool]
