import uuid

try:
    from langchain_core.messages import HumanMessage
except ImportError:  # pragma: no cover - fallback for older langchain versions
    from langchain.schema import HumanMessage

from agent_config import get_compiled_graph


def main():
    llm_graph = get_compiled_graph()
    thread_id = uuid.uuid4()
    config = {"configurable": {"thread_id": thread_id}}
    while True:
        user_input = input("User: ")
        if user_input.lower() in {"exit", "quit"}:
            break
        input_message = HumanMessage(content=user_input)
        for event in llm_graph.stream(
            {"messages": [input_message]}, config, stream_mode="values"
        ):
            event["messages"][-1].pretty_print()


if __name__ == "__main__":
    main()