# app/graph.py
from typing import Annotated, Sequence, TypedDict
import operator
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, SystemMessage, ToolMessage
from app.config import OPENAI_API_KEY, OPENAI_MODEL
from app.rag.indexer import QUERY_ENGINE
from app.tools import tavily_search, ocr, obj_detect
from app.utils import extract_user_text

class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]

def _llm_with_tools():
    llm = ChatOpenAI(model=OPENAI_MODEL, api_key=OPENAI_API_KEY)
    return llm.bind_tools([tavily_search, ocr, obj_detect], tool_choice="auto")

def rag_node(state: State) -> State:
    user_text = extract_user_text(state["messages"])
    try:
        rag_resp = QUERY_ENGINE.query(user_text)
        rag_answer = getattr(rag_resp, "response", str(rag_resp))
        ctx = (
            "RAG context (for the assistant to use as background):\n"
            f"{rag_answer}\n"
            "End of RAG context."
        )
        return {"messages": [SystemMessage(content=ctx)]}
    except Exception as e:
        return {"messages": [SystemMessage(content=f"RAG context unavailable: {e}")]}

def agent_node(state: State) -> State:
    llm = _llm_with_tools()
    resp = llm.invoke(state["messages"])
    return {"messages": [resp]}

def tool_node(state: State) -> State:
    last = state["messages"][-1]
    tool_messages = []
    if getattr(last, "tool_calls", None):
        tool_map = {"tavily_search": tavily_search, "ocr": ocr, "obj_detect": obj_detect}
        for tc in last.tool_calls:
            name = tc["name"]
            args = tc.get("args", {}) or {}
            if isinstance(args, str):
                if name == "tavily_search":
                    args = {"query": args}
                elif name == "ocr":
                    args = {"hint": args}
                elif name == "obj_detect":
                    args = {"image": args}
            out = tool_map[name].invoke(args)
            tool_messages.append(ToolMessage(content=out, tool_call_id=tc["id"]))
    return {"messages": tool_messages}

def should_continue(state: State) -> str:
    last = state["messages"][-1]
    return "tools" if getattr(last, "tool_calls", None) else END

def build_graph():
    builder = StateGraph(State)
    builder.add_node("rag", rag_node)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", tool_node)
    builder.add_edge(START, "rag")
    builder.add_edge("rag", "agent")
    builder.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    builder.add_edge("tools", "agent")
    graph = builder.compile()

    system_msg = SystemMessage(
        content=(
            "You are a text-only assistant with access to tools.\n"
            "A RAG context message will always be provided before you respond; use it if helpful.\n"
            "You CANNOT see or analyze images yourself. Never attempt to describe or read an image directly.\n"
            "Goal: Answer the user. Call tools ONLY if they materially improve the result.\n"
            "Tools:\n"
            "- ocr(hint): extract English text but ignore Chinese.\n"
            "- obj_detect(image): detect objects on a local image; returns detections and a saved annotated image path (and data_url if available).\n"
            "- tavily_search(query): helpful for external facts/sources.\n"
            "If ocr is called, prefer sensible English tokens; ignore gibberish.\n"
        )
    )
    config = {"configurable": {"thread_id": "run-1"}}
    return graph, config, system_msg
