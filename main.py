import argparse
from app.graph import build_graph
from langchain_core.messages import SystemMessage, HumanMessage

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--question", required=True, help="User question")
    p.add_argument("--image", default="", help="Local path or data URL (optional)")
    return p.parse_args()

def main():
    args = parse_args()
    graph, config, system_msg = build_graph()

    # Build user message: text + optional image path hint for tools
    content = [{"type": "text", "text": args.question}]
    if args.image:
        content[0]["text"] += f"\nImage path: {args.image}"

    msg = HumanMessage(content=content)
    result = graph.invoke({"messages": [system_msg, msg]}, config)

    # Print final LLM message
    final = result["messages"][-1]
    print("\n=== ASSISTANT ===\n")
    print(final.content)

if __name__ == "__main__":
    main()
