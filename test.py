import time
from app.agent import graph

print("Initializing LangGraph agent pipeline...", flush=True)
config = {"configurable": {"thread_id": "user_123"}}

question = "What causes headaches?"
print(f"Asking: '{question}'", flush=True)
print("Running pipeline (Pinecone Knowledge Base -> Web Search -> Gemini LLM)...", flush=True)

start_time = time.time()
result = graph.invoke(
    {
        "question": question,
        "history": [],
        "image_description": "",
        "kb": [],
        "web": [],
        "answer": "",
        "route": "",
    },
    config=config,
)
elapsed = round(time.time() - start_time, 2)

print(f"\n--- Response received in {elapsed}s (Route: {result.get('route')}) ---")
print(result.get("answer", "No answer generated."))
print("\n--- Sources Used ---")
print(f"KB (Gale PDF) Excerpts: {len(result.get('kb', []))} citations")
for item in result.get('kb', []):
    print(f"  - Page {item.get('page')} (score: {item.get('score')})")
print(f"Web Excerpts: {len(result.get('web', []))} citations")
for item in result.get('web', []):
    print(f"  - {item.get('title')}: {item.get('url')}")