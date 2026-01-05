import argparse, yaml, json, sys
from utils.llm_client import LLMClient

def check(client: LLMClient, name: str) -> dict:
    info = client.list_models()
    # do a very small chat call
    msgs = client.make_messages(system_prompt="You are a test server.", user_prompt="Reply with exactly: OK")
    try:
        out = client.chat_complete(msgs, max_tokens=8, temperature=0.0)
    except Exception as e:
        out = f"ERROR: {e}"
    return {"name": name, "models": info, "sample": out}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/models.yaml")
    args = ap.parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    results = {}
    for role in ["optimize_model", "evaluate_model", "execute_model"]:
        client = LLMClient(cfg[role])
        results[role] = check(client, role)
    print(json.dumps(results, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
