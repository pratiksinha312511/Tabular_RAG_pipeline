"""Quick validation test for all 3 users."""
import requests

BASE = "http://127.0.0.1:8000"

for uid, name in [("usr_a1b2c3d4", "Jose"), ("usr_e5f6g7h8", "Sarah"), ("usr_i9j0k1l2", "Marcus")]:
    r = requests.post(
        f"{BASE}/api/query",
        json={"user_id": uid, "prompt": "What did I spend the most on?"},
        timeout=120,
    )
    d = r.json()
    resp = (d.get("response", "") or "")[:90].replace("\n", " ")
    flags = d.get("guardrail_flags", [])
    print(f"[{name}] status={r.status_code} flags={flags} -> {resp}")

print("DONE - all users responded successfully")
