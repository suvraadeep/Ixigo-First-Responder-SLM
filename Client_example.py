import requests

API_URL = "http://localhost:8000/chat"


def first_response(user_utterance: str) -> dict:

    r = requests.post(
        API_URL,
        json={"text": user_utterance, "max_new_tokens": 7},
        timeout=10,
    )
    r.raise_for_status()
    return r.json()


if __name__ == "__main__":
    samples = [
        "I want to cancel my ticket",
        "When will I get my refund",
        "My payment is not going through",
    ]
    for s in samples:
        out = first_response(s)
        print(f"user:     {s}")
        print(f"response: {out['response']}")
        print(f"ttft:     {out['ttft_ms']:.1f} ms")
        print(f"full:     {out['latency_ms']:.1f} ms")
        print()