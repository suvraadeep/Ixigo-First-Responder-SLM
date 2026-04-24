import sys
import time
import statistics
import requests

URL = "http://localhost:8000"

PROBES = [
    "I want to cancel my ticket",
    "When will I get my refund",
    "Help me with web check-in",
    "Confirm my booking",
    "Web check-in not working",
    "Has my payment been received",
    "Refund is still pending",
    "Verify name correction",
]


def wait_for_server(timeout=120):
    print(f"Waiting for server at {URL} ...")
    for _ in range(timeout):
        try:
            r = requests.get(f"{URL}/health", timeout=1)
            if r.ok:
                info = r.json()
                print(f"  ✅ Ready — device={info['device']}, model={info['model']}")
                return
        except Exception:
            pass
        time.sleep(1)
    sys.exit(f"  ❌ Server did not come up within {timeout}s")


def warmup(n=8):
    print(f"\nWarming up client ({n} iterations)...")
    for _ in range(n):
        requests.post(f"{URL}/chat", json={"text": "hello"}, timeout=15)


def smoke_test():
    print("\n" + "=" * 70)
    print("SMOKE TEST")
    print("=" * 70)
    full_lats, ttft_lats = [], []
    for q in PROBES:
        r = requests.post(f"{URL}/chat", json={"text": q}, timeout=15).json()
        full_lats.append(r["latency_ms"])
        ttft_lats.append(r["ttft_ms"])
        print(f"  {q!r:45s} -> {r['response']!r:40s}  "
              f"[full {r['latency_ms']:5.1f} ms | TTFT {r['ttft_ms']:5.1f} ms]")
    return full_lats, ttft_lats


def benchmark(n_rounds=5):
    print("\n" + "=" * 70)
    print(f"BENCHMARK ({n_rounds} rounds × {len(PROBES)} probes)")
    print("=" * 70)
    full_lats, ttft_lats = [], []
    for _ in range(n_rounds):
        for q in PROBES:
            r = requests.post(f"{URL}/chat", json={"text": q}, timeout=15).json()
            full_lats.append(r["latency_ms"])
            ttft_lats.append(r["ttft_ms"])
    return full_lats, ttft_lats


def stats(name, lats):
    lats = sorted(lats)
    p50 = lats[len(lats) // 2]
    p95 = lats[int(len(lats) * 0.95)]
    avg = statistics.mean(lats)
    ok  = "✅" if p50 < 100 else "❌"
    print(f"  {name:25s}  avg {avg:6.1f} ms  |  p50 {p50:6.1f} ms  |  p95 {p95:6.1f} ms   {ok} <100 ms")
    return p50


def test_invalid():
    print("\n" + "=" * 70)
    print("INPUT VALIDATION TESTS")
    print("=" * 70)

    r = requests.post(f"{URL}/chat", json={"text": ""}, timeout=5)
    print(f"  empty text           -> {r.status_code}  {'✅' if r.status_code == 422 else '❌'}")

    long_text = "word " * 150
    r = requests.post(f"{URL}/chat", json={"text": long_text}, timeout=5)
    print(f"  >120 word text       -> {r.status_code}  {'✅' if r.status_code == 400 else '❌'}")

    r = requests.post(f"{URL}/chat", json={"text": "hi", "max_new_tokens": 100}, timeout=5)
    print(f"  oversized token req  -> {r.status_code}  {'✅' if r.status_code == 422 else '❌'}")


def main():
    wait_for_server()
    warmup()
    smoke_test()
    full_lats, ttft_lats = benchmark(n_rounds=5)

    print("\n" + "=" * 70)
    print(f"LATENCY ({len(full_lats)} requests)")
    print("=" * 70)
    p50_full = stats("Full response", full_lats)
    p50_ttft = stats("TTFT", ttft_lats)

    test_invalid()

    passed = (p50_ttft < 100) or (p50_full < 100)
    print("\n" + "=" * 70)
    print(f"BONUS <100 ms TARGET:  {'✅ PASS' if passed else '❌ FAIL'}")
    print("=" * 70)


if __name__ == "__main__":
    main()