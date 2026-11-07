#!/usr/bin/env python3
"""Live smoke test against a running TTS service.

Usage:  python scripts/smoke.py [base_url] [--out DIR]

Checks the whole contract the app relies on: health, capabilities, one English
and one Russian render (headers, WAV sanity), lenient parameter handling, and
idempotent voice install. Prints a JSON summary and exits non-zero on failure.

Deliberately NOT a pytest test: it needs a real service with real weights, which
CI (and a fresh clone) does not have.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import httpx

BASE = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1].startswith("http") \
    else "http://127.0.0.1:9000"
OUT = Path.home() / "tts-smoke"

EN_TEXT = ("That was Nirvana with Come as You Are, from the 1991 album Nevermind. "
           "Next up, something from the same decade.")
RU_TEXT = ("Это была группа Кино с песней Звезда по имени Солнце. "
           "Виктор Цой записал её в 1989 году.")

failures: list[str] = []


def check(cond: bool, what: str) -> None:
    print(("  ok   " if cond else "  FAIL ") + what)
    if not cond:
        failures.append(what)


def wav_duration(payload: bytes) -> tuple[int, float, str]:
    import io
    import wave

    with wave.open(io.BytesIO(payload), "rb") as wf:
        return wf.getframerate(), wf.getnframes() / wf.getframerate(), \
            f"{wf.getnchannels()}ch/{wf.getsampwidth() * 8}bit"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    client = httpx.Client(base_url=BASE, timeout=300)

    print(f"== health ({BASE})")
    health = client.get("/v1/health").json()
    check(health.get("ok") is True, f"health ok (detail={health.get('detail')!r})")
    check(health.get("steps") == 5, f"fixed steps == 5 (got {health.get('steps')})")

    print("== capabilities")
    caps = client.get("/v1/capabilities").json()
    check(caps["api"]["version"] == "1", "api version 1")
    check(bool(caps.get("voices")), f"{len(caps.get('voices', []))} voices offered")
    check([lang["code"] for lang in caps["languages"]] == ["en", "ru"],
          "languages are en+ru")
    param_keys = [p["key"] for p in caps["params"]]
    check(bool(param_keys), f"params declared: {param_keys}")
    for p in caps["params"]:
        check({"key", "label", "type", "default"} <= set(p),
              f"param {p['key']} is fully described")

    # A voice per language, from what the service actually offers.
    def pick(lang: str) -> str:
        for v in caps["voices"]:
            if lang in v["languages"] and v["installed"]:
                return v["id"]
        return caps["defaults"]["voice"]

    results = []
    for tag, text, lang, voice in (
            ("en", EN_TEXT, "en", pick("en")),
            ("ru", RU_TEXT, "ru", "M2" if any(v["id"] == "M2" for v in caps["voices"])
             else pick("ru"))):
        print(f"== speech {tag.upper()} (voice {voice})")
        params = {"speed": 1.0}
        if any(p["key"] == "sentence_pause" for p in caps["params"]):
            params["sentence_pause"] = 0.3
        resp = client.post("/v1/speech", json={
            "text": text, "voice": voice, "language": lang, "params": params})
        check(resp.status_code == 200, f"200 with audio ({resp.status_code})")
        if resp.status_code != 200:
            print(resp.text[:400])
            continue
        rate, duration, fmt = wav_duration(resp.content)
        check(duration > 1.0, f"audio is {duration:.2f}s @ {rate} Hz ({fmt})")
        check(resp.headers.get("X-TTS-Language") == lang, "language echoed in headers")
        check(resp.headers.get("X-TTS-Steps") == "5", "render used 5 steps")
        check(json.loads(resp.headers["X-TTS-Params-Used"])["speed"] == 1.0,
              "params used echoed")
        path = OUT / f"smoke_{tag}.wav"
        path.write_bytes(resp.content)
        results.append({"language": lang, "voice": voice, "duration": duration,
                        "sample_rate": rate, "render_ms":
                        int(resp.headers.get("X-TTS-Render-Ms", 0)),
                        "file": str(path)})

    print("== lenient parameters")
    resp = client.post("/v1/speech", json={
        "text": "Test.", "voice": results[0]["voice"] if results else None,
        "params": {"speed": 99, "noise_w": 0.8, "nonsense": 1}})
    check(resp.status_code == 200, "stale params do not fail the render")
    if resp.status_code == 200:
        clamped = json.loads(resp.headers["X-TTS-Params-Clamped"])
        ignored = json.loads(resp.headers["X-TTS-Params-Ignored"])
        check("speed" in clamped, f"out-of-range clamped: {clamped}")
        check("noise_w" in ignored and "nonsense" in ignored,
              f"unknown keys ignored: {ignored}")

    print("== errors")
    check(client.post("/v1/speech", json={"text": " "}).status_code == 400,
          "empty text is 400")
    check(client.post("/v1/speech",
                      json={"text": "hi", "voice": "NOPE"}).status_code == 404,
          "unknown voice is 404")
    check(client.post("/v1/speech",
                      json={"text": "hi", "language": "xx"}).status_code == 400,
          "bad language is 400")

    print("== install idempotency")
    voice = caps["voices"][0]["id"]
    again = client.post(f"/v1/voices/{voice}/install").json()
    check(again.get("installed") is True and again.get("already") is True,
          f"re-install of {voice} reports already=true")

    print("== health counters")
    health2 = client.get("/v1/health").json()
    expected = len(results) + 1  # the two language renders + the lenient one
    check(health2["renders"] >= expected,
          f"renders counted ({health2['renders']} >= {expected})")
    check(bool(health2["last_render_ms"]), "last render time recorded")

    summary = {"base": BASE, "engine": health2["engine"],
               "voices": len(caps["voices"]), "params": param_keys,
               "renders": results, "failures": failures}
    (OUT / "smoke.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "renders"}, indent=2))
    print(f"\nfiles in {OUT}")
    if failures:
        print(f"FAILED {len(failures)} check(s)")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
