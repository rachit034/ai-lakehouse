#!/usr/bin/env python3
"""
Accuracy check for local-LLM graph Q&A (/graph/nl).

For each case in graph_nl_cases.json: ask /graph/nl the question, run the
case's reference Cypher via /graph/query, and compare the *values* the two
return (column names are ignored — the model may alias differently). Reports
per-case pass/fail, attempts, latency, and overall accuracy, so a change of
model, prompt, or example library is a measured decision instead of a guess.

Keep these cases out of gateway/graph_examples.json — an example the model
has seen verbatim proves nothing about questions it hasn't.

Usage (stack running, data_modeler key so results aren't redacted):
    export API_KEY_DATA_MODELER=$(grep API_KEY_DATA_MODELER .env | cut -d= -f2)
    python3 scripts/eval_graph_nl.py [--schema ecommerce] [--only case-id ...]

Compare modes:
    set          same multiset of rows (order ignored)
    scalar       single value equal
    first_value  first column of first row equal (e.g. "which X is top")
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

CASES = Path(__file__).with_name("graph_nl_cases.json")


def _post(base: str, key: str, path: str, body: dict, timeout: int = 900) -> dict:
    req = urllib.request.Request(
        f"{base}{path}",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "X-API-Key": key, "X-Agent-Name": "eval_graph_nl"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def _norm(v):
    if isinstance(v, float):
        return round(v, 4)
    if isinstance(v, str):
        return v.strip().lower()
    return v


def _rows(records: list[dict]) -> list[tuple]:
    return [tuple(_norm(v) for v in rec.values()) for rec in records]


def matches(mode: str, got: list[dict], want: list[dict]) -> bool:
    g, w = _rows(got), _rows(want)
    if mode == "set":
        return Counter(g) == Counter(w)
    if mode == "scalar":
        return len(g) == 1 and len(w) == 1 and g[0][:1] == w[0][:1]
    if mode == "first_value":
        return bool(g) and bool(w) and g[0][0] == w[0][0]
    raise ValueError(f"unknown compare mode {mode!r}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gateway", default=os.environ.get("GATEWAY_URL", "http://localhost:8088"))
    ap.add_argument("--schema", default=None, help="pass through to /graph/nl")
    ap.add_argument("--only", nargs="*", help="run only these case ids")
    args = ap.parse_args()

    key = os.environ.get("API_KEY_DATA_MODELER")
    if not key:
        sys.exit("Set API_KEY_DATA_MODELER (see the usage note at the top of this file).")

    cases = json.loads(CASES.read_text())
    if args.only:
        cases = [c for c in cases if c["id"] in set(args.only)]

    passed = 0
    for case in cases:
        start = time.monotonic()
        try:
            got = _post(args.gateway, key, "/graph/nl",
                        {"question": case["question"], "schema": args.schema, "answer": False})
            want = _post(args.gateway, key, "/graph/query", {"cypher": case["reference_cypher"]})
            ok = matches(case["compare"], got["records"], want["records"])
            detail = f"attempts={got['attempts']}"
        except urllib.error.HTTPError as e:
            ok, got = False, None
            detail = f"HTTP {e.code}: {e.read().decode()[:200]}"
        except Exception as e:  # noqa: BLE001 — report and keep going
            ok, got = False, None
            detail = f"{type(e).__name__}: {e}"
        passed += ok
        print(f"{'PASS' if ok else 'FAIL'}  {case['id']:<32} {time.monotonic() - start:6.1f}s  {detail}")
        if not ok and got:
            print(f"      generated: {got['cypher']}")

    print(f"\n{passed}/{len(cases)} passed ({100 * passed / max(len(cases), 1):.0f}%)")
    return 0 if passed == len(cases) else 1


if __name__ == "__main__":
    sys.exit(main())
