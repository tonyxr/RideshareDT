#!/usr/bin/env python3
"""Optional Ollama replication of numeric_profile_threshold_v11."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import urllib.request
from typing import Any, Mapping


PROMPT_PROTOCOL = "numeric_profile_threshold_v11"


def prompt(profile: Mapping[str, Any]) -> str:
    numeric = json.dumps(dict(profile), sort_keys=True, separators=(",", ":"))
    return (
        "Return one JSON object with a single numeric field price_threshold. "
        "Estimate the maximum ride price in dollars this rider would authorize "
        "from the supplied numeric profile. Do not add prose. Profile: " + numeric
    )


def query_ollama(url: str, model: str, text: str) -> float:
    request = urllib.request.Request(
        url.rstrip("/") + "/api/generate",
        data=json.dumps({"model": model, "prompt": text, "stream": False,
                         "options": {"temperature": 0}}).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        payload = json.loads(response.read().decode("utf-8"))
    value = json.loads(payload["response"])
    threshold = float(value["price_threshold"])
    if not 0.25 <= threshold <= 250.0:
        raise ValueError(f"implausible threshold {threshold}")
    return threshold


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="qwen2.5:7b-instruct")
    parser.add_argument("--url", default="http://127.0.0.1:11434")
    parser.add_argument("--limit", type=int, default=4000)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed = 0
    with args.profiles.open(encoding="utf-8") as source, args.output.open("w", encoding="utf-8") as target:
        for line in source:
            if completed >= args.limit:
                break
            profile = json.loads(line)
            threshold = query_ollama(args.url, args.model, prompt(profile))
            target.write(json.dumps({
                "ProfileID": profile.get("ProfileID"),
                "PriceThreshold": threshold,
                "ThresholdPromptProtocol": PROMPT_PROTOCOL,
                "ThresholdPromptModel": args.model,
                "ThresholdPromptTemperature": 0,
            }, sort_keys=True) + "\n")
            target.flush()
            completed += 1
    if completed != args.limit:
        raise ValueError(f"expected {args.limit} profiles, processed {completed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
