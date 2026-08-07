"""One-off role labeling of the ver1.1.7 natural example commands.

Each of the catalog's example commands is classified once by an LLM into
  condition - a state/status expression (usable as an if-clause; VALUE path)
  action    - an imperative device command (FUNCTION path)
  fragment  - incomplete or ambiguous text
and the result is stored in data/path_controlled/query_roles.json so the
benchmark is reproducible and the labels are auditable. Requires an OpenAI
key in the environment (OPENAI_API_KEY or the project variables).
"""

from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CAT = os.path.join(HERE, "data", "service_list_ver1.1.7_eng.json")
OUT = os.path.join(HERE, "data", "path_controlled", "query_roles.json")
BATCH = 50
MODEL = "gpt-4.1-mini"

PROMPT = """Classify each numbered smart-home phrase into exactly one class:
- "condition": states or asks about a device state/status (could follow "if"), e.g. "The light is on", "Is the pump closed?", "Check the temperature"
- "action": an imperative command that actuates a device, e.g. "Turn on the fan", "Set the volume to 20"
- "fragment": incomplete, truncated, or not classifiable

Answer with a JSON object mapping each phrase NUMBER (as a string) to its label:
{"labels": {"1": "condition", "2": "action", ...}}

Phrases:
"""


def main() -> int:
    api_key = (os.getenv("OPENAI_API_KEY_PROJ_BENCH") or os.getenv("JOI_EVAL_OPENAI_API_KEY")
               or os.getenv("JOI_V15_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY"))
    if not api_key:
        print("no OpenAI API key configured", file=sys.stderr)
        return 1
    from openai import OpenAI

    client = OpenAI(api_key=api_key)

    with open(CAT, encoding="utf-8") as f:
        cat = json.load(f)
    items = [(dev, str(q).strip()) for dev, obj in cat.items()
             for q in obj.get("examples", []) or [] if str(q).strip()]
    print(f"{len(items)} examples to label")

    labels = {}
    valid = {"condition", "action", "fragment"}
    for start in range(0, len(items), BATCH):
        chunk = items[start:start + BATCH]
        numbered = "\n".join(f"{i + 1}. {q}" for i, (_, q) in enumerate(chunk))
        for attempt in range(3):
            resp = client.chat.completions.create(
                model=MODEL, temperature=0,
                response_format={"type": "json_object"},
                messages=[{"role": "user", "content": PROMPT + numbered}],
            )
            out = json.loads(resp.choices[0].message.content)["labels"]
            got = {int(k): v for k, v in out.items()
                   if str(k).isdigit() and 1 <= int(k) <= len(chunk) and v in valid}
            if len(got) == len(chunk):
                break
            print(f"  batch {start}: {len(got)}/{len(chunk)} valid, retry {attempt + 1}", flush=True)
        else:
            print(f"batch {start}: failed after retries", file=sys.stderr)
            return 1
        for i, (dev, q) in enumerate(chunk):
            labels[f"{dev}\t{q}"] = got[i + 1]
        print(f"  {start + len(chunk)}/{len(items)}", flush=True)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({"model": MODEL, "labels": labels}, f, ensure_ascii=False, indent=1)
    from collections import Counter

    print("distribution:", dict(Counter(labels.values())))
    print(f"saved -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
