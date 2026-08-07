"""Does pre-mapping cost the generation path anything?

``runtime_coverage.py`` measures what the shortlist *removes*: how much of what
an accepted answer uses is still in the prompt after twelve services have
replaced the catalogs. It cannot say whether removing it costs anything, because
the model also supplies services from its own training. That is the question a
reader deciding whether to pass ``--premap`` actually has, and it needs the
generation path, not the retriever.

So this script runs the same commands twice through ``run.py`` itself - once
with ``--premap`` and once without - and scores both the same way:

  sample    a fixed, seeded draw from ``JOICommands-170.csv``; seed 2025, the
            seed the notebook's pools use, so the draw is the same every run
  arms      ``run.py --premap version0_6 "<cmd>"`` against ``run.py "<cmd>"``,
            each a fresh process, so nothing of one arm's state reaches the
            other
  scored    the service set the generated ``code`` names, against the service
            set of each of ``gt1``-``gt3``; a row is correct when it matches
            some one accepted scenario exactly. Services are read with
            ``runtime_coverage.gold_pairs``, so both measurements agree on what
            a service is
  reported  correct counts per arm, and the rows where the two arms disagree,
            named by ``test_no`` so a disagreement can be re-run by hand

Exact set match is a strict score and both arms pay it equally; the comparison
between them is the point, not either number on its own. A sample this size
settles direction, not a small difference: with 20 rows a one-row gap is noise,
and the file records the sample size next to the counts so it cannot be read as
more than it is.

Needs ``OPENAI_API_KEY`` in the environment, like any generation run, and makes
two model calls per row. Writes ``results/premap_ab.json``.

    python3 premapping/premap_ab.py --rows 20
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import random
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "gpt_mg"))
import premap  # noqa: E402

from runtime_coverage import (GOLD_FIELDS, catalog_pairs, gold_pairs,  # noqa: E402
                              provenance)

COMMANDS = os.path.join(ROOT, "JOICommands-170.csv")
VERSION_DIR = os.path.join(ROOT, "gpt_mg", "version0_6")
RUN_PY = os.path.join(ROOT, "gpt_mg", "run.py")
OUT = os.path.join(HERE, "results", "premap_ab.json")
SEED = 2025
# The two arms do not take the same branch of run.py and do not print the same
# lines: only the pre-mapped arm reaches the `Candidate #1:` line. `Output:: >>`
# carries the parsed scenarios in both, so it is what both arms are read from -
# reading a line only one arm prints would have scored the other arm zero and
# looked like pre-mapping winning every row.
OUTPUT_RE = re.compile(r"^Output:: >>\s+(\[.*\])$", re.M)


def generate(command, premap_on):
    """One `run.py` process. Returns (services, premap_line, raw_ok)."""
    argv = [sys.executable, os.path.basename(RUN_PY)]
    if premap_on:
        argv += ["--premap", "version0_6"]
    argv.append(command)
    proc = subprocess.run(argv, cwd=os.path.dirname(RUN_PY), text=True,
                          capture_output=True, timeout=300)
    hit = OUTPUT_RE.search(proc.stdout or "")
    if not hit:
        # a row the model or the parse lost is not a wrong answer: it is kept
        # out of both counts and reported separately, so a run that fails for an
        # unrelated reason cannot read as the arm being worse
        return None, None, False
    try:
        scenarios = ast.literal_eval(hit.group(1))
        code = scenarios[0].get("code", "") if scenarios else ""
    except (ValueError, SyntaxError, AttributeError, IndexError):
        return None, None, False
    line = next((ln for ln in proc.stdout.splitlines()
                 if ln.startswith("Premap ::")), None)
    return code, line, True


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="run.py with --premap against run.py without it.")
    ap.add_argument("--rows", type=int, default=20,
                    help="sample size, drawn with seed 2025 (default 20)")
    args = ap.parse_args(argv)

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit(
            "OPENAI_API_KEY is not set; both arms of this comparison call the "
            "model, so it cannot run without one. Export it and re-run - it is "
            "read from the environment and never from a file in this repository."
        )
    try:
        with open(COMMANDS, encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
    except OSError as exc:
        raise SystemExit(f"cannot read the command set: {exc}; "
                         f"it belongs at {os.path.basename(COMMANDS)} in the repository root")
    devices = {p.split(".", 1)[0]
               for role in catalog_pairs(VERSION_DIR).values() for p in role}

    eligible = [r for r in rows
                if (r.get("command_eng") or "").strip()
                and gold_pairs(r.get("gt1"), devices)]
    sample = random.Random(SEED).sample(eligible, min(args.rows, len(eligible)))

    tally = {"premap": 0, "no_premap": 0}
    lost = {"premap": 0, "no_premap": 0}
    disagreements, per_row = [], []
    started = time.perf_counter()

    for i, row in enumerate(sample, 1):
        command = row["command_eng"].strip()
        accepted = [g for g in (gold_pairs(row.get(f), devices) for f in GOLD_FIELDS) if g]
        result = {"test_no": row.get("test_no"), "command": command}
        for arm, on in (("premap", True), ("no_premap", False)):
            code, line, ok = generate(command, on)
            if not ok:
                lost[arm] += 1
                result[arm] = {"correct": None, "note": "no candidate parsed"}
                continue
            got = gold_pairs(code, devices)
            correct = any(got == a for a in accepted)
            tally[arm] += correct
            result[arm] = {"correct": correct, "services": sorted(got)}
            if on:
                result["premap_line"] = line
        if result["premap"].get("correct") != result["no_premap"].get("correct"):
            disagreements.append(result["test_no"])
        per_row.append(result)
        print(f"[{i}/{len(sample)}] {result['test_no']}  "
              f"premap={result['premap'].get('correct')}  "
              f"plain={result['no_premap'].get('correct')}")

    report = {
        "run": "premap_ab",
        "question": ("does replacing the catalogs with a twelve-service "
                     "shortlist change what the generation path produces"),
        "corpus": {
            "file": "JOICommands-170.csv",
            "query_field": "command_eng",
            "gold_fields": list(GOLD_FIELDS),
            "sample_size": len(sample),
            "seed": SEED,
            "eligible_rows": len(eligible),
        },
        "scoring": ("exact match between the service set the generated code "
                    "names and the service set of some one accepted scenario"),
        "arms": {
            "premap": {
                "command": 'run.py --premap version0_6 "<cmd>"',
                "correct": tally["premap"],
                "rate": round(tally["premap"] / len(sample), 4),
                "rows_without_a_parsed_candidate": lost["premap"],
            },
            "no_premap": {
                "command": 'run.py "<cmd>"',
                "correct": tally["no_premap"],
                "rate": round(tally["no_premap"] / len(sample), 4),
                "rows_without_a_parsed_candidate": lost["no_premap"],
            },
        },
        "delta_correct": tally["premap"] - tally["no_premap"],
        "rows_where_the_arms_disagree": disagreements,
        "elapsed_seconds": round(time.perf_counter() - started, 1),
        "provenance": provenance(
            __file__, regenerable_without_gpu_or_key=False,
            needs="OPENAI_API_KEY and two model calls per row; the model is "
                  "sampled at temperature 0.3, so a re-run need not repeat this "
                  "one row for row"),
        "rows": per_row,
    }

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    try:
        with open(OUT, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
            f.write("\n")
    except OSError as exc:
        raise SystemExit(f"cannot write {os.path.relpath(OUT, ROOT)}: {exc}")

    a = report["arms"]
    print(f"\nsample {len(sample)} rows, seed {SEED}")
    print(f"  with --premap: {a['premap']['correct']}/{len(sample)} = {a['premap']['rate']}")
    print(f"  without:       {a['no_premap']['correct']}/{len(sample)} = {a['no_premap']['rate']}")
    print(f"  delta: {report['delta_correct']:+d}   disagreeing rows: {disagreements}")
    print(f"written: {os.path.relpath(OUT, ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
