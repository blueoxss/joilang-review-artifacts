#!/usr/bin/env python3
"""Confirm the committed notebook outputs and results files are the ones a run wrote.

The tutorial ships executed, and `results/` ships the numbers those cells produced,
so a reader who never runs anything still reads real measurements. That only holds
while nobody edits a stored output or a results file by hand, and "nobody edited it"
is not something prose can establish. This script establishes it: it hashes the three
things that carry measured values into the repository and compares them against the
digests recorded below.

    python3 premapping/check_stored_outputs.py

It prints one line per artifact and a VERDICT line, and exits 0 when everything
matches, 1 otherwise. It reads only; it never writes and never executes the notebook.

What is hashed, and how
-----------------------
Three surfaces, because they fail in three different ways:

1. `outputs`   — the stored output of every code cell, in file order. Catches an
                 edited number, a trimmed traceback, a pasted-in result.
2. `execution_count` — the counters, in file order. Catches a cell re-run in
                 isolation, or a run that was not top-to-bottom.
3. `results/*` — the JSON and Markdown a run writes out. Catches a figure adjusted
                 to agree with a document.

Cell *source* is deliberately outside the scope of all three. Prose and comments are
meant to be edited; a markdown fix is not an integrity breach. What must not drift is
the relationship between a stored number and the run that produced it.

The canonicalisation is fixed here rather than left to the caller, because a digest
is only evidence if two people compute it the same way. Outputs and counters are
serialised with `json.dumps(..., sort_keys=True, ensure_ascii=False,
separators=(",", ":"))` over a list built from the code cells in file order, and that
serialisation is hashed with SHA-256. Results files are hashed as raw bytes. Both the
full digest and its first 32 characters are printed, so a digest quoted in short form
elsewhere can still be checked against this output.

Updating the baseline
---------------------
Re-executing the notebook rewrites outputs, counters and results files together, and
every digest below changes at once. That is expected, and the fix is to re-run this
script with `--update` and commit the digests it prints along with the re-executed
notebook. Editing a digest here to make a hand-edited output pass defeats the only
thing the script does.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

NOTEBOOK = "tutorials/premapping_finetune_tutorial.ipynb"
RESULTS_DIR = "premapping/results"

# Digests of the committed run. Regenerate with --update after a re-execution;
# never edit one by hand to make a modified artifact pass.
EXPECTED = {
    "notebook:outputs": "c4944c8586989ae1ad868000cccdc559b0908f7185e01db6ac5c9feae83f1ce3",
    "notebook:execution_count": "c94747afa847f98b263eca5299b28438181251bdd86c930a56c2e16f685c4294",
    "results/premap_ab.json": "4d140e0b861b9e580dacdc5d4c5e4d8a9b4b2042d3284bf491316d5c2210f95e",
    "results/premap_eval_all_candidates_finetuned.json": "cadca907b4be0b43f6b3c2afae605286365103ac5d4f7d1dd43b70dda7f6bb4a",
    "results/premap_eval_all_candidates_finetuned.md": "b4651c75181d49c48f8c945ad42f85972f7e3a11b97c83ef2efb05ae9dcc1307",
    "results/premap_eval_default_run.json": "f0c5385cacb1acfd79b05a01c741f8fb55c8ae1266711713f9d359dafea27d71",
    "results/premap_eval_default_run.md": "01dbe332849ea48a274aeae62995f6828e6993bb08db55f53d5a57f753d08304",
    "results/premap_eval_split_fixes.json": "1dcd73c21c876a0be7b0d46a7312daded07c3f3d5843e51e8672fda4ef1e5a72",
    "results/premap_eval_split_fixes.md": "2a27f77c52645bfadf985d1f94e359bd2054cfecd8408e228aa86f8698c32d1c",
    "results/runtime_coverage.json": "1a71057e562c3c381c556b6cc60c92631e87f7197da94b8c75ccffbfa0950243",
    "results/runtime_coverage_e5-split-v7.json": "95b0234fa1b5c238f0539c89ab818100412197df8a804224fb8d67adc3b0d2ce",
    "results/runtime_coverage_sparse.json": "af1c4a004743194216d86446ea3c6d9e29d73e39bbbe304fb435f109cdf39b18",
}


def find_repo_root(start: Path) -> Path:
    """Walk up until a directory holds both gpt_mg/ and premapping/.

    Same rule the tutorial's setup cell uses -- walk up until gpt_mg/ and
    premapping/ are both present -- anchored at this file rather than at the
    caller's working directory, so the script finds its own checkout from
    anywhere. JOI_REPO_ROOT names the root outright when the search cannot.
    """
    override = os.environ.get("JOI_REPO_ROOT")
    if override:
        root = Path(override).expanduser().resolve()
        if not ((root / "gpt_mg").is_dir() and (root / "premapping").is_dir()):
            raise SystemExit(
                f"JOI_REPO_ROOT={root} does not hold both gpt_mg/ and premapping/"
            )
        return root
    for candidate in [start.resolve(), *start.resolve().parents]:
        if (candidate / "gpt_mg").is_dir() and (candidate / "premapping").is_dir():
            return candidate
    raise SystemExit(
        "could not find the repository root above "
        f"{start.resolve()}: no parent holds both gpt_mg/ and premapping/. "
        "Set JOI_REPO_ROOT to the checkout."
    )


def canonical(value) -> bytes:
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def notebook_digests(path: Path) -> dict[str, str]:
    try:
        nb = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise SystemExit(f"notebook not found: {path}")
    except json.JSONDecodeError as exc:
        raise SystemExit(f"notebook is not valid JSON ({path}): {exc}")
    code = [c for c in nb.get("cells", []) if c.get("cell_type") == "code"]
    return {
        "notebook:outputs": sha256(canonical([c.get("outputs") for c in code])),
        "notebook:execution_count": sha256(
            canonical([c.get("execution_count") for c in code])
        ),
    }


def results_digests(results_dir: Path) -> dict[str, str]:
    if not results_dir.is_dir():
        raise SystemExit(f"results directory not found: {results_dir}")
    # a directory that exists but cannot be listed or read is the same class of
    # configuration error as one that is not there - the check cannot be made -
    # and is refused the same way rather than reaching the caller as a
    # PermissionError traceback out of pathlib
    try:
        entries = sorted(results_dir.iterdir())
    except OSError as exc:
        raise SystemExit(f"cannot read the results directory {results_dir}: {exc}")
    out = {}
    for f in entries:
        if f.is_file() and f.suffix in {".json", ".md"}:
            try:
                out[f"results/{f.name}"] = sha256(f.read_bytes())
            except OSError as exc:
                raise SystemExit(f"cannot read {f}: {exc}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Verify committed notebook outputs and results files are unmodified."
    )
    ap.add_argument(
        "--update",
        action="store_true",
        help="print a fresh EXPECTED block instead of checking (use after a re-execution)",
    )
    args = ap.parse_args()

    root = find_repo_root(Path(__file__).parent)
    digests = notebook_digests(root / NOTEBOOK)
    digests.update(results_digests(root / RESULTS_DIR))

    if args.update:
        print("EXPECTED = {")
        for name, digest in digests.items():
            print(f'    "{name}": "{digest}",')
        print("}")
        return 0

    # the checkout is named, not located: this line is pasted into reports, and
    # the digests below are what identify the run either way
    print(f"checkout: {root.name}/\n")
    print(f"{'artifact':52s} {'sha256 (first 32)':34s} status")
    ok = True
    for name, digest in digests.items():
        expected = EXPECTED.get(name)
        if not expected:
            status = "NO BASELINE — add it with --update"
            ok = False
        elif expected == digest:
            status = "OK"
        else:
            status = f"MODIFIED (expected {expected[:32]})"
            ok = False
        print(f"{name:52s} {digest[:32]:34s} {status}")

    missing = sorted(set(EXPECTED) - set(digests))
    for name in missing:
        print(f"{name:52s} {'-':34s} MISSING FROM THE CHECKOUT")
        ok = False

    print(
        "\nVERDICT:",
        "stored outputs and results files are unmodified"
        if ok
        else "*** ONE OR MORE ARTIFACTS DIFFER FROM THE COMMITTED RUN ***",
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
