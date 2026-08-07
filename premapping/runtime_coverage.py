"""Runtime coverage of the shipped shortlist, measured on the evaluation set.

The figures in ``results/premap_eval_*.json`` come from the notebook's own
harness, which differs from ``run_premap_split`` in the ways this directory's
README lists. They say how well an encoder ranks devices in that harness. They
do not say what a reader usually wants to know next, so this script measures it
directly from the shipped component: on the commands the generation path is
measured against, how much of what an accepted answer uses is still in the
prompt after pre-mapping has replaced the catalogs with twelve services?

  corpus    ``JOICommands-170.csv``, field ``command_eng``
  gold      every ``<device>.<service>`` pair an accepted scenario names, read
            out of ``gt1``; the second arm reads ``gt1``-``gt3`` and asks only
            that some one accepted scenario be covered whole
  selector  a scenario addresses devices by a parenthesised tag group, one tag
            (``(#Light).switch_on()``) or several (``all(#Livingroom
            #Light).switch_on()``). The device is the tag that names a catalog
            device; the rest are room, sector, group and parity tags, which
            pick instances and not services. No group in the corpus names two
            catalog devices, so that rule is unambiguous; where no tag names
            one, the first is kept, and the pair lands under ``not_in_catalog``
            with the other names no shortlist can carry. Because the tags
            select instances, ``(#Fan #SectorA).switch_switch`` and ``(#Fan
            #SectorB).switch_switch`` are one gold service, not two
  ranker    ``gpt_mg/premap.py``, ``premap_from_version_dir(command,
            version0_6)`` - the default retriever at the shipped budget, with
            none of the harness code in the path. ``--model`` serves another
            one, the way ``run.py --premap-model`` does, and writes its own
            file so an arm is never read as the shipped one
  covered   a gold pair is covered when ``<device>.<service>`` is in
            ``metadata["selected"]``, whose ``value:``/``function:`` prefix is
            dropped first

Two rates are reported. **Pair recall** is the share of gold pairs that survive;
**full coverage** is the share of commands that lose none of theirs, which is the
stricter of the two and the one a prompt has to meet for the shortlist alone to
answer the command. Neither is a generation score: a command whose services were
cut can still be answered correctly, because pre-mapping replaces the catalogs
and nothing else -- the grammar manual and the caution block reach the model
whole -- and because the model brings what it remembers on top of that.

Both rates are reported against a **budget ceiling**: what a ranker that could
read the answer would reach under the shipped budget, six services per role and
twelve merged, counting only gold pairs the runtime catalogs actually hold. It is
arithmetic over the gold sets, not a retrieval run, and it separates the two
reasons a service can be missing -- the budget had no room for it, or the ranking
did not put it there. The distance between the measured rate and the ceiling is
the part a better ranking could still recover.

Writes ``results/runtime_coverage.json`` and prints the same figures. Needs no
API key. The dense half needs ``sentence-transformers``; without it the run
still completes on the sparse half alone, and the file records which search ran
so two runs are never compared across different retrievers. A run that would
replace a stored file written by a *different* retriever stops instead, because
the documented minimal install has no dense half and would otherwise overwrite
the hybrid figures the READMEs quote; ``--label`` keeps such a run in its own
file and ``--force`` overrides.

    python3 premapping/runtime_coverage.py
    python3 premapping/runtime_coverage.py --model premapping/models/e5-split-v7
    python3 premapping/runtime_coverage.py --label sparse   # interpreter without
                                                            # sentence-transformers
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import re
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
# the component is imported, not reimplemented: this script measures what
# `run.py --premap` puts in front of the model, so it has to be the same code
sys.path.insert(0, os.path.join(ROOT, "gpt_mg"))
import premap  # noqa: E402

COMMANDS = os.path.join(ROOT, "JOICommands-170.csv")
VERSION_DIR = os.path.join(ROOT, "gpt_mg", "version0_6")
OUT = os.path.join(HERE, "results", "runtime_coverage.json")
GOLD_FIELDS = ("gt1", "gt2", "gt3")
# A JOILang scenario names a service as `<selector>.service`, in a call or a
# read. The selector is a parenthesised group of one or more `#Tag`s, optionally
# behind `all`/`any`: `(#Light)`, `all(#Livingroom #Light)`, `(#SectorB #Odd
# #Irrigator)`. Matching only the one-tag form drops the rest of the corpus's
# gold silently, which is worse than scoring it, so the whole group is captured
# and the device is resolved out of it below.
PAIR_RE = re.compile(r"\(\s*((?:#\w+\s*)+)\)\s*\.\s*(\w+)")
TAG_RE = re.compile(r"#(\w+)")


def gold_pairs(scenario, devices):
    """The set of ``Device.service`` names one accepted scenario uses.

    ``devices`` is the set of device names the runtime catalogs hold. It is what
    tells a device tag from a room, sector, group or parity tag inside a
    multi-tag selector: the device is the one tag naming a catalog device. Where
    no tag does, the first is kept so the pair is still counted -- as a miss no
    ranking can lift, alongside the other names absent from the catalogs.
    """
    pairs = set()
    for tags, svc in PAIR_RE.findall(scenario or ""):
        tag_list = TAG_RE.findall(tags)
        named = [t for t in tag_list if t in devices]
        pairs.add(f"{named[0] if named else tag_list[0]}.{svc}")
    return pairs


def catalog_pairs(version_dir):
    """``Device.service`` names the runtime catalogs hold, per role.

    Read through the component's own loader so a catalog that is missing or not
    shaped like one is refused in the single line both entry points use.
    """
    roles = {}
    for role in ("value", "function"):
        path = os.path.join(version_dir, f"service_list_ver1.5.4_{role}.json")
        roles[role] = {f"{i.get('device')}.{i.get('service')}"
                       for i in premap._load_catalog(path)}
    return roles


def guard_overwrite(out, backend, force):
    """Refuse to replace a stored run made with a different retriever.

    ``requirements.txt`` leaves ``sentence-transformers`` commented out, so the
    documented minimal install scores the sparse half alone. Writing that over a
    hybrid run would leave the READMEs quoting figures the file no longer holds,
    and nothing on screen would say so.
    """
    try:
        with open(out, encoding="utf-8") as f:
            stored = json.load(f).get("ranker", {}).get("backends_seen", {})
    except (OSError, ValueError):
        return
    seen = sorted({k.split(" (")[0] for k in stored})
    if not seen or seen == [backend] or force:
        return
    raise SystemExit(
        f"{os.path.relpath(out, ROOT)} holds a run made with {', '.join(seen)}; "
        f"this one would score with {backend}, and the two are not comparable. "
        "Install sentence-transformers to reproduce the stored run, pass "
        "--label <name> to keep this one in its own file, or --force to "
        "replace it."
    )


def provenance(script, regenerable_without_gpu_or_key, needs=""):
    """What wrote a results file, under what interpreter, and whether it redoes.

    A stored figure is checkable only when the command that produced it travels
    with it. Arguments go through the component's own path rule -- a results file
    is read on machines other than the one that wrote it, so a ``--model``
    pointing into somebody's home directory is reduced to its last name here the
    way it is in every other metadata field.
    """
    args = " ".join(premap._strip_host_paths(a) for a in sys.argv[1:])
    return {
        "command": " ".join(filter(None, [
            "python3", os.path.relpath(os.path.abspath(script), ROOT), args])),
        "interpreter": f"{os.path.basename(sys.executable)} {sys.version.split()[0]}",
        "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "regenerable_without_gpu_or_key": regenerable_without_gpu_or_key,
        "needs": needs,
    }


def ensure_writable(out):
    """Refuse an unwritable destination before the scoring loop, not after it.

    The loop takes minutes. A results directory this process cannot write is a
    configuration error like a missing catalog, and finding it only at the end -
    as a traceback out of ``json.dump``, with the run's work already spent - is
    the one way this script can cost a reader that time and tell them nothing.
    """
    try:
        os.makedirs(os.path.dirname(out), exist_ok=True)
    except OSError as exc:
        raise SystemExit(f"cannot create the directory for "
                         f"{os.path.relpath(out, ROOT)}: {exc}")
    # the file when it already exists, its directory when it does not: those are
    # the two things a write touches
    probe = out if os.path.exists(out) else os.path.dirname(out)
    if not os.access(probe, os.W_OK):
        raise SystemExit(
            f"cannot write {os.path.relpath(out, ROOT)}: no write permission on "
            f"{os.path.relpath(probe, ROOT)}. Fix that before re-running; the "
            "scoring loop takes minutes and its result would have nowhere to go."
        )


def model_label(model):
    """How a served model is named in the report and in the file name.

    A local checkpoint is named the way the caller could type it again, relative
    to the repository, because the report is read on other machines. A checkpoint
    outside the repository has no such name and keeps its basename, which is what
    ``premap`` records in ``model_name`` for the same reason.
    """
    if model and os.path.isdir(model):
        resolved = os.path.abspath(model)
        inside = os.path.relpath(resolved, ROOT)
        return inside if not inside.startswith(os.pardir) else os.path.basename(resolved)
    return model


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Coverage of the shipped shortlist on the evaluation set.")
    ap.add_argument("--force", action="store_true",
                    help="replace the stored results file even when it was "
                         "written by a different retriever")
    ap.add_argument("--model", default=None,
                    help="serve another retriever: a model key or a checkpoint "
                         "directory, as run.py --premap-model takes. The default "
                         "writes results/runtime_coverage.json; any other model "
                         "writes results/runtime_coverage_<name>.json, so the "
                         "shipped figure is never replaced by a different arm's")
    ap.add_argument("--label", default=None,
                    help="write this run to results/runtime_coverage_<label>.json. "
                         "The default arm is one retriever with sentence-transformers "
                         "and another without it; a label is how the second one is "
                         "measured without overwriting the first")
    args = ap.parse_args(argv)

    served = model_label(args.model)
    # a label names the file; failing that, a served model does. The shipped
    # file is what an unlabelled run of the default arm writes, and nothing else
    name = args.label or (os.path.basename(served) if served else "")
    out = OUT if not name else os.path.join(
        os.path.dirname(OUT),
        "runtime_coverage_%s.json" % re.sub(r"[^A-Za-z0-9._-]+", "-", name),
    )

    # a missing or malformed input is a configuration error like the ones the
    # two entry points make: one line naming the file and what it was wanted
    # for, exit 1, no traceback out of the csv or retrieval stack
    try:
        with open(COMMANDS, encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
    except OSError as exc:
        raise SystemExit(f"cannot read the command set: {exc}; "
                         f"it belongs at {os.path.basename(COMMANDS)} in the repository root")
    try:
        roles = catalog_pairs(VERSION_DIR)
    except (OSError, ValueError, TypeError) as exc:
        raise SystemExit(f"{exc}; the catalogs under {os.path.relpath(VERSION_DIR, ROOT)} "
                         "are what both the runtime and this measurement read")
    in_catalog = roles["value"] | roles["function"]
    devices = {p.split(".", 1)[0] for p in in_catalog}

    # one call before the loop, so a run that is about to overwrite a stored
    # file written by another retriever stops before spending the minutes
    probe = premap.premap_from_version_dir("turn on the light", VERSION_DIR,
                                          model=args.model)[1]
    guard_overwrite(out, probe["backend"], args.force)
    ensure_writable(out)
    # A named arm that could not be loaded still ranks - on the sparse half - and
    # would land in a file named after the encoder that never ran. Scoring it
    # under that name is the one failure this measurement cannot report on its
    # own, so a request for a model the run could not serve stops here.
    if args.model and probe["effective_search"] != probe["configured_search"] and not args.force:
        raise SystemExit(
            f"{args.model} did not load, and the run would score with "
            f"{probe['backend']} under its name: {probe['fallback']}. "
            "Check the model key or the checkpoint path, install "
            "sentence-transformers, or pass --force to measure the arm that did run."
        )

    scored = full = withheld = over_budget = 0
    hits = total = 0
    ceiling_hits = ceiling_full = 0
    full_any_gt = 0
    role_hits = collections.Counter()
    role_total = collections.Counter()
    missing_from_catalog = set()
    by_size = collections.defaultdict(lambda: collections.Counter())
    backends, budgets = collections.Counter(), collections.Counter()
    latencies = []
    no_service = []

    for row in rows:
        query = (row.get("command_eng") or "").strip()
        gold = gold_pairs(row.get("gt1"), devices)
        # a row with no command, or an answer that names no service, has no
        # coverage question to ask: it is left out rather than scored as a miss,
        # and named in the report so the exclusion can be checked rather than
        # taken on trust
        if not query or not gold:
            no_service.append(row.get("test_no"))
            continue
        started = time.perf_counter()
        _services, meta = premap.premap_from_version_dir(query, VERSION_DIR,
                                                         model=args.model)
        latencies.append((time.perf_counter() - started) * 1000.0)
        shortlist = {s.split(":", 1)[1] for s in meta["selected"]}
        backends[f'{meta["backend"]} ({meta["effective_search"]})'] += 1
        budgets[meta["max_total"]] += 1

        scored += 1
        if meta.get("no_match"):
            # a role whose scores separated nothing keeps its full catalog, so
            # its services were never at risk; counted, not silently excluded
            withheld += 1
        if len(gold) > meta["max_total"]:
            over_budget += 1
        covered = {p for p in gold if p in shortlist}
        hits += len(covered)
        total += len(gold)
        full += len(covered) == len(gold)
        # what a ranker reading the answer could have returned at this same
        # budget: the gold pairs the catalogs hold, six per role and twelve
        # merged. A pair is charged to the role whose catalog holds it, the way
        # the shortlist is filled, so a command asking seven VALUE services is
        # over the budget even though it is under the cap
        reach = collections.Counter(
            "value" if p in roles["value"] else "function"
            for p in gold if p in in_catalog
        )
        room = min(
            sum(min(reach[r], meta["top_k_per_role"]) for r in ("value", "function")),
            meta["max_total"],
        )
        ceiling_hits += room
        ceiling_full += room == len(gold)
        by_size[len(gold)]["queries"] += 1
        by_size[len(gold)]["fully_covered"] += len(covered) == len(gold)
        by_size[len(gold)]["pair_hits"] += len(covered)
        by_size[len(gold)]["pair_total"] += len(gold)
        for pair in gold:
            role = ("value" if pair in roles["value"] else
                    "function" if pair in roles["function"] else "not_in_catalog")
            role_total[role] += 1
            role_hits[role] += pair in shortlist
            if pair not in in_catalog:
                missing_from_catalog.add(pair)
        # the three ground truths are alternatives, so a command is answerable
        # from the shortlist alone as soon as one of them survives whole
        full_any_gt += any(
            g and g <= shortlist
            for g in (gold_pairs(row.get(f), devices) for f in GOLD_FIELDS)
        )

    if not scored:
        # writing a file of zeros would look like a measurement; nothing was
        # measured, and the column names are the thing to check
        raise SystemExit(
            f"no row of {os.path.basename(COMMANDS)} pairs a command_eng with a gt1 "
            "naming a <selector>.service: check those two column names"
        )

    # whether an encoder scored anything, read off the backends the rows actually
    # ran on rather than off the model that was requested. A run on an
    # interpreter without sentence-transformers ranks on the sparse half alone,
    # and naming an encoder here would put a model beside backends_seen that
    # never scored a row
    dense_ran = any(k.endswith("(hybrid)") or k.endswith("(dense)") for k in backends)

    report = {
        "run": "runtime_coverage",
        "corpus": {
            "file": "JOICommands-170.csv",
            "rows": len(rows),
            "query_field": "command_eng",
            "gold_field": "gt1",
            "scored": scored,
            "skipped_rows": len(rows) - scored,
            "skipped_test_no": no_service,
            "selector_rule": (
                "a gold service is <selector>.<service>; the selector is a tag "
                "group of one or more #Tags and the device is the tag naming a "
                "catalog device, so all(#Livingroom #Light).switch_on() is "
                "Light.switch_on. Tags select instances, so (#Fan "
                "#SectorA).switch_switch and (#Fan #SectorB).switch_switch are "
                "one gold service"
            ),
        },
        "ranker": {
            "entry_point": "premap.premap_from_version_dir",
            "version_dir": "gpt_mg/version0_6",
            "model": (served or "default (e5-small)") if dense_ran else "none (sparse only)",
            "model_key": probe.get("model_key") if dense_ran else None,
            "base_model_key": probe.get("base_model_key") if dense_ran else None,
            "mode": "default (split)",
            "backends_seen": dict(backends),
            "max_total_seen": {str(k): v for k, v in budgets.items()},
            # this machine's own run, like every latency in this directory, and
            # measured around premap_from_version_dir only: the CSV read, the
            # catalog load and the model load sit outside the timer
            "latency_ms": {
                "mean": round(statistics.mean(latencies), 1),
                "p50": round(statistics.median(latencies), 1),
                "calls": len(latencies),
                "scope": ("premap_from_version_dir, warm process, model preloaded"
                          if dense_ran else
                          "premap_from_version_dir, warm process, sparse half only"),
            },
        },
        "gt1": {
            "full_coverage": full,
            "full_coverage_rate": round(full / scored, 4) if scored else None,
            "pair_hits": hits,
            "pair_total": total,
            "pair_recall": round(hits / total, 4) if total else None,
            "rows_with_a_no_match_role": withheld,
            "rows_whose_gold_exceeds_the_budget": over_budget,
        },
        # not a measurement of the retriever: arithmetic over the gold sets and
        # the budget, giving the rate an oracle ranker would reach here. The gap
        # between it and the figures above is what ranking left on the table;
        # the gap between it and 1.0 is what the budget and the catalogs cost
        "budget_ceiling": {
            "rule": ("gold pairs the runtime catalogs hold, up to top_k_per_role "
                     "per role and max_total merged"),
            "pair_hits": ceiling_hits,
            "pair_total": total,
            "pair_recall": round(ceiling_hits / total, 4) if total else None,
            "full_coverage": ceiling_full,
            "full_coverage_rate": round(ceiling_full / scored, 4) if scored else None,
        },
        "any_accepted_scenario": {
            "gold_fields": list(GOLD_FIELDS),
            "full_coverage": full_any_gt,
            "full_coverage_rate": round(full_any_gt / scored, 4) if scored else None,
        },
        "by_role": {
            role: {
                "pair_hits": role_hits[role],
                "pair_total": role_total[role],
                "pair_recall": (round(role_hits[role] / role_total[role], 4)
                                if role_total[role] else None),
            }
            for role in ("value", "function", "not_in_catalog")
        },
        # a ceiling on pair recall that no retriever can lift: these name
        # devices the runtime catalogs do not hold, so no shortlist can carry
        # them. Counted twice over, because one name can be gold in two rows.
        "gold_services_absent_from_the_runtime_catalog": {
            "distinct": len(missing_from_catalog),
            "occurrences": role_total["not_in_catalog"],
            "distinct_names": sorted(missing_from_catalog),
        },
        "by_gold_size": {
            str(n): {
                "queries": by_size[n]["queries"],
                "fully_covered": by_size[n]["fully_covered"],
                "pair_hits": by_size[n]["pair_hits"],
                "pair_total": by_size[n]["pair_total"],
                "pair_recall": round(by_size[n]["pair_hits"] / by_size[n]["pair_total"], 4),
            }
            for n in sorted(by_size)
        },
    }

    report["provenance"] = provenance(
        __file__, regenerable_without_gpu_or_key=True,
        needs="the sparse half alone reproduces a sparse run; the hybrid arms "
              "need sentence-transformers, and no API key is used")

    try:
        with open(out, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
            f.write("\n")
    except OSError as exc:
        raise SystemExit(f"cannot write {os.path.relpath(out, ROOT)}: {exc}")

    g = report["gt1"]
    lat = report["ranker"]["latency_ms"]
    print(f"scored {scored} of {len(rows)} commands   backend {list(backends)}")
    print(f"latency per call: mean {lat['mean']} ms  p50 {lat['p50']} ms "
          f"(this machine, model preloaded)")
    c = report["budget_ceiling"]
    print(f"gt1 full coverage:  {g['full_coverage']}/{scored} = {g['full_coverage_rate']}"
          f"   (oracle at this budget: {c['full_coverage']}/{scored} = {c['full_coverage_rate']})")
    print(f"gt1 pair recall:    {g['pair_hits']}/{g['pair_total']} = {g['pair_recall']}"
          f"   (oracle at this budget: {c['pair_hits']}/{c['pair_total']} = {c['pair_recall']})")
    print(f"any accepted scenario covered whole: "
          f"{full_any_gt}/{scored} = {report['any_accepted_scenario']['full_coverage_rate']}")
    print(f"rows with a no_match role: {g['rows_with_a_no_match_role']}   "
          f"rows whose gold exceeds the budget: {g['rows_whose_gold_exceeds_the_budget']}")
    for role in ("value", "function", "not_in_catalog"):
        r = report["by_role"][role]
        print(f"  {role:16s} {r['pair_hits']:>4}/{r['pair_total']:<4} recall {r['pair_recall']}")
    print(f"{'gold services':>14} {'queries':>8} {'fully covered':>14} {'pair recall':>12}")
    for n, b in report["by_gold_size"].items():
        print(f"{n:>14} {b['queries']:>8} {b['fully_covered']:>14} {b['pair_recall']:>12}")
    print(f"written: {os.path.relpath(out, ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
