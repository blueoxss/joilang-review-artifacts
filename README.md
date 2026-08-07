# joilang-review-artifacts

Artifacts for service-level IoT automation experiments.

This repository translates natural-language IoT instructions into **JOILang** code
using an LLM. Given a Korean (or English) command, the model returns a JOILang
scenario as JSON: `{ "name", "cron", "period", "code" }`.

It covers two things:

1. **NL → JOILang generation** with Service Pre-mapping (`gpt_mg/run.py --premap`);
2. **the retrieval stage behind pre-mapping** — how it is built, trained and
   measured — in `tutorials/premapping_finetune_tutorial.ipynb`, committed with
   its executed outputs.

## Contents

| Path | Description |
| --- | --- |
| `gpt_mg/run.py` | Generation entry point (CLI). |
| `gpt_mg/premap.py` | Service Pre-mapping: hybrid retrieval shortlist for the prompt. |
| `gpt_mg/version0_6/` | The model version: `config_loader.py`, the prompt blocks it loads (`*.md`), the service catalogs (`service_list_ver1.5.4_{value,function}.json`), and `model_config.json`. |
| `tutorials/premapping_finetune_tutorial.ipynb` | The retrieval stage end to end: documents → query pools → fine-tuning → evaluation → pre-mapping → generation. Shipped with executed outputs. |
| `premapping/` | The two inputs that notebook cannot derive from the catalog (ver1.1.7 commands + their role audit), the training pairs a run writes out (`data/integrated/`, named by path in every `results/` file's `provenance.pair_files`), `runtime_coverage.py` measuring the shipped shortlist against the evaluation set, and `results/` with the measured numbers — see `premapping/README.md`. |
| `JOICommands-170.csv` | The 170-command evaluation set the generation path is measured against: `command_kor` / `command_eng`, the `connected_devices` and `options` a row needs, and three accepted ground-truth scenarios per command (`gt1`–`gt3`). Three things read it: the "Run a command from the evaluation set" usage below, which feeds one row to the CLI; `premapping/runtime_coverage.py`, which scores every row's shortlist against its answer; and `premapping/premap_ab.py`, which sends a seeded draw of rows through generation twice, with pre-mapping and without. |
| `requirements.txt` | Runtime dependencies (generation + pre-mapping). |
| `requirements-full.txt` | Full environment for the notebook (torch, sentence-transformers, datasets, jupyter). |

## Requirements

- Python 3.10+ (generation is tested on 3.12; the notebook's committed run used 3.10)
- Dependencies:
  ```bash
  pip install -r requirements.txt      # generation + the sparse half of pre-mapping
  pip install sentence-transformers    # optional: adds the dense half (see below)
  ```
  `requirements.txt` leaves `sentence-transformers` commented out, so out of the
  box pre-mapping ranks on its sparse half alone and says so in every run's
  metadata. The second line is what enables the `e5-small` encoder the section
  below describes.
- An OpenAI API key (the default model is `gpt-4.1-mini`, set in
  `gpt_mg/version0_6/model_config.json`).

## API key setup

Never hardcode a key in the source. Provide it via the environment, either by
exporting it or by creating a `.env` file (loaded automatically via
`python-dotenv`). The first of the following that is set is used:

```
OPENAI_API_KEY_PROJ_BENCH
JOI_EVAL_OPENAI_API_KEY
JOI_V15_OPENAI_API_KEY
OPENAI_API_KEY
```

Example:

```bash
export OPENAI_API_KEY="sk-..."
# or: create gpt_mg/.env with  OPENAI_API_KEY=sk-...
```

If no key is found, `run.py` refuses in one line —
`OpenAI API key is not configured; set OPENAI_API_KEY (see README)`, exit 1, no traceback.
A key the API rejects stops the run the same way — `OpenAI rejected the API key: …`,
exit 1 — rather than logging a row for a command that never reached the model.

## Usage (`run.py`)

Run from the `gpt_mg/` directory (so the version package resolves correctly):

```bash
cd gpt_mg
```

### 1. Generate JOILang for a single command

```bash
python run.py "If the light turns on, close the curtain."
python run.py "불이 켜지면 커튼을 닫아 줘."
```

The generated JOILang scenario is printed to stdout (after the `Output:: >>`
marker) and appended to `output_each_command.csv` (columns: `sentence, model,
generated_code`). Set `JOI_DEBUG=1` to add, on stdout, the resolved version
module, the raw response object the API returned, and the run's own log —
timings, token counts and, under `--premap`, the pre-mapping metadata. The
request payload is not printed, and the CSV is the same either way. Both run
logs are appended to when they already exist, so both headers are read before
the first request is sent: one holding different columns is refused in one line
naming the file, rather than after a generation has been paid for.

### 2. Generate with Service Pre-mapping (`--premap`)

```bash
python run.py --premap version0_6 "sound the alarm siren once when the door closes"
```

Both positional arguments are required, and the command must be quoted so it
stays one argument. Given the command alone, `run.py` takes its one-argument
branch and never sees the version directory; given the version directory alone,
it names the missing command on stderr and falls through to the batch-benchmark
mode for that version; given neither, it falls through to
the batch-benchmark mode for `version0_6` (see the limitations below). Any other
argument count is refused with a usage message.

This form (`<version> "<command>"`) prints the scenario and then opens an
interactive review loop — `y` saves the candidate to `sentence_best_code_log.csv`,
`n` quits, Enter shows the next candidate, any other text is appended as an extra
requirement and regenerates. With no terminal attached (a pipe, `< /dev/null`, a
CI job) the loop is skipped: the scenario is printed and logged, and the command
exits 0. Example output for the command above — one representative run, not a
fixture: `temperature` is `0.3`, so the scenario a rerun prints will differ:

```json
{"name": "문닫힘알람사이렌", "cron": "", "period": -1,
 "code": "wait until ((#DoorLock).doorControl_door == \"closed\")\n(#Alarm).alarm_siren()"}
```

A scenario that comes out right does not show that pre-mapping put what it needed
in the prompt. The shortlist is not the prompt's only source of service names —
pre-mapping replaces the two catalogs and nothing else, so the grammar manual and
the caution block are sent whole and write services into their own examples — and
the model supplies more from its own training, so a command whose service the
shortlist cut can still be answered correctly, and a generated scenario read on
its own says nothing either way about retrieval. What the run does print is
whether the shortlist carried a service at all: the services the `code` names,
against the entries on the `Premap ::` line above it. Row `345` below is that
case in one row, and how often it happens is measured under
**Service Pre-mapping**.

### 3. Built-in example batch

```bash
python run.py dataset
```

Runs the bundled Korean example commands one by one; results are appended to
`output_each_command.csv`.

### 4. Run a command from the evaluation set

`JOICommands-170.csv` (repository root) is the command set the generation path is
measured against — 170 commands in Korean and English, each with three accepted
ground-truth scenarios. No module on the generation path imports it — one row is
fed to the CLI above by its `test_no`, `premapping/runtime_coverage.py` reads
every row to score the shortlists, and `premapping/premap_ab.py` draws rows from
it to run generation with pre-mapping and without:

```bash
cd gpt_mg
python run.py --premap version0_6 "$(python3 -c 'import csv, sys; print(next(r["command_eng"] for r in csv.DictReader(open("../JOICommands-170.csv", encoding="utf-8-sig")) if r["test_no"] == sys.argv[1]))' 345)"
```

Use `command_kor` for the Korean form of the same row. Comparing the printed
scenario against that row's `gt1`–`gt3` is left to the reader: no scoring code
ships here.

Row `345` is here for the shape pre-mapping is built for — a command naming one
device for its condition and another for its action — and was chosen for that
shape rather than for how it turns out. The shortlist leads with the fan's `switch_on`
and carries more of the fan with it. The temperature reading the row's `gt1`
uses, `TemperatureSensor.temperatureMeasurement_temperature`, is not on it: that
device comes back as `temperatureMeasurement_temperatureRange` instead, beside
`AirConditioner.airConditionerMode_targetTemperature`, on either half of the mix.
Installing the dense half changes which near-misses fill the rest of the list —
`WeatherProvider.weatherProvider_temperatureWeather` with it,
`Clock.clock_second` in second place without it — and not that outcome. Run the
command and the scenario still reads
`(#TemperatureSensor).temperatureMeasurement_temperature`, matching `gt1`, out of
a prompt whose `[Device and Service Mapping]` block no longer holds that service.
Nothing about that turn needs the model's memory to explain it: pre-mapping swaps
the catalogs and leaves the rest of the prompt alone, and the grammar manual
(`gpt_mg/version0_6/grammar_ver1.5.10.md`) writes that exact call in its own
examples while the caution block (`caution_prompt_8.md`) names it as the preferred
temperature service. The call was in front of the model either way, so this row
is not evidence about retrieval even in principle. What the
`code`-against-`Premap ::` check reads is narrower and is the thing worth
reading: whether the shortlist carried the service the answer used.

Row `59` was picked for the opposite shape, a command with nothing to rank at
all. It asks for the remainder of 10 divided by 3 and names no device, and
pre-mapping still returns a full twelve-entry shortlist with `retrieval_signal`
`ok`, led by services the command has nothing to do with — and which ones those
are depends on which half of the mix ran. Nothing has gone wrong there that the metadata hides:
it is the min-max property described under **Service Pre-mapping** below, where
the scores say how far the shortlist's entries separated from each other and
never how well any of them matches the command. Row `59` is worth running for
that reason, as the case that shows what a confident `top_score` does not claim.

## Service Pre-mapping (`--premap`)

`gpt_mg/premap.py` runs hybrid retrieval — `intfloat/e5-small + BM25` with a
**VALUE/FUNCTION split** index and dense weight `w = 0.6`, the remaining `0.4` going to a
sparse half scored as `0.7 · BM25 + 0.3 · token overlap`. That is the configured
mix; the dense half needs `sentence-transformers`, which `requirements.txt`
leaves commented out, so a run after the documented install ranks on the sparse
half alone until `pip install sentence-transformers` adds the encoder. Every run
names both: `configured_search` is the mix asked for, `effective_search` the one
that ran (`hybrid`, `dense`, `sparse`, or `none` where every index was empty and
nothing was scored at all), and `backend` follows the latter. The VALUE and FUNCTION catalogs
are searched separately, the top-6 candidates of each role are merged and capped
at 12 services, and the shortlist replaces the full catalogs in the
`[Device and Service Mapping]` block of the generation prompt (system-prompt
size drops by roughly 65 % — the tutorial cell that calls `load_version_config`
twice, with and without the shortlist, prints the measured
`173,204 chars -> 60,794 chars (64.9% smaller)` for the command it uses). A role
the shortlist selected nothing for renders an empty block rather than the whole
catalog.

How much of what a command needs survives that cut is measured rather than
assumed. `premapping/runtime_coverage.py` runs this component over
`JOICommands-170.csv` and scores each shortlist against the services that row's
accepted scenario names: of the 169 rows whose `gt1` names any service, 313 of
500 named services are still in the prompt after the cut, and 84 rows — under
half — keep every service their `gt1` uses
(`premapping/results/runtime_coverage.json`). Counting a row as covered once any
one of its three accepted scenarios survives whole raises that to 89. Those
figures belong to one retriever, the file's `backends_seen` naming it for every
row: `e5-small+bm25(split)`, the hybrid path, which the optional
`pip install sentence-transformers` above is what enables. A run on the sparse
half alone measures a different retriever, so the script records the backend and
refuses to write such a run over a stored hybrid one — it stores it beside it
instead. Under an interpreter with no working `sentence-transformers`, `--label`
writes that run to its own file, and over the same commands, the same budget and
the same gold it keeps 234 of the 500 services and 50 of the 169 rows
(`premapping/results/runtime_coverage_sparse.json`, `backends_seen`
`bm25(split)`). The second pair is what an install that stops at
`requirements.txt` measures; the distance between the two pairs is what the
optional line buys. The budget is not what
decides the shortfall on either: a ranker that could read the answer would keep
486 of the 500 under the same six-per-role, twelve-merged budget and 161 of the
169 rows whole (`budget_ceiling` in `runtime_coverage.json`, arithmetic over the
answers rather than a retrieval run), so most of the distance between 313 and 500
is ranking and not room. One row of the 169 does ask for more than twelve services, and ten of the
named services are absent from the runtime catalogs altogether, so no ranking
could have returned them. Pre-mapping is a narrowing, and those figures are the
size of it, to be read against the prompt reduction above that buys them. They
are the runtime's own and are not the retrieval figures in
`premapping/README.md`, which its **Harness and runtime** section separates.

A shortlist is only used as one where the scores chose it. When a role's scores
separate nothing — every entry scores alike, so the order that comes back is the
catalog's own — that role is reported in `logs["premap"]["no_match"]` and keeps
the **full catalog** in the prompt instead of an arbitrary twelfth of it, and
`run.py` says so on the line that prints the shortlist, which lists only the
entries the prompt actually carried: a role whose scores separated nothing has
its candidates withheld from that line and counted there
(`[unused - no ranking signal for function, value; full catalog kept]
(12 of 12 candidates withheld)`), so a shortlist that was not applied is never
displayed as though it had been. Those logs are a `run.py` run's own
copy of the pre-mapping metadata: `JOI_DEBUG=1` prints them, and `premap.py`
prints those same fields for a command without generating anything — all of them
but `applied`, which only a run that builds a prompt can answer and which
`run.py` adds itself. `logs["premap"]["applied"]` answers what
the prompt was built with: it reads `false` when no role's shortlist reached it —
because every role was unranked, or because the version's loader takes no
shortlist at all. A role scores nothing just as surely when neither half of the
mix can read the command: the catalogs' text and the default `e5-small` are
English, so a Korean command usually has no lexical overlap to match and an
embedding with no training signal behind it, and that role is reported in
`no_match` too rather than ranked on what the dense half separates. Usually and
not always — a Latin token the catalogs also carry is enough for the sparse
half to rank that role, as below. Such a shortlist is applied in
full and no role reaches `no_match`, so `run.py` marks it on the line that
prints it rather than only in the notes below: the shortlist is followed by
`(thin signal - the command's script is not the encoder's)`, and where a role
was also withheld the marker joins that role's parenthetical instead of adding
a second one. An operator reads the entries and the reason to distrust them in
the same place.
`retrieval_signal` (`ok` / `partial` / `none`), `top_score` and
`selected_scores` sit next to `selected_count`, so a shortlist the scores ranked
and one nothing ranked are told apart from the log alone. Those scores are
min-max normalized inside each role's index, so they say how far the shortlist's
entries separated from each other, not how well any of them matches the command:
`top_score` reads `1.0` only where one entry tops every term of the mix, and
lower where the terms rank different entries first, since each is normalized
before they are blended. It reads `0.0` where every candidate scored alike,
which min-max maps to zeros, and where the shortlist is empty because no
candidate was scored at all — an empty catalog leaves `selected_scores` empty
and there is no maximum to report. A role that ranked nothing can still carry a
non-zero `top_score`: where the sparse half matched no token but the dense half
still separated on noise, the blend tops out at the dense weight — a
punctuation-only command and a Korean one on the English encoder both read `0.6`
while `retrieval_signal` reads `none`. `retrieval_signal` and `no_match`, not
`top_score`, are what say whether a ranking happened.
`dense_weight` and `sparse_mix` say how the halves were combined — `0.6` and
`{"bm25": 0.7, "overlap": 0.3}` on the default hybrid path. Both are blend
weights, not a record of which half ran (`effective_search` is that), so each
reads `null` wherever no blend applied: a sparse-only run names the sparse split
alone, and a dense-only model names neither, its ranking being one cosine score
with nothing to weigh it against. Degradation is tracked per role, because it can
strike one role and leave the other blended: `effective_search` names the
strongest search any role's ranking came from, and a role that lost its dense
half while another kept it is named in `notes` (`dense_unavailable:<role>`) and
in a `role_search` map reported only where the roles differ — where they agree,
`effective_search` already says it. A run whose VALUE ranking really was
`0.6 · dense + 0.4 · sparse` therefore keeps `dense_weight` `0.6` and names the
role that missed out, instead of relabelling both roles sparse and reporting no
weight for scores the encoder produced. A local checkpoint carries one field a shipped
model row does not: `base_model_key`, the released encoder row its search mode
and query-prefix convention were resolved from, read off the checkpoint's
directory name. The fine-tuned checkpoint below reports `model_key`
`e5-split-v7` and `base_model_key` `e5-small`. `notes`
records the soft conditions behind such a run: an empty query (`empty_query`), a
query with no word token (`no_query_tokens`), a role whose sparse half matched
nothing (`no_lexical_match:<role>`), a role whose whole shortlist came from one
device (`single_device_shortlist:<role>:<device>`), a role scored without the
dense half while another role kept it (`dense_unavailable:<role>`), a query
longer than the encoder's context (`dense_query_truncated:<tokens>`, the count
the encoder had to truncate), a non-Latin command the configured
encoder does not cover (`non_latin_query_on_english_encoder:e5-small`, or
`non_latin_query_on_english_catalog` where no encoder ran at all), a catalog
holding the same `(role, device, service)` twice
(`duplicate_entries_dropped:<n>`, the repeats the merge collapsed — they cost the
shortlist that many slots, so `selected_count` comes back under the budget for a
reason that is not the ranking), and a role the ranking left out of the shortlist
altogether (`empty_role_shortlist:<role>`). That last one is the `nosplit` index's
to produce: one ranking over both roles is free to spend every slot on one of
them, where `split` gives each role its own budget. The prompt then carries an
**empty** block for that role — an empty shortlist is still a shortlist, and only
a role under `no_match` keeps its full catalog — so the note is what says a whole
role's services left the prompt while `no_match` is empty, `retrieval_signal`
reads `ok` and `selected_count` reads the full budget.
Those nine are the whole vocabulary. A dense half
that ran but not on the device it asked for is reported separately, in
`dense_note` (`dense_model_cpu_fallback:<Exc>` for an encoder that could not be
**loaded** on a full GPU and was loaded on the CPU instead). A GPU that fills
later, while the catalog or the query is being encoded, is not retried on the
CPU: that role falls back to the sparse scores for that call, and the reason
lands in `fallback` rather than in `dense_note`.
`no_lexical_match` says the sparse half matched nothing and the dense half
carried that role's ranking alone; where the script note says the encoder does
not cover the command either, that ranking is noise, which is why most Korean
commands with an encoder loaded carry both notes and both roles under
`no_match`. Not all of them do. A Korean command whose text also carries a Latin
token the catalogs themselves carry gives the sparse half something to match —
a device or service name, not any Latin token — and the role that matched is
then ranked like any other: its shortlist
replaces that role's full catalog in the prompt, on a signal that thin.
`run.py`
prints the notes whether or not they cost a role its shortlist. Generation
itself is unaffected where both roles are withheld: the model reads Korean, and
a command whose roles are reported unranked is sent with the full catalogs,
exactly as a run without `--premap`. Where one role was ranked and the other was
not, only the withheld role keeps its catalog, so what pre-mapping did to a
Korean command is a per-run question rather than a general one:
`retrieval_signal` and `no_match`, on that run's own `Premap ::` line, are the
authority for it.

The default loads the **released** `intfloat/e5-small`. The fine-tuned
checkpoint behind the 0.9830 row in `premapping/README.md` is a different
encoder and is reachable only through `--premap-model`. What serving it does to
the coverage figures above is measured rather than left to the reader:
`premapping/runtime_coverage.py --model` scores any arm on the same commands, and
that section reports the checkpoint's own run beside the default's.

Every benchmarked retriever is selectable, and the shortlist entries are always
routed by role into the prompt's `[service_list_value]` / `[service_list_function]`
blocks:

```bash
# other retrievers / modes
python run.py --premap-model mxbai-embed-xsmall-v1 version0_6 "..."
python run.py --premap-model multilingual-e5-small version0_6 "..."   # dense-only
python run.py --premap-model bge-m3 version0_6 "..."                  # dense-only
python run.py --premap-model ../premapping/models/e5-split-v7 version0_6 "..."  # fine-tuned ckpt
python run.py --premap --premap-mode nosplit version0_6 "..."

# equivalent env vars: JOI_PREMAP=1 JOI_PREMAP_MODEL=bge-m3 JOI_PREMAP_MODE=split
```

The `../premapping/models/e5-split-v7` line needs that checkpoint to exist, and
`premapping/models/` is git-ignored: a clean checkout has no checkpoints until
the notebook has been run (see `premapping/README.md`). Until then that command
still runs — the encoder fails to load, the shortlist comes off the sparse half
alone, and both the `Premap ::` lines and `fallback` say so.

Any of these arms can degrade to the sparse scores: the encoder fails to load or
to encode, the run keeps going on the sparse half alone, and `fallback` records
the reason. A `bge-m3` command whose `Premap ::` line reads `bm25(split)` rather
than `bge-m3(dense,split)` says the dense half did not run; it does not say why,
and the causes need different fixes — a contended GPU, `sentence-transformers`
not installed, or a model id that resolves to nothing all produce that same
line. `run.py` prints the reason on the next line whenever any role's ranking
came from a weaker search than the one configured. Where every role fell short
the line names the run (`Premap :: degraded to sparse (configured hybrid): …`);
where one role fell short and another kept its dense half — which leaves
`effective_search` reading `hybrid`, since it names the strongest search any
role reached — the line names the roles instead
(`Premap :: degraded for value (configured hybrid): …`), so the reason sits next
to the `notes` line that raises the question rather than waiting for a
`JOI_DEBUG` rerun. A role that reached no search at all counts as falling short
and is named the same way: a catalog file holding an empty list is never scored,
and `fallback` reads `empty_catalog:<role>` for it while that role is reported
under `no_match`. `JOI_DEBUG=1`, or `premap.py` run directly, prints the whole
`fallback` field.

Naming a retriever is itself a request for pre-mapping: `JOI_PREMAP_MODEL` or
`JOI_PREMAP_MODE` alone enables it, `JOI_PREMAP=1` is not additionally required.
A variable left exported from an earlier run therefore still applies to a later
`run.py` invocation that does not pass `--premap`.

Dense retrieval requires `sentence-transformers` (`pip install
sentence-transformers`; `requirements.txt` carries the pin commented out);
if it is not installed, retrieval degrades to sparse (BM25 + lexical) scoring,
the reason is recorded in the run logs (`logs["premap"]["fallback"]`) and the
printed backend reads `bm25(split)` rather than naming an encoder that never ran.
That reason carries the import's own error, because the package is often
installed and a library underneath it is what failed — a bare "unavailable"
would send a reader to reinstall something already there. It carries the error's
text and not the host's filesystem: `fallback` is machine-readable and is read on
other machines, so an absolute path inside it is reduced to the name at its end —
`…/nvidia/cusparse/lib/libcusparse.so.12: undefined symbol: …` is recorded as
`libcusparse.so.12: undefined symbol: …`. The rule is that no piece of the host's
filesystem travels, not a list of the shapes one takes, so the three that get
past a plain `/segment/segment` rule are cut the same way: a Windows root
(`C:\Users\…`), a UNC share (`\\server\share\…`), which carries no drive letter
for a rule keyed to one to find, and a path holding spaces
(`/home/… …/My Models/config.json`), which a rule ending a path at the first
space would leave carrying a user name. Runs of separators count as one, so a
Windows path escaped on its way into a log (`C:\\Users\\…`) is cut as well.
The broken library stays nameable,
which is the whole point of carrying the error; the directory layout of the
machine that hit it does not travel with the metadata. `model_name` follows the
same rule rather than an exception to it: a hub id, a relative path and a URL are
the caller's own text and are left whole, but a checkpoint served by an absolute
path is reported as the name at its end, so `--premap-model
/home/…/models/e5-split-v7` is recorded as `e5-split-v7`. The encoder cache still
keys on the resolved path, so `dir` and `dir/` load one model rather than two;
only the reported name is cut. `premapping/runtime_coverage.py` names a served
checkpoint relative to the repository for the same reason, so a results file
names the arm without naming the host.
The tutorial applies the same rule from the other side: its first
cell silences the two warnings whose text carries a path rather than information —
`IProgress not found`, which names the environment it ran in, and the
`DeprecationWarning` raised by importing `losses` from `sentence_transformers`,
which the kernel prefixes with its scratch file for the importing cell
(`/tmp/ipykernel_<pid>/<n>.py:2`). Both filters match by message, so nothing else
is hidden. The committed outputs predate the second filter and still show that
scratch path in the fine-tuning cell; it clears the next time the notebook is
executed. A stored output is a record of one execution and is not edited by hand,
which is why the filter is what removes the path rather than an edit to the cell
below it. The rule the `fallback` field follows is stricter still, because that
field is machine-readable and travels to other machines; a stored cell output
does not.

Two other cells are ahead of their stored outputs for the same reason, and are
listed here rather than left for a reader to trip over. The runtime pre-mapper
cell now prints `retrieval_signal` and `no_match`, the selected entries one per
line, and a Korean command whose roles are both withheld; its stored output is
the earlier device-level printout, so the prose beside it describes lines the
committed run does not show. The cell below it now also reads the measured
runtime coverage out of `premapping/results/runtime_coverage.json`. Both fill in
at the next execution of the notebook, which is the owner's to run: executing it
retrains the encoder and rewrites that run's `premap_eval_*` files under
`premapping/results/`, so it is not run to refresh a printout.
Standalone check, from the same `gpt_mg/` directory and without any API key:

```bash
python premap.py "if the door is closed, sound the alarm once"
```

It prints the metadata for that command; `python premap.py --help` prints the
arguments it takes. Quote the command: every positional argument that is not a
flag is joined with single spaces into one query, so a command the shell split
on whitespace still pre-maps as one sentence rather than only its first word.
With no argument it runs a demo
command and says so on stderr; an argument that is empty is refused
(`empty query: pass the command to pre-map`) rather than answered with the
demo command's shortlist. A mistyped option is refused whether or not a command
was also given (`unrecognized argument '--mdoe'`), so joining never quietly
absorbs one. `--mode` accepts `split` or `nosplit` and refuses
anything else, as does `run.py --premap-mode`, before either starts work.
`run.py` resolves the version directory in the same place, so an unknown one is
named (`unknown version directory 'nosuchversion': …`) before the command is
echoed rather than after it has been sent.
A version directory whose catalogs are missing, unreadable, or not shaped like a
catalog is refused by both entry points the same way — one line naming the file
or the offending row (`value catalog row 0 is str, expected an object: 'not'`),
exit 1, no traceback. A directory standing where a catalog file belongs is named
as that (`service catalog is a directory, not a file: …`) rather than reported
missing, which would send a reader to create a name they can already see.

Note: this option ports the retrieval stage's decomposition-and-scoring path,
including the rule-based condition/action clause decomposition that gives split
mode its two sub-queries; when a command has such a clause boundary, the split
is logged as `logs["premap"]["decomposed_query"]`. The rules are English, like
the catalogs: the boundary is a leading English conditional keyword (`if`,
`when`, `while`, …) followed by a comma, so a Korean command is not cut, no
`decomposed_query` is logged for it, and both roles are searched with the whole
command. The condition side must also name something beyond the keyword that
opened it, so `if, sound the alarm` is not cut either: a bare `if` searched
against the VALUE catalog would come back as a ranked shortlist of six services
with nothing behind it. An action clause that opens with notification wording
(`notify me`, `alert me`, `send an alert`, `let me know`) is not cut either,
because notification is not a device action. That test reads the opening words
only, and two ordinary commands land on it: `notify me and sound the alarm` is
not cut although its tail does name a device action, and neither is `when the
door opens, alert the light` — `alert` opens the clause as notification wording
even though the shipped FUNCTION catalog also carries it as a service verb
(`switchLevel_alert`). Such a command is searched whole in both buckets rather
than as a FUNCTION sub-query; the action is still searched for, alongside the
condition instead of on its own. Every refusal to cut widens what is searched —
both buckets get the whole command — and never narrows it. Three parts of the full pipeline are not here: the RAG-prompt-block
stage, the hard schema-constraint
mask that zeroes candidates violating a required type or signature, and the
static gate that screens the shortlist and re-runs pre-mapping with adjusted
parameters. `premap.py` scores, ranks and cuts; it never masks, screens or
retries. Nor is there an aliases field: the catalogs carry no synonyms, so each
entry is indexed under its own device and service names and its descriptors.

## Retrieval evaluation

`tutorials/premapping_finetune_tutorial.ipynb` builds the pre-mapper's retrieval
stage from the catalog files up — documents, query pools, encoder fine-tuning,
scoring — and then calls the runtime pre-mapper on a real command. It is
committed with its outputs, so the numbers are readable without a GPU;
re-running the fine-tuning cells needs CUDA and `requirements-full.txt`.

Each run also writes its totals, per-pool figures and training provenance to
`premapping/results/`, so the numbers survive independently of the notebook's
saved outputs. See `premapping/README.md` for what is measured and how.

```bash
cd tutorials
jupyter nbconvert --to notebook --execute premapping_finetune_tutorial.ipynb
JOI_FT_ALL=1 jupyter nbconvert --to notebook --execute premapping_finetune_tutorial.ipynb
```

The notebook names the kernel `joi`; both lines want that name registered on the
machine first, which `premapping/README.md` gives as one command under **Running
it**, alongside the flag that runs the notebook under a kernel already installed.

## Outputs

| File | Content |
| --- | --- |
| `gpt_mg/output_each_command.csv` | One row per generated command: `sentence, model, generated_code`. |
| `gpt_mg/sentence_best_code_log.csv` | Append-only log of `sentence, best_code`: one row per generation, plus one more when a candidate is accepted with `y`. |

Both files are created next to `run.py` on first run and appended to afterwards.

## Notes and limitations

- Two code paths in `run.py` depend on assets that are **not shipped** here:
  - the batch-benchmark mode (`python run.py`, no arguments) requires a
    `datasets/` directory (`things.json`, `final_output_*.csv`);
  - the review loop's Korean back-translation line requires a
    `<version>_reconverted/` package; without it the loop prints the original
    command as the reconverted sentence and sends no second request.

  Use the single-command, `--premap`, and `dataset` modes above.
- A request that fails for any other reason — a rate limit, a dropped
  connection — is printed as `exception in response: …` and the command is
  logged with an empty `generated_code`, so a batch keeps going; only a rejected
  key stops the run.
- Model output is non-deterministic; `temperature` is set to `0.3` in
  `model_config.json`.
