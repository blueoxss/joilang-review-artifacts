# joilang-review-artifacts

Anonymized artifacts for service-level IoT automation experiments.

This repository translates natural-language IoT instructions into **JOILang** code
using an LLM. Given a Korean (or English) command, the model returns a JOILang
scenario as JSON: `{ "name", "cron", "period", "code" }`.

## Contents

| Path | Description |
| --- | --- |
| `gpt_mg/run.py` | Generation entry point (CLI). |
| `gpt_mg/version0_6/` | One model version: `config_loader.py`, prompt blocks (`*.md`), service lists (`service_list_*.json`), and `model_config.json`. |
| `JOICommands-170.csv` | Evaluation command set (NL command, connected devices, options, ground-truth JOILang). |
| `requirements.txt` | Python dependencies. |

## Requirements

- Python 3.11+ (tested on 3.12)
- Dependencies:
  ```bash
  pip install -r requirements.txt
  ```
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

If no key is found, `run.py` exits with `RuntimeError: OpenAI API key is not configured`.

## Usage (`run.py`)

Run from the `gpt_mg/` directory (so the version package resolves correctly):

```bash
cd gpt_mg
```

### 1. Generate JOILang for a single command

Pass one quoted natural-language command:

```bash
python run.py "If the light turns on, close the curtain."
python run.py "불이 켜지면 커튼을 닫아 줘."
```

The generated JOILang scenario is printed to stdout and appended to
`output_each_command.csv` (columns: `sentence, model, generated_code`).
A running log is also appended to `sentence_best_code_log.csv`.

### 2. Generate for the built-in example batch

Runs the bundled set of example Korean commands one by one:

```bash
python run.py dataset
```

Results are appended to `output_each_command.csv`.

### Selecting a model version

The default version is `version0_6`. To use a different version, add a sibling
directory under `gpt_mg/` that contains its own `config_loader.py` +
`model_config.json` + prompt assets, then edit `model_default` at the top of
`run.py` (or call `joilang_each_command(sentence, model="<version>")`).

## Outputs

| File | Content |
| --- | --- |
| `output_each_command.csv` | One row per generated command: `sentence, model, generated_code`. |
| `sentence_best_code_log.csv` | Append-only log of `sentence, best_code`. |

Both files are written next to `run.py` and are appended to across runs.

## Notes and limitations

- This is an **anonymized review artifact**. Two additional code paths in
  `run.py` depend on assets that are intentionally **not shipped** here:
  - the interactive candidate-selection mode (`python run.py <version> "<command>"`)
    requires a `version0_6_reconverted/` package for Korean back-translation;
  - the batch-benchmark mode (`python run.py`, no arguments) requires a
    `datasets/` directory (`things.json`, `final_output_*.csv`).

  Use the single-command and `dataset` modes above for reproduction.
- Model output is non-deterministic; `temperature` is set to `0.3` in
  `model_config.json`.
