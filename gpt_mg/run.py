from dotenv import load_dotenv
from openai import OpenAI, AuthenticationError
import os
import sys
import time
import importlib
import inspect

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

load_dotenv()

api_key = (
    os.getenv("OPENAI_API_KEY_PROJ_BENCH")
    or os.getenv("JOI_EVAL_OPENAI_API_KEY")
    or os.getenv("JOI_V15_OPENAI_API_KEY")
    or os.getenv("OPENAI_API_KEY")
)

if not api_key:
    # a missing key is a configuration refusal like every other one this CLI
    # makes, not a crash: one line naming what to set, exit 1, no traceback
    raise SystemExit("OpenAI API key is not configured; set OPENAI_API_KEY (see README)")

os.environ["OPENAI_API_KEY"] = api_key
client = OpenAI(api_key=api_key)

import pandas as pd
import json, ast
from datetime import datetime
from tqdm import tqdm
import csv
import re


model_default = 'version0_6'
# Raw request/response objects are diagnostics, not the run's output: they are
# printed only when JOI_DEBUG is set, so what stdout shows for a normal run is
# the generated scenario and the pre-mapping lines that explain it.
DEBUG = os.getenv("JOI_DEBUG", "0").strip().lower() in ("1", "true", "yes")


def debug_print(*parts):
    if DEBUG:
        print(*parts)


# Service Pre-mapping (default: e5-small + BM25, VALUE/FUNCTION split, w=0.6).
# Enable with `--premap` (or JOI_PREMAP=1). Select the retriever with
# `--premap-model {e5-small|mxbai-embed-xsmall-v1|multilingual-e5-small|bge-m3|<path>}`
# and `--premap-mode {split|nosplit}` (or JOI_PREMAP_MODEL / JOI_PREMAP_MODE).
PREMAP_MODEL = os.getenv("JOI_PREMAP_MODEL") or None
PREMAP_MODE = os.getenv("JOI_PREMAP_MODE") or None
# Selecting a retriever enables pre-mapping, exactly as --premap-model /
# --premap-mode do, so neither surface can ask for a retriever and be ignored.
PREMAP_ENABLED = (
    os.getenv("JOI_PREMAP", "0").strip().lower() in ("1", "true", "yes")
    or bool(PREMAP_MODEL) or bool(PREMAP_MODE)
)
all_items = []
choice_no = 0
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

#from Evaluation.compare_soplang_ir import compare_codes
# the second run log, named once: generation writes it from inside
# generate_joi_code, and the CLI reads its header before the first request
SENTENCE_LOG_NAME = "sentence_best_code_log.csv"
SENTENCE_LOG_FIELDS = ["sentence", "best_code"]


def _run_log_path(filename):
    """A run log is written next to run.py, whatever the working directory is."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)


def _log_needs_header(path, fieldnames):
    """True when the run log still needs its header row; refuses a foreign one.

    Both logs below are appended to when they already exist, so the header on
    disk is what the appended rows are read against. A file carrying different
    columns is refused in the same one line as a log this run cannot write:
    appending would leave rows that no longer match the header above them, and
    nothing downstream could tell the two apart.
    """
    if not os.path.isfile(path):
        return True
    try:
        with open(path, "r", newline='', encoding="utf-8-sig") as f:
            header = next(csv.reader(f), None)
    except OSError as exc:
        raise SystemExit(
            f"cannot read the run log {path}: {exc.strerror}; check the file's permissions"
        ) from exc
    except UnicodeDecodeError as exc:
        raise SystemExit(
            f"the run log {path} is not valid UTF-8 ({exc}); move it aside"
        ) from exc
    if header is None:
        return True                 # an empty file is a log that never got its header
    if header != list(fieldnames):
        raise SystemExit(
            f"the run log {path} holds columns {header}, not {list(fieldnames)}; "
            "move it aside or point the run at another file"
        )
    return False


def save_sentence_and_code(sentence, best_code, filename=SENTENCE_LOG_NAME):
    """
    입력 sentence와 best_code를 한 파일에 누적으로 저장 (중복 허용)

    The header is read again here for the row this call may have to write; the
    refusal it can raise has already been raised by the caller, before the
    model was called.
    """
    filename = _run_log_path(filename)
    needs_header = _log_needs_header(filename, SENTENCE_LOG_FIELDS)
    #print(filename)
    try:
        f = open(filename, "a", newline='', encoding="utf-8-sig")
    except OSError as exc:
        # a log this run cannot write is a configuration error like the ones the
        # CLI names above, not a crash: the model call has already been paid for
        # by the time it lands, so it is refused in one line naming the file
        raise SystemExit(
            f"cannot write the run log {filename}: {exc.strerror}; check the file's permissions"
        ) from exc
    with f:
        writer = csv.DictWriter(f, fieldnames=["sentence", "best_code"])
        if needs_header:
            writer.writeheader()
        writer.writerow({"sentence": sentence, "best_code": best_code})

def _known_version_dirs():
    """The version directories shipped next to run.py, by name.

    A version is a directory holding a ``config_loader``, so that file is what
    is looked for rather than a hardcoded list that a new directory would leave
    out of the refusal below.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        names = os.listdir(here)
    except OSError:
        return []
    return sorted(n for n in names
                  if os.path.isfile(os.path.join(here, n, "config_loader.py")))


def _import_version_loader(model):
    """Import a version directory's ``config_loader``, naming a bad directory.

    The CLI resolves the version this way before the first request as well, so
    an unknown directory is refused where the other configuration arguments
    are - not after the command has been echoed and the model called.
    """
    model_path = f"{model}.config_loader"
    try:
        return importlib.import_module(model_path)
    except ModuleNotFoundError as exc:
        # a version directory that does not exist is a configuration error, and
        # is reported as one here rather than as an import traceback; a missing
        # import *inside* the loader is a different failure and stays raw
        if exc.name and not model_path.startswith(exc.name):
            raise
        # naming the cause alone leaves the reader guessing at the spelling, so
        # the names this CLI does accept are listed with it
        known = _known_version_dirs()
        hint = (f"; available here: {', '.join(known)}" if known else
                "; no directory next to run.py holds a config_loader.py")
        raise SystemExit(
            f"unknown version directory {model!r}: no module {model_path!r}{hint}"
        ) from exc


def generate_joi_code(sentence: str, model: str, connected_devices: dict, current_time: str = None, other_params: dict = None) -> dict:
    # 1. 메시지 구성
    start = time.perf_counter()
    # the prompt carries a real timestamp even when the caller has none, so a
    # single-command run never sends the literal "Current Time: None"
    current_time = current_time or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    #from version0_1.config_loader import load_version_config
    #model_path = f"gpt_mg.{model}.config_loader"
    debug_print("Model Path :: ", f"{model}.config_loader")
    config_loader_module = _import_version_loader(model)
    load_version_config = getattr(config_loader_module, 'load_version_config')
    path_tmp = '.'#os.path.join("gpt_mg",version_path,"config_loader")

    premap_services, premap_meta = None, None
    if PREMAP_ENABLED:
        try:
            import premap as premap_mod
        except ImportError:
            from gpt_mg import premap as premap_mod
        version_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), model.split('.')[-1])
        try:
            premap_services, premap_meta = premap_mod.premap_from_version_dir(
                sentence, version_dir, model=PREMAP_MODEL, mode=PREMAP_MODE)
        except (OSError, ValueError, TypeError) as exc:
            # a version directory whose catalogs are missing, unreadable or not
            # shaped like a catalog is a configuration error, and is named as one
            # rather than left to reach the operator as a traceback out of the
            # retrieval stack. The three are one class from where the operator
            # stands - the catalog file is wrong - and premap.py's own message
            # already names the offending row, so the traceback adds only
            # internals. premap.py refuses the same input the same way.
            raise SystemExit(str(exc)) from exc
        # What reaches the prompt is decided before anything is printed, so the
        # first line an operator reads is the one that says it. A shortlist can
        # be withheld for two reasons - a role whose scores separated nothing
        # (config_loader keeps the full catalog for it), or a version whose
        # loader predates the shortlist entirely - and a line that listed the
        # candidates either way would show an operator a shortlist the prompt
        # never carried.
        supported = "premap_services" in inspect.signature(load_version_config).parameters
        unranked = list(premap_meta.get("no_match") or ())
        shortlist = list(premap_meta.get("selected") or ())
        if not supported:
            applied_entries, withheld = [], f"{model_path}.load_version_config takes no premap_services"
            premap_services = None
        else:
            # an entry is named "<role>:<device>.<service>", and only the roles
            # absent from no_match are the ones config_loader substitutes
            applied_entries = [e for e in shortlist if e.split(":", 1)[0] not in unranked]
            withheld = ("no ranking signal for " + ", ".join(unranked)) if unranked else ""
        # "n of m" rather than a bare count: a role that ranked nothing has no
        # candidates to withhold, and "0 of 6" says that where "0 withheld"
        # would read as though nothing had been held back at all
        count = f"{len(shortlist) - len(applied_entries)} of {len(shortlist)} candidates withheld"
        # the line already names the roles it withheld, so its own grammar
        # follows how many there were rather than reading "those roles" over one
        those_roles = "those roles" if len(unranked) > 1 else "that role"
        # a role can rank on very little and still rank. Where the command's
        # script is not the encoder's, the dense half has no training signal to
        # contribute and the sparse half carries the ranking alone - often off a
        # single incidental Latin token - yet no role lands in no_match and the
        # shortlist is applied in full. The notes line below reports it, but an
        # operator reads the shortlist on THIS line and reaches that one only
        # afterwards, so the qualification is put where the shortlist is rather
        # than under it. The line stays unchanged where no such note was raised.
        thin = ""
        if any(str(n).startswith("non_latin_query_on_english_encoder")
               for n in (premap_meta.get("notes") or ())):
            thin = "thin signal - the command's script is not the encoder's"
        if not withheld:
            if thin:
                print("Premap ::", premap_meta.get("backend"), applied_entries,
                      f"({thin})")
            else:
                print("Premap ::", premap_meta.get("backend"), applied_entries)
        elif applied_entries:
            detail = f"{count} - {withheld}; full catalog kept for {those_roles}"
            if thin:
                detail += f"; {thin}"
            print("Premap ::", premap_meta.get("backend"), applied_entries,
                  f"({detail})")
        else:
            print("Premap ::", premap_meta.get("backend"),
                  f"[unused - {withheld}; full catalog kept]", f"({count})")
        # a backend naming no encoder says the dense half did not run; it does
        # not say why, and the causes need different fixes (a contended GPU, an
        # uninstalled package, a model id that resolves to nothing). The reason
        # is printed next to the line that raised the question rather than left
        # for a JOI_DEBUG rerun.
        #
        # effective_search names the STRONGEST search any role's ranking came
        # from, so degradation that strikes one role and leaves the other
        # blended does not move it. Testing it alone stayed silent in exactly
        # the case worth reading about: a backend string advertising an encoder
        # half the shortlist never used, a notes line raising the question, and
        # the reason for it reachable only by re-running under JOI_DEBUG. Every
        # role that fell short of the configured search is named instead -
        # including a role that reached no search at all, an empty catalog
        # being a shortfall like any other.
        configured = premap_meta.get("configured_search")
        effective = premap_meta.get("effective_search")
        # reported only where the roles differ; where they agree the run-level
        # comparison above already covers them
        role_search = premap_meta.get("role_search") or {}
        short = sorted(r for r, s in role_search.items() if s != configured)
        if effective != configured or short or premap_meta.get("fallback"):
            reason = premap_meta.get("fallback") or "no reason recorded"
            if effective != configured:
                scope = f"degraded to {effective}"
            elif short:
                scope = "degraded for " + ", ".join(short)
            else:
                # a reason recorded while every role reached the configured
                # search is not reachable today; printing it beats dropping it
                scope = "degraded"
            print(f"Premap :: {scope} (configured {configured}):", reason)
        # the soft conditions are printed whether or not they cost a role its
        # shortlist: a run that kept every shortlist can still have been scored
        # on half the mix, and the operator sees that here
        if premap_meta.get("notes"):
            print("Premap :: notes:", premap_meta["notes"])
        # "applied" answers what the prompt was built with, not what the loader
        # would accept. premap already reports whether any role's shortlist
        # reaches the prompt; the only thing it cannot know is whether THIS
        # version's loader takes one at all, which is what narrows it here.
        premap_meta["applied"] = supported and bool(premap_meta.get("applied"))
    premap_kwargs = {"premap_services": premap_services} if (premap_meta or {}).get("applied") else {}

    start_infer = time.perf_counter()
    try:
        config, model_input = load_version_config(f"Current Time: {current_time}\n\nGenerate JOI Lang code for Natural Language: {sentence}", \
                                            connected_devices, other_params, path_tmp, **premap_kwargs)
    except (OSError, ValueError, TypeError) as exc:
        # same refusal as the pre-mapping call above, for the same class of
        # input: a version directory whose prompt assets are missing, unreadable
        # or not the JSON they claim to be. The loader's message already names
        # the offending file, so the traceback adds only internals - and this
        # path had been the one place the CLI let that class through raw.
        raise SystemExit(str(exc)) from exc
    logs={}
    end_infer = time.perf_counter()
    logs["inference_time"] = f"{end_infer - start_infer:.4f} seconds"
    if premap_meta is not None:
        logs["premap"] = premap_meta

    # 2. 모델 호출
    response = {}
    global all_items  # 전체 item을 누적할 리스트
    global choice_no
    all_items = []
    choice_no = 0
    try:
        response = client.chat.completions.create(**model_input)
        debug_print("Response:: ", response)

        try:
            generated_code = response.choices[0].message.content
            print("Response Content :: ", generated_code)
            if isinstance(generated_code, dict):
                generated_code = generated_code.get("choices", "")
            # ✅ Case 1: Triple backtick with json block
            if generated_code.strip().startswith("```json"):
                # Remove ```json and final ```
                cleaned = re.sub(r"^```json", "", generated_code)
                cleaned = re.sub(r"```$", "", cleaned)

            # ✅ Case 2: Pipe-prefixed format (e.g., "json|{ ... }")
            elif generated_code.strip().startswith("json|"):
                # Remove only the "json|" part
                cleaned = generated_code.strip()[len("json|"):].strip()

            # ✅ Case 3: Already-clean content
            else:
                cleaned = generated_code.strip()

            # ✅ 문자열 → dict로 파싱
            # The model sometimes writes real newlines inside the "code" string
            # instead of \n escapes. strict=False accepts literal control
            # characters inside JSON strings, so the text is parsed as it stands
            # and no regex repairs it first - a quote-terminated one stops at the
            # first escaped \" inside the code and leaves every newline after it.
            parsed = json.loads(cleaned, strict=False)
            all_items.append(parsed)
        except Exception as e:
            # the model returned something this branch could not read as a
            # scenario. The operator line carries the same classification as the
            # log, so the parser's own message ("Expecting value: line 1 column
            # 1 (char 0)") arrives named rather than as a bare exception that
            # reads like a crash in the retrieval or prompt code above it.
            logs["error"] = f"response_parse_failed: {e}"
            print("## <<2. generate_joi_code>> \n", logs["error"])
    except AuthenticationError as e:
        # a key the API rejects is a configuration error, exactly like a missing
        # one, and is raised as one here: logging it as a sentence that generated
        # nothing would write an empty row and exit 0 on a run that never reached
        # the model
        raise SystemExit(f"OpenAI rejected the API key: {e}")
    except Exception as e:
        print("exception in response:", e)
        logs["error"] = f"model_call_failed: {e}"
    # 후보 목록이 곧 결과: 파싱에 실패하면 빈 문자열
    generated_code = all_items if all_items else ""

    end = time.perf_counter()
    logs["response_time"] = f"{end - start:.4f} seconds"
    print("Response Time: ", logs["response_time"])
    
    best_code = generated_code
    logs["translated_sentence"] = ""
    logs["mapped_devices"] = ""
    logs["best_code"] = best_code
    save_sentence_and_code(sentence, best_code)
    try:
        # dict 형태일 경우
        if isinstance(response, dict):
            usage = response.get("usage", {})
            logs["prompt_tokens"] = usage.get("prompt_tokens", "")
            logs["completion_tokens"] = usage.get("completion_tokens", "")
            logs["total_tokens"] = usage.get("total_tokens", "")
        else:
            # 객체일 경우 (예: OpenAI 응답 객체)
            usage = getattr(response, "usage", None)
            logs["prompt_tokens"] = getattr(usage, "prompt_tokens", "") if usage else ""
            logs["completion_tokens"] = getattr(usage, "completion_tokens", "") if usage else ""
            logs["total_tokens"] = getattr(usage, "total_tokens", "") if usage else ""
    except Exception as e:
        # 어떤 예외가 발생해도 빈값으로
        logs["prompt_tokens"] = ""
        logs["completion_tokens"] = ""
        logs["total_tokens"] = ""
    return {
        "code": 
            best_code
        ,
        "log": logs
    }

def merge_duplicate_blocks(generated_code):
    import re
    import json
    # 리스트라면 첫 번째 요소를 꺼냄
    if isinstance(generated_code, list):
        if generated_code:
            generated_code = generated_code[0]
        else:
            generated_code = ""

    code = generated_code#.strip()
    # 맨 앞/뒤에 ``` 있으면 제거
    if code.startswith("```"):
        code = code[3:]
    if code.endswith("```"):
        code = code[:-3]
    #code = code.strip()

    # 정확히 "key" 형식만 매칭하도록 수정
    name_pattern = re.compile(r'"name"\s*:\s*"([^"]*)"', re.DOTALL)
    cron_pattern = re.compile(r'"cron"\s*:\s*"([^"]*)"', re.DOTALL)
    period_pattern = re.compile(r'"period"\s*:\s*(-?\d+(?:\.\d+)?)', re.DOTALL)
    code_pattern = re.compile(r'"code"\s*:\s*"((?:\\.|[^"\\])*)"', re.DOTALL)
    

    name_fields = [m.group(1).strip().strip(',').strip() for m in name_pattern.finditer(code)]
    cron_fields = [m.group(1).strip().strip(',').strip() for m in cron_pattern.finditer(code)]
    period_fields = [m.group(1).strip().strip(',').strip() for m in period_pattern.finditer(code)]
    #code_fields = [m.group(1).strip().strip(',').strip('"') for m in code_pattern.finditer(code)]
    # code 필드 디코딩 적용 (이스케이프 문자 복원)
    code_fields = [m.group(1).encode().decode("unicode_escape") for m in code_pattern.finditer(code)]

    n_blocks = min(len(name_fields), len(cron_fields), len(period_fields), len(code_fields))
    if (n_blocks<2):
        return generated_code, 0
    
    blocks = []
    for i in range(n_blocks):
        if not (name_fields[i] and period_fields[i] and code_fields[i]):
            continue  # 값이 하나라도 없으면 건너뜀

        try:
            period_val = int(period_fields[i])
        except Exception:
            period_val = period_fields[i]
        blocks.append({
            "name": name_fields[i],
            "cron": cron_fields[i],
            "period": period_val,
            "code": code_fields[i]
        })
    # cron, period가 같으면 code를 합침
    merged = {}
    for b in blocks:
        print(f"\nBlock :: {b} \n")
        key = (b["cron"], b["period"])
        b["code"] = b["code"].replace('---', '')  # --- 구분자 제거, 이스케이프 문자 복원
        if key in merged:
            merged[key]["code"] += "\n"+ b["code"]   # --- 구분자 제거
        else:
            merged[key] = b
    merged_list = list(merged.values())
    #merged_list = [b for b in merged_list if all(str(b.get(k, "")).strip() for k in ("name", "cron", "period", "code"))]
    
    # 코드 필드 내부 \\n → \n, \\" → ", 등 이스케이프 복원 처리
    #for b in merged_list:
    #    b["code"] = b["code"].encode().decode("unicode_escape")
    if len(merged_list) == 1:
        json_result = merged_list[0]
        #json_str = json.dumps(merged_list[0], ensure_ascii=False)
    else:
        json_result = merged_list
        #json_str = json.dumps(merged_list, ensure_ascii=False)
    
    # 2. 후처리로 이스케이프 문자 제거 (\n, \", 등)
    #cleaned_str = json_str.replace('""', '"') #json_str.encode().decode("unicode_escape")
    #cleaned_str = json_str.encode().decode("unicode_escape").replace('""', '"')

    # 3. 반환
    return json_result, 1 #cleaned_str, 1

def benchmark_each_command(model=model_default):
    # 실행 (assets are resolved next to run.py, so the cwd does not decide
    # which dataset is read or where the results land)
    datasets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datasets")
    things_path = os.path.join(datasets_dir, "things.json")
    f_file_name = os.path.join(datasets_dir, "final_output_250630_test.csv")
    # this mode's dataset is not part of the artifact: the missing files are
    # named in one line, where a traceback out of the first open() would leave
    # the reader to work out which mode they had fallen into
    missing = [p for p in (things_path, f_file_name) if not os.path.isfile(p)]
    if missing:
        raise SystemExit(
            "batch-benchmark mode needs the datasets/ directory, which is not shipped here.\n"
            "missing: " + ", ".join(os.path.normpath(p) for p in missing) + "\n"
            'Use: run.py "<command>" | run.py dataset | run.py --premap <version> "<command>"'
        )
    with open(things_path, "r") as f:
        things = json.load(f)
    # print(things)

    #model = "Qwen2.5-Coder:7B"
    is_english = False
    if is_english:
        english = "_english"
    else:
        english = ""

    # df = pd.read_excel("./final_output_250616.xlsx", engine='openpyxl')
    df = pd.read_csv(f_file_name, encoding='utf-8-sig')

    # 출력 경로 생성 및 한 줄 실시간 기록
    # 현재 시간 문자열 생성 (예: 20240625_153012)
    now_str = datetime.now().strftime("%y%m%d_%H%M%S")
    if f"joi{english}_pred_{model}" not in df.columns:
        df[f"joi_pred{english}_{model}"] = ""
        df[f"cloud_similarity{english}_{model}"] = 0.0
        df[f"script_similarity{english}_{model}"] = 0.0
        df[f"response_time{english}_{model}"] = ""
    fieldnames = list(df.columns)
    # 기존 파일명에서 확장자 분리
    base, ext = os.path.splitext(f_file_name)
    output_path = f"{base}_{now_str}{ext}"
    with open(output_path, "w", newline='', encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i, row in tqdm(df.iterrows()):    #.head(2) #.iterrows()
            if row.get("selected") != 1:
                continue
            print(f"Processing row {i+1}/{len(df)}: {row['command']}")

            sentence = row[f"command{english}"]
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            connected_devices = row["connected_devices"]
            other_params = row["options"]

            # 파싱
            if pd.isna(other_params):
                other_params = []
            else:
                try:
                    other_params = ast.literal_eval(other_params)
                except Exception as e:
                    print(f"row {i}: options not parseable ({e}); using []")
                    other_params = []

            if isinstance(connected_devices, str) and not pd.isna(connected_devices) and connected_devices.strip():
                try:
                    connected_devices = ast.literal_eval(connected_devices)
                except Exception as e:
                    print(f"row {i}: connected_devices not parseable ({e}); using things.json")
                    connected_devices = things
            else:
                connected_devices = things

            merged = 0
            start = time.time()
            resp = generate_joi_code(sentence=sentence,\
                                    model=model,\
                                    connected_devices= connected_devices,\
                                    current_time= current_time,\
                                    other_params= other_params)
            end = time.time()
            try:
                generated_code = (resp.get("code", ""))
            except Exception as e:
                print(f"row {i}: no code in the response ({e})")
                generated_code = []
            #print("Output:: ", generated_code)
            if isinstance(generated_code, list):
                if (generated_code[0].count('"name"') < 2):
                    generated_code = generated_code[0]
                    merged = 0
                else:
                    generated_code, merged = merge_duplicate_blocks(generated_code)
            elif generate_joi_code is None:
                generated_code = str("{}")
            else:
                generated_code = str(generated_code)

            resp_time = f"{end - start:.3f}"

            # gt = ast.literal_eval(row["joi_gt"])

            # eval = compare_codes(gt, generated_code)
            # total_score = eval["ast_similarity"]
            # if eval["cron_eqaul"]:
            #     total_score *= 0.5
            # if eval["period_eqaul"]:
            #     total_score *= 0.5
            
            df.at[i, f"joi_pred{english}_{model}"] = generated_code
            df.at[i, f"cloud_similarity{english}_{model}"] = 0.0
            df.at[i, f"script_similarity{english}_{model}"] = 0.0
            df.at[i, f"response_time{english}_{model}"] = float(resp_time)

            #df.to_csv(output_path, index=False, encoding='utf-8-sig')
            print("GT :: ", df.loc[i, 'joi_gt'])
            writer.writerow(df.loc[i].to_dict())
            f.flush()  # 즉시 디스크에 기록
    print(">> ", output_path, "... saved")


def joilang_each_command(sentences, model=model_default):
    # written next to run.py, like sentence_best_code_log.csv: a run from
    # another directory appends to the same log rather than starting a new one
    output_path = _run_log_path("output_each_command.csv")
    fieldnames = ["sentence", "model", "generated_code"]
    if isinstance(sentences, str):
        sentences = [sentences]
    # 파일이 있으면 append, 없으면 write
    needs_header = _log_needs_header(output_path, fieldnames)
    # generation writes both logs, so both headers are read here, before the
    # first request. sentence_best_code_log.csv is written from inside
    # generate_joi_code, and a foreign header found only there would refuse a
    # generation the model had already been paid for.
    _log_needs_header(_run_log_path(SENTENCE_LOG_NAME), SENTENCE_LOG_FIELDS)
    mode = "a" if os.path.isfile(output_path) else "w"
    resp = {}
    try:
        out = open(output_path, mode, newline='', encoding="utf-8-sig")
    except OSError as exc:
        # refused here, before the first request is sent: a run whose output the
        # CLI cannot record is a configuration error, and naming it costs
        # nothing where letting it through costs a generation per command
        raise SystemExit(
            f"cannot write the run log {output_path}: {exc.strerror}; check the file's permissions"
        ) from exc
    with out as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if needs_header:
            writer.writeheader()
        for sentence in sentences:
            try:
                print("Input Sentence:: ", sentence)
                resp = generate_joi_code(
                    sentence=sentence,
                    model=model,
                    connected_devices=None,
                    other_params=None #'user_id': "noname"'
                )   # current_time defaults to now inside generate_joi_code
                debug_print("Response:: ", resp)
                generated_code = resp.get("code", "")
                print("Output:: >> ", generated_code)
            except (ValueError, TypeError, ImportError, OSError):
                # a misconfiguration - an unknown pre-mapping mode, a missing
                # version package, an unreadable catalog or prompt file - is not
                # a sentence that generated nothing, and must not be logged as
                # one: it is raised so the caller sees it and exits non-zero
                raise
            except Exception as e:
                # generation itself already logs its own failures inside
                # generate_joi_code; whatever is left is per-sentence
                print("## <<1. joilang_each_command>> \n Error :", e)
                generated_code = ""

            debug_print("Output row:: ", generated_code)
            writer.writerow({
                "sentence": sentence,
                "model": model,
                "generated_code": generated_code
            })
            f.flush()
    return resp

if __name__ == '__main__':
    args = sys.argv[1:]
    # the value-taking flags are read before --premap is dropped, so an option
    # name standing where a value belongs is still visible as one
    for _flag, _var in (('--premap-model', 'PREMAP_MODEL'), ('--premap-mode', 'PREMAP_MODE')):
        if _flag in args:
            _i = args.index(_flag)
            # the next option is not this one's value: taking it as one would
            # send an option name to the retriever as a model id or a mode. A
            # blank value names neither, so it is missing too - premap.py reads
            # its own --model / --mode by the same rule
            if (_i + 1 >= len(args) or args[_i + 1].startswith('--')
                    or not args[_i + 1].strip()):
                raise SystemExit(f"{_flag} requires a value")
            globals()[_var] = args[_i + 1]
            del args[_i:_i + 2]
            PREMAP_ENABLED = True
    if '--premap' in args:
        args.remove('--premap')
        PREMAP_ENABLED = True
    if PREMAP_ENABLED:
        # the retriever's configuration is checked here, where the other
        # argument errors are refused, so an unknown mode or a model named by an
        # empty variable stops the run instead of surfacing later as a command
        # that generated nothing
        try:
            import premap as _premap
        except ImportError:
            from gpt_mg import premap as _premap
        try:
            _premap.normalize_model(PREMAP_MODEL)
            _premap.normalize_mode(PREMAP_MODE)
        except ValueError as exc:
            raise SystemExit(str(exc))
    if len(args) == 2:
        # the version directory is the remaining configuration argument, so it
        # is resolved here with the others: an unknown one named after the
        # command has been echoed reads as a failure of the command
        _import_version_loader(args[0])
    #dataset = ["if button 4 is swiped up, sound the alarm's siren.", '버튼4가 위로 스와이프되었으면 알람의 사이렌을 울려줘.']
    text = """블라인드가 열려 있고 조명이 꺼져 있으며 습도가 80% 이상이면 블라인드를 닫고 조명을 켜 줘. 창문이 닫혀 있고 에어컨이 꺼져 있으면 창문을 열고 에어컨을 켜 줘.
TV가 꺼져 있고 커튼이 닫혀 있으며 선풍기가 꺼져 있으면 TV를 켜고 커튼을 열어 줘. 스피커가 재생 중이고 조명이 꺼져 있으면 조명을 켜고 블라인드를 닫아 줘.
TV가 켜져 있고 스피커가 꺼져 있으며 조명이 꺼져 있으면 스피커를 켜고 조명을 켜 줘. 에어컨이 난방 모드이고 온도가 21도 이상이면 에어컨을 끄고 블라인드를 닫아 줘.
10초마다 알람과 사이렌을 껐다 켰다 반복해 줘.
매일 30초마다 선풍기를 껐다 켰다 반복해 줘.
토양 습도가 25% 이하가 되면 급수기를 켜고 이후 3초마다 상태를 확인해서 습도가 30% 이상이 될 때까지 급수기를 껐다 켰다 반복해 줘.
조명이 켜지면 블라인드를 닫고, 이후 2초마다 커튼을 열었다 닫았다 반복해 줘.
주말에는 5초마다 펌프가 꺼져 있으면 켜고, 켜져 있으면 꺼 주고, 선풍기 속도를 10으로 설정해 줘.
실시간으로 확인하여 토양 습도 센서 값이 연속으로 3회 15 이하를 기록하고 그 중 두 번째 값이 가장 낮았다면 관개 장치를 켜고 블라인드를 닫아 줘. 단, 3월은 제외하고.
매주 평일 오전 9시 창문이 닫혀 있고 이산화탄소 농도가 1000ppm 이상이며 기온이 30도 이상이면, 5초 후 창문을 열고, 팬이 꺼져 있으면 켜 줘. 그 후, 실시간으로 확인하여 1분 연속으로 미세먼지 농도가 50 이상으로 유지되면 창문을 다시 닫고, 팬도 꺼 줘. 만약 그 때 온실 내 습도가 40% 이하이면 가습기를 작동시켜 줘. 또한 마찬가지로 그 때 토양 습도가 25% 이하이고 관개 장치가 꺼져 있으면 물을 줘.
매일 밤 10시에 하단부에 있는 창문이 열려 있고 짝수 태그가 붙은 조명이 켜져 있으면 사이렌을 울려 줘. 사이렌을 울렸다면 5초 후 블라인드를 닫아 줘. 사이렌을 울렸는지 상관 없이 5초 후부터는 실시간으로 확인해서 홀수 태그가 붙은 문이 두 번 열렸다 닫히면 커튼을 닫아 줘. 커튼을 닫은 후 문이 두 번 열린 총 시간만큼 알람의 경광등을 켰다가 꺼 줘.
그룹2번이 하나라도 80을 초과하면 그룹2번은 꺼줘.
불이 30분이상 켜져있으면 알림을 울려줘.
1초마다 확인하여 관개 장치가 꺼졌다 켜진 횟수가 4번을 초과하고 펌프가 2번 이상 작동했으면 블라인드를 닫고 커튼을 내려 줘.
1초 주기로 확인해서 관개 장치가 직전에 꺼지고 이후 켜지는 횟수가 4번을 초과하고 펌프가 2번 이상 작동했으면 블라인드를 닫고 커튼을 내려 줘."""
    dataset = ['"' + line.strip() + '"' for line in text.strip().split('\n') if line.strip()]
    debug_print('args length:: ', len(args))
    if len(args) == 2:
        selected_model = args[0]
        # a blank command names nothing to generate from: sent as one it comes
        # back as a scenario built out of the prompt's boilerplate alone. premap.py
        # refuses an empty query in the same one line, for the same reason.
        if not args[1].strip():
            raise SystemExit("empty command: pass the command to generate from")
        dataset = [args[1]]
        resp = joilang_each_command(dataset, selected_model)

        try_no = 0
        current_sentence = dataset[0]  # 요구사항 누적용
        while True:
            if (all_items is None) or (len(all_items) == 0):
                print("No candidates generated.")
                break
            else:
                print(f"---\nCandidate #{choice_no+1}: {all_items[choice_no]}")
                ###########################
                ### re-converted sentence (optional; skipped if the module is absent)
                response_kor = ""
                try:
                    model_path = f"{selected_model}_reconverted.config_loader"
                    config_loader_module = importlib.import_module(model_path)
                    load_version_config = getattr(config_loader_module, 'load_version_config')

                    config, model_input = load_version_config(f"""넌 한글 언어학자 마스터야.
        {all_items[choice_no]}를 다시 한글 명령어로 바꿔서 아래 [ ] 를 채워줘. 단, 아래 조건들을 모두 만족하는, 한글 1~3 줄 커맨드로 구체적이고 정확하게 알아듣기 좋게 잘 변환해줘.
    all이나 any가 없는 경우 임의의 ~를 ~한다 임의의 ~가 ~를 만족하면과 같이 구체적으로 명시해줘. all/any도 마찬가지고.
    사용자 지정, 임의의 태그는 정확히 한글로 명시하는데, 특히 A, B, C와 같은 태그는 반드시 한글로 구분해줘. 예를 들어, 온실A, 온실B, 온실C와 같이.
    """) 
                    response_kor = client.chat.completions.create(**model_input)
                except ModuleNotFoundError:
                    print(f"{selected_model}_reconverted not found; skipping reverse translation.")
                    response_kor = ""
                if response_kor:
                    resp['log']['translated_sentence'] = response_kor.choices[0].message.content.strip() #content
                else:
                    print("No reconverted response, using original sentence.")
                    resp['log']['translated_sentence'] = current_sentence

                print("Reconverted Version of The Detailed Sentence: \n", resp['log']["translated_sentence"])
                ###########################
                # the scenario is already printed and logged at this point; the
                # follow-up loop is interactive, so a batch/CI run (no tty, or a
                # closed stdin) stops here instead of dying on EOFError
                if not sys.stdin.isatty():
                    print("non-interactive run: first candidate printed, exiting.")
                    break
                try:
                    answer = input(
                        "Accept this result? "
                        "(y: save / n: quit / Enter: next candidate / anything else: add a requirement) >>> ")
                except EOFError:
                    print("\nno input available; exiting.")
                    break


                if answer.lower() == 'y':
                    # logged as the one-element candidate list generation itself
                    # writes, so the column holds one shape whichever path wrote it
                    save_sentence_and_code(current_sentence, [all_items[choice_no]])
                    break
                elif answer.lower() == 'n':
                    print("exiting.")
                    break
                elif answer == "":
                    # 빈칸 엔터: 다음 후보로 이동
                    choice_no += 1
                    if choice_no >= len(all_items):
                        print("no further candidates.")
                        break
                else:
                    # 어떤 문자열이든 요구사항으로 누적
                    all_items = []
                    choice_no = 0
                    current_sentence += " " + f"+추가 조건: {answer}"
                    new_dataset = [current_sentence]
                    joilang_each_command(new_dataset, selected_model)
                    print(f"requirement {try_no} {answer!r} added; regenerated")
                    try_no += 1

    elif len(args) == 1:
        #selected_model = args[0]
        mode = args[0]
        # names no command, no version directory and no mode; refused where the
        # two-argument form refuses it, rather than run as an empty command
        if not mode.strip():
            raise SystemExit("empty command: pass the command to generate from")
        # a lone argument is a command unless it names a version directory next
        # to run.py, so a command may contain the word "version"
        version_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), mode)
        if(mode=='dataset'):
            for each_sentence in dataset:
                print('dataset length:: ', len(dataset))
                joilang_each_command(sentences=each_sentence)
        elif os.path.isfile(os.path.join(version_dir, "model_config.json")):
            # pre-mapping asked for with a version directory and no command is
            # the two-argument form with its command dropped far more often than
            # it is a request to pre-map a whole benchmark run; the batch mode
            # still runs, but not before the missing argument is named
            if PREMAP_ENABLED:
                print(f"note: {mode!r} names a version directory and no command followed it "
                      f'- for a single command use: run.py --premap {mode} "<command>"',
                      file=sys.stderr)
            benchmark_each_command(model = mode)
        else:
            # a single word is a version directory misspelled far more often
            # than it is a command; it is still run as one, but the reason it
            # took that branch is said before it reaches the model and the logs
            if len(mode.split()) < 2:
                print(f"note: {mode!r} names no version directory next to run.py "
                      "and is not 'dataset' - running it as a command",
                      file=sys.stderr)
            joilang_each_command(sentences=mode)
    elif not args:
        benchmark_each_command()
    else:
        # an unquoted multi-word command arrives as several arguments; saying so
        # beats silently starting an unrelated batch benchmark
        raise SystemExit(
            f"unrecognized arguments: {args}\n"
            'Usage: run.py [--premap] [--premap-model M] [--premap-mode split|nosplit] '
            '<version_dir> "<command>"\n'
            '       run.py "<command>"     single command, default version\n'
            '       run.py dataset         bundled example batch\n'
            '       run.py                 batch benchmark (needs ../datasets/)\n'
            "Quote the command so it stays one argument."
        )
