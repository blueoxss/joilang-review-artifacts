"""Service pre-mapping for JOILang generation.

The default retriever uses ``intfloat/e5-small`` with a BM25/token-overlap
sparse term, separate VALUE and FUNCTION indexes, dense weight 0.6, and a
six-per-role shortlist. Dense retrieval is optional; metadata records the
configured and effective search paths, fallbacks, and roles without a usable
ranking signal.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re

import numpy as np

DEFAULT_MODEL = "e5-small"
DENSE_WEIGHT = 0.6

SPARSE_BM25_WEIGHT = 0.7
SPARSE_OVERLAP_WEIGHT = 0.3
TOP_K_PER_ROLE = 6

MAX_TOTAL = 12

PREMAP_MODELS = {
    "e5-small": {"hf": "intfloat/e5-small", "search": "hybrid", "prefix": True},
    "mxbai-embed-xsmall-v1": {"hf": "mixedbread-ai/mxbai-embed-xsmall-v1", "search": "hybrid", "prefix": False},
    "multilingual-e5-small": {"hf": "intfloat/multilingual-e5-small", "search": "dense", "prefix": True},
    "bge-m3": {"hf": "BAAI/bge-m3", "search": "dense", "prefix": False},
}

_CHECKPOINT_BASES = (
    ("bge-m3", "bge-m3"),
    ("multilingual", "multilingual-e5-small"),
    ("mxbai", "mxbai-embed-xsmall-v1"),
    ("e5", "e5-small"),
)


def resolve_model(name: str | None):
    """Resolve a model key, Hugging Face id, or checkpoint path."""
    name = (name or "").strip() or DEFAULT_MODEL
    lowered = name.lower()

    for key, cfg in PREMAP_MODELS.items():
        if lowered in (key.lower(), cfg["hf"].lower()):
            out = dict(cfg)
            out["key"] = key
            return out

    base_name = os.path.basename(name.rstrip("/")) or name

    hf_id = os.path.normpath(os.path.abspath(name)) if os.path.isdir(name) else name
    for token, key in _CHECKPOINT_BASES:
        if token in base_name.lower():
            out = dict(PREMAP_MODELS[key])
            out["key"] = base_name
            out["hf"] = hf_id
            out["base_key"] = key
            return out
    return {"key": base_name, "hf": hf_id, "search": "hybrid", "prefix": False}

def _tokenize(text):
    """Return lowercase word tokens while keeping identifiers intact."""
    return re.findall(r"[0-9a-zA-Z_가-힣]+", (text or "").lower())

class SimpleBM25:
    def __init__(self, texts, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.docs = [_tokenize(t) for t in texts]
        self.doc_lens = np.array([len(d) for d in self.docs], dtype=np.float32)

        self.avgdl = (float(self.doc_lens.mean()) if len(self.doc_lens) else 1.0) or 1.0
        self.tf = []
        df = {}
        for doc in self.docs:
            counts = {}
            for tok in doc:
                counts[tok] = counts.get(tok, 0) + 1
            self.tf.append(counts)
            for tok in counts:
                df[tok] = df.get(tok, 0) + 1
        n_docs = max(1, len(self.docs))
        self.idf = {tok: math.log(1.0 + (n_docs - v + 0.5) / (v + 0.5)) for tok, v in df.items()}

    def get_scores(self, query):
        q = _tokenize(query)
        scores = np.zeros(len(self.docs), dtype=np.float32)
        if not q or not len(self.docs):
            return scores
        for idx, counts in enumerate(self.tf):
            dl = self.doc_lens[idx] if len(self.doc_lens) else 1.0
            denom_base = self.k1 * (1.0 - self.b + self.b * (dl / self.avgdl))
            total = 0.0
            for tok in q:
                if tok not in counts:
                    continue
                freq = float(counts[tok])
                total += self.idf.get(tok, 0.0) * (freq * (self.k1 + 1.0)) / (freq + denom_base)
            scores[idx] = total
        return scores

def _minmax(x):
    if x.size == 0:
        return x
    den = float(x.max() - x.min())
    if abs(den) < 1e-9:
        return np.zeros_like(x)
    return (x - x.min()) / den

def _entry_text(item, kind):
    device = str(item.get("device", ""))
    service = str(item.get("service", ""))
    descriptor = str(item.get("descriptor", ""))
    if kind == "value":
        extra = str(item.get("return_descriptor", ""))
    else:
        extra = " ".join(
            [
                str(item.get("argument_descriptor", "")),
                str(item.get("argument_type", "")),
                str(item.get("argument_bounds", "")),
            ]
        )
    return f"{device} {service} {descriptor} {extra}"

_DENSE_MODELS: dict = {}
_DENSE_ERRS: dict = {}
_DENSE_NOTES: dict = {}
_PASSAGE_CACHE: dict = {}

_ST_UNAVAILABLE = ""


def _path_pattern(seg, sep):
    """Build one absolute-path regular-expression branch."""
    space = r"[ ](?=(?:%s[ ])*%s%s)" % (seg, seg, sep)
    return r"(?:%s%s(?:%s%s)*)+" % (sep, seg, space, seg)

_WIN_SEG = r"[^\\/\s:,;'\"]+"
_WIN_SEP = r"[\\/]+"

_WIN_EXT = r"\\\\[?.][\\/](?:UNC)?(?:[A-Za-z]:)?"
_ABS_PATH_RE = re.compile(
    r"(?<![\w:/.])" + _path_pattern(r"[^/\s:,;'\"]+", "/")
    + r"|(?<![\w\\/])" + _WIN_EXT + _path_pattern(_WIN_SEG, _WIN_SEP)
    + r"|(?<!\w)[A-Za-z]:" + _path_pattern(_WIN_SEG, _WIN_SEP)
    + r"|(?<![\w\\/])(?=\\\\)" + _path_pattern(_WIN_SEG, _WIN_SEP)
)
_SEP_RE = re.compile(r"[\\/]")


def _strip_host_paths(text):
    """Replace absolute host paths with their final component."""
    return _ABS_PATH_RE.sub(lambda m: _SEP_RE.split(m.group(0))[-1], text)

def _short(exc, limit=200):
    """Format an exception for one-line metadata without host paths."""
    text = " ".join(f"{type(exc).__name__}: {exc}".split())
    text = _strip_host_paths(text)
    return text if len(text) <= limit else text[: limit - 3] + "..."

def _is_permanent_load_error(exc):
    """Return whether retrying the same model id cannot succeed."""
    if isinstance(exc, (FileNotFoundError, NotADirectoryError)):
        return True
    text = f"{type(exc).__name__}: {exc}".lower()
    return (
        "repositorynotfound" in text
        or "repository not found" in text
        or "hfvalidationerror" in text
        or "not a valid model identifier" in text
    )

def _is_resource_error(exc):
    """Return whether a failure indicates exhausted compute memory."""
    text = f"{type(exc).__name__}: {exc}".lower()
    return "outofmemory" in text or "out of memory" in text

def _load_dense_model(hf_id):
    global _ST_UNAVAILABLE
    if hf_id in _DENSE_MODELS:
        return _DENSE_MODELS[hf_id], ""
    if _ST_UNAVAILABLE:
        return None, _ST_UNAVAILABLE
    if hf_id in _DENSE_ERRS:
        return None, _DENSE_ERRS[hf_id]
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as exc:
        _ST_UNAVAILABLE = f"sentence_transformers_unavailable:{_short(exc)}"
        return None, _ST_UNAVAILABLE
    try:
        _DENSE_MODELS[hf_id] = SentenceTransformer(hf_id)
    except Exception as exc:
        if _is_resource_error(exc):
            try:
                _DENSE_MODELS[hf_id] = SentenceTransformer(hf_id, device="cpu")
                _DENSE_NOTES[hf_id] = f"dense_model_cpu_fallback:{type(exc).__name__}"
                return _DENSE_MODELS[hf_id], ""
            except Exception as cpu_exc:
                exc = cpu_exc

        err = f"dense_model_load_failed:{_short(exc)}"

        if _is_permanent_load_error(exc):
            _DENSE_ERRS[hf_id] = err
        return None, err
    return _DENSE_MODELS[hf_id], ""

ROLE_ENC_PREFIX = {"value": "condition: ", "function": "action: "}


def _query_overflow(model, text):
    """Return the untruncated token count when it exceeds the encoder limit."""
    limit = int(getattr(model, "max_seq_length", 0) or 0)
    tokenizer = getattr(model, "tokenizer", None)
    if not limit or tokenizer is None or len(text) < limit:
        return 0
    log = logging.getLogger("transformers")
    level = log.level
    log.setLevel(logging.ERROR)
    try:
        n_tokens = len(tokenizer(text, truncation=False)["input_ids"])
    except Exception:
        return 0
    finally:
        log.setLevel(level)
    return n_tokens if n_tokens > limit else 0

def _dense_scores(query, texts, cache_key, hf_id, prefix, enc_query=None):
    """Return cosine scores, a fallback reason, and any query overflow."""
    model, err = _load_dense_model(hf_id)
    if model is None:
        return None, err, 0
    try:
        digest = hashlib.blake2b("\x00".join(texts).encode("utf-8"), digest_size=16).hexdigest()
        key = (hf_id, cache_key, len(texts), digest)
        if key not in _PASSAGE_CACHE:
            passages = [("passage: " + t) if prefix else t for t in texts]
            _PASSAGE_CACHE[key] = np.asarray(
                model.encode(passages, normalize_embeddings=True, show_progress_bar=False)
            )
        q_raw = enc_query if enc_query is not None else query
        q_text = ("query: " + q_raw) if prefix else q_raw
        overflow = _query_overflow(model, q_text)
        q_emb = np.asarray(model.encode([q_text], normalize_embeddings=True, show_progress_bar=False))
        return np.matmul(q_emb, _PASSAGE_CACHE[key].T).flatten(), "", overflow
    except Exception as exc:
        return None, f"dense_encode_failed:{_short(exc)}", 0

def _search_entries(query, entries, cache_key, cfg, top_k, weight, role=None):
    """Rank one index and return hits, fallback information, and diagnostics.

    Diagnostics distinguish a real ranking from all-equal scores and record the
    search path used for the role.
    """
    texts = [e["text"] for e in entries]
    if not texts:
        return [], f"empty_catalog:{role or cache_key}", {
            "ranked": False,
            "lexical": None,
            "overflow": 0,
            "one_device": "",
            "search": None,
        }
    enc_query = (ROLE_ENC_PREFIX[role] + query) if role in ROLE_ENC_PREFIX else None
    dense, fallback, overflow = _dense_scores(
        query, texts, cache_key, cfg["hf"], cfg["prefix"], enc_query)

    lexical_signal = None
    if cfg["search"] == "dense" and dense is not None:
        final = _minmax(dense)
        role_search = "dense"
    else:
        bm25_scores = SimpleBM25(texts).get_scores(query)
        q_tokens = set(_tokenize(query))
        q_len = max(1, len(q_tokens))
        lexical = np.zeros(len(texts), dtype=np.float32)
        for idx, text in enumerate(texts):
            tokens = set(_tokenize(text))
            lexical[idx] = len(q_tokens & tokens) / math.sqrt(max(1, q_len * max(1, len(tokens))))
        sparse = SPARSE_BM25_WEIGHT * _minmax(bm25_scores) + SPARSE_OVERLAP_WEIGHT * _minmax(lexical)
        lexical_signal = bool(sparse.size) and float(sparse.max()) > 0.0
        if dense is not None:
            final = weight * _minmax(dense) + (1.0 - weight) * sparse
            role_search = "hybrid"
        else:
            final = sparse
            role_search = "sparse"
            fallback = fallback or "dense_unavailable"

    separates = bool(final.size) and float(final.max() - final.min()) > 1e-9
    diagnosis = {
        "ranked": bool(separates and _tokenize(query)),
        "lexical": lexical_signal,
        "overflow": overflow,
        "search": role_search,
    }

    ranked = np.argsort(final)[::-1][:top_k]
    out = []
    for idx in ranked:
        e = entries[int(idx)]
        out.append(
            {
                "kind": e["kind"],
                "device": e["payload"].get("device", ""),
                "service": e["payload"].get("service", ""),
                "score": float(final[int(idx)]),
                "payload": e["payload"],
            }
        )

    hit_devices = {h["device"] for h in out}
    index_devices = {e["payload"].get("device", "") for e in entries}
    diagnosis["one_device"] = (
        next(iter(hit_devices))
        if len(out) > 1 and len(hit_devices) == 1 and len(index_devices) > 1
        else ""
    )
    return out, fallback, diagnosis

def _make_entries(items, kind):
    """Convert catalog rows to role-tagged retrieval entries."""
    if not isinstance(items, (list, tuple)):
        raise TypeError(
            f"{kind} catalog is {type(items).__name__}, expected a list of service objects"
        )
    entries = []
    for i, it in enumerate(items):
        if not isinstance(it, dict):
            raise TypeError(
                f"{kind} catalog row {i} is {type(it).__name__}, expected an object: {it!r}"
            )

        missing = [
            k for k in ("device", "service")
            if it.get(k) is None or not str(it[k]).strip()
        ]
        if missing:
            raise ValueError(
                f"{kind} catalog row {i} has no {' or '.join(missing)}: {it!r}"
            )
        entries.append({"kind": kind, "text": _entry_text(it, kind), "payload": it})
    return entries

_MULTILINGUAL_KEYS = {"multilingual-e5-small", "bge-m3"}
_NON_LATIN_RE = re.compile(r"[가-힣ぁ-ゖァ-ヿ一-鿿]")


def _diagnose(query, cfg, diagnoses, dense_ran):
    """Summarize ranking signal and soft retrieval warnings."""
    notes = []

    uncovered_script = False
    if not query.strip():
        notes.append("empty_query")
    elif not _tokenize(query):
        notes.append("no_query_tokens")
    elif _NON_LATIN_RE.search(query) and cfg.get("base_key", cfg["key"]) not in _MULTILINGUAL_KEYS:
        uncovered_script = True

        notes.append(
            f"non_latin_query_on_english_encoder:{cfg['key']}" if dense_ran
            else "non_latin_query_on_english_catalog"
        )
    for role, d in sorted(diagnoses.items()):
        if d["ranked"] and d["lexical"] is False:
            notes.append(f"no_lexical_match:{role}")
        if d.get("one_device"):
            notes.append(f"single_device_shortlist:{role}:{d['one_device']}")

    searched = [role for role, d in diagnoses.items() if d.get("search")]
    degraded = [role for role, d in sorted(diagnoses.items()) if d.get("search") == "sparse"]
    if degraded and len(degraded) < len(searched):
        notes.extend(f"dense_unavailable:{role}" for role in degraded)
    overflow = max([d.get("overflow") or 0 for d in diagnoses.values()] or [0])
    if overflow:
        notes.append(f"dense_query_truncated:{overflow}")

    no_match = sorted(
        role for role, d in diagnoses.items()
        if not d["ranked"] or (uncovered_script and d["lexical"] is False)
    )
    if not no_match:
        signal = "ok"
    elif len(no_match) == len(diagnoses):
        signal = "none"
    else:
        signal = "partial"
    return no_match, signal, notes

def _finalize(selected, cfg, mode, weight, top_k_per_role, max_total, fallback, query, diagnoses,
              dropped=0):

    role_search = {role: d.get("search") for role, d in sorted(diagnoses.items())}
    searches = {s for s in role_search.values() if s}
    if "hybrid" in searches:
        effective_search = "hybrid"
    elif "dense" in searches:
        effective_search = "dense"
    elif "sparse" in searches:
        effective_search = "sparse"
    else:
        effective_search = "none"
    if effective_search == "hybrid":
        backend = f"{cfg['key']}+bm25({mode})"
    elif effective_search == "dense":
        backend = f"{cfg['key']}(dense,{mode})"
    elif effective_search == "sparse":
        backend = f"bm25({mode})"
    else:
        backend = f"none({mode})"
    metadata = {
        "backend": backend,
        "model_key": cfg["key"],

        "model_name": _strip_host_paths(cfg["hf"]),

        "configured_search": cfg["search"],
        "effective_search": effective_search,
        "prefix": cfg["prefix"],
        "mode": mode,
        "top_k_per_role": top_k_per_role,
        "max_total": max_total,
        "fallback": fallback,
        "selected_count": len(selected),
        "selected": [f'{h["kind"]}:{h["device"]}.{h["service"]}' for h in selected],
    }

    no_match, signal, notes = _diagnose(
        query, cfg, diagnoses, bool(searches & {"hybrid", "dense"}))

    if dropped:
        notes.append(f"duplicate_entries_dropped:{dropped}")

    if selected:
        ranked_kinds = {h["kind"] for h in selected}
        notes.extend(
            f"empty_role_shortlist:{role}"
            for role in ("value", "function")
            if role not in no_match and role not in ranked_kinds
        )
    scores = [round(float(h["score"]), 4) for h in selected]
    metadata["retrieval_signal"] = signal
    metadata["no_match"] = no_match

    metadata["applied"] = bool({"value", "function"} - set(no_match))
    metadata["notes"] = notes
    metadata["top_score"] = max(scores) if scores else 0.0
    metadata["selected_scores"] = scores

    metadata["dense_weight"] = weight if "hybrid" in searches else None
    metadata["sparse_mix"] = (
        {"bm25": SPARSE_BM25_WEIGHT, "overlap": SPARSE_OVERLAP_WEIGHT}
        if searches & {"hybrid", "sparse"} else None
    )

    if len(set(role_search.values())) > 1:
        metadata["role_search"] = role_search
    if cfg.get("base_key"):
        metadata["base_model_key"] = cfg["base_key"]

    note = _DENSE_NOTES.get(cfg["hf"], "")
    if note and searches & {"hybrid", "dense"}:
        metadata["dense_note"] = note
    premap_services = {
        "value": [h["payload"] for h in selected if h["kind"] == "value"],
        "function": [h["payload"] for h in selected if h["kind"] == "function"],

        "no_match": no_match,
    }
    return premap_services, metadata

_COND_CLAUSE_RE = re.compile(
    r"^(if|when|whenever|while|until|once|as soon as|the moment|in case|suppose)\b[^,]*",
    re.IGNORECASE,
)

_NOTIFY_RE = re.compile(
    r"^(please )?(send (me )?an? ?(notification|alert|message)|notify( me)?|alert( me)?|let me know)\b",
    re.IGNORECASE,
)


def is_notification_clause(clause):
    return bool(_NOTIFY_RE.match(clause.strip()))

def decompose_query(query):
    """Split a leading conditional clause from its action.

    Commands without a usable boundary, and notification-led action tails, are
    searched in full for both roles.
    """
    m = _COND_CLAUSE_RE.match(query.strip())
    if m and "," in query:
        condition = m.group(0).strip()
        action = query[query.index(",") + 1:].strip()

        if (condition and action and _tokenize(condition[m.end(1):])
                and not is_notification_clause(action)):
            return condition, action
    return query, query

def run_premap_split(
    query,
    service_list_value,
    service_list_function,
    model=None,
    weight=DENSE_WEIGHT,
    top_k_per_role=TOP_K_PER_ROLE,
    max_total=MAX_TOTAL,
):
    """Search VALUE and FUNCTION indexes separately and merge both budgets."""

    if top_k_per_role < 1:
        raise ValueError(
            f"top_k_per_role ({top_k_per_role}) must be at least 1: a role's "
            "budget below one selects nothing for that role, and a negative one "
            "is read as a slice offset rather than a budget"
        )
    if 2 * top_k_per_role > max_total:
        raise ValueError(
            f"max_total ({max_total}) must admit both roles' candidates "
            f"(2 * top_k_per_role = {2 * top_k_per_role}): the two role scores "
            "are normalized inside their own index and cannot be compared, so "
            "the cap must not be what chooses between them"
        )
    cfg = resolve_model(model)
    q_value, q_function = decompose_query(query)
    value_hits, fb_v, diag_v = _search_entries(
        q_value, _make_entries(service_list_value, "value"), "value", cfg, top_k_per_role, weight,
        role="value",
    )
    function_hits, fb_f, diag_f = _search_entries(
        q_function, _make_entries(service_list_function, "function"), "function", cfg, top_k_per_role, weight,
        role="function",
    )

    merged = sorted(value_hits + function_hits, key=lambda h: h["score"], reverse=True)
    seen, selected, dropped = set(), [], 0
    for hit in merged:
        key = (hit["kind"], hit["device"], hit["service"])
        if key in seen:
            dropped += 1
            continue
        seen.add(key)
        selected.append(hit)

        if len(selected) >= max_total:
            break

    fallback = "; ".join(dict.fromkeys(f for f in (fb_v, fb_f) if f))
    services, metadata = _finalize(
        selected, cfg, "split", weight, top_k_per_role, max_total, fallback,
        query, {"value": diag_v, "function": diag_f}, dropped=dropped,
    )
    if (q_value, q_function) != (query, query):
        metadata["decomposed_query"] = {"value": q_value, "function": q_function}
    return services, metadata

def run_premap_nosplit(
    query,
    service_list_value,
    service_list_function,
    model=None,
    weight=DENSE_WEIGHT,
    max_total=MAX_TOTAL,
):
    """Search one combined index while retaining catalog roles in output."""
    cfg = resolve_model(model)
    entries = _make_entries(service_list_value, "value") + _make_entries(service_list_function, "function")
    hits, fallback, diag = _search_entries(query, entries, "all", cfg, max_total, weight)

    seen, selected, dropped = set(), [], 0
    for hit in hits:
        key = (hit["kind"], hit["device"], hit["service"])
        if key in seen:
            dropped += 1
            continue
        seen.add(key)
        selected.append(hit)

    diagnoses = {}
    for role in ("value", "function"):
        role_hits = [h for h in selected if h["kind"] == role]
        role_devices = {h["device"] for h in role_hits}
        role_index_devices = {
            e["payload"].get("device", "") for e in entries if e["kind"] == role
        }
        diagnoses[role] = dict(
            diag,
            one_device=(
                next(iter(role_devices))
                if len(role_hits) > 1 and len(role_devices) == 1
                and len(role_index_devices) > 1
                else ""
            ),
        )
    return _finalize(selected, cfg, "nosplit", weight, None, max_total, fallback,
                     query, diagnoses, dropped=dropped)

def normalize_mode(mode=None):
    """Resolve the pre-mapping mode and reject blank or unknown values."""
    if mode is not None and not str(mode).strip():
        raise ValueError("empty pre-mapping mode; expected 'split' or 'nosplit'")
    mode = (mode or os.getenv("JOI_PREMAP_MODE") or "").strip().lower() or "split"
    if mode not in ("split", "nosplit"):
        raise ValueError(f"unknown pre-mapping mode {mode!r}; expected 'split' or 'nosplit'")
    return mode

def normalize_model(model=None):
    """Resolve the retriever id and reject an explicitly blank value."""
    if model is not None and not str(model).strip():
        raise ValueError(
            "empty pre-mapping model; pass a model key, an HF id or a checkpoint path"
        )
    return model or (os.getenv("JOI_PREMAP_MODEL") or "").strip() or DEFAULT_MODEL

def _load_catalog(path):
    """Load and validate one JSON service catalog."""

    if os.path.isdir(path):
        raise IsADirectoryError(f"service catalog is a directory, not a file: {path}")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"service catalog not found: {path}")
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except OSError as exc:
        raise OSError(f"service catalog is not readable: {path} ({exc.strerror or exc})") from exc
    except UnicodeDecodeError as exc:
        raise ValueError(f"{path} is not valid UTF-8: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(data, list):
        raise TypeError(f"{path} holds {type(data).__name__}, expected a list of service objects")
    return data

def premap_from_version_dir(query, version_dir, model=None, mode=None, **kwargs):
    """Load a version catalog pair and run the selected pre-mapper."""
    model = normalize_model(model)
    mode = normalize_mode(mode)

    service_list_value = _load_catalog(os.path.join(version_dir, "service_list_ver1.5.4_value.json"))
    service_list_function = _load_catalog(os.path.join(version_dir, "service_list_ver1.5.4_function.json"))

    if mode == "nosplit":
        if "top_k_per_role" in kwargs:
            raise ValueError(
                "top_k_per_role does not apply in nosplit mode; "
                "the combined index takes max_total"
            )
        return run_premap_nosplit(query, service_list_value, service_list_function, model=model, **kwargs)
    return run_premap_split(query, service_list_value, service_list_function, model=model, **kwargs)

DEMO_QUERY = "If it rains, switch the living-room air purifier to auto mode."

USAGE = (
    'Usage: premap.py [--mode split|nosplit] [--model M] "<command>"\n'
    "       M: a model key (e5-small, mxbai-embed-xsmall-v1, multilingual-e5-small,\n"
    "          bge-m3), a Hugging Face id, or a local checkpoint directory.\n"
    "Prints the pre-mapping metadata for the command as JSON; with no command it\n"
    "runs a demo one. No API key is needed.\n"
    "Quote the command. Bare words left unquoted by the shell are joined with\n"
    'single spaces, so `premap.py sound the alarm` pre-maps "sound the alarm";\n'
    "a mistyped option is still refused rather than folded into the query."
)


if __name__ == "__main__":
    import sys

    argv = sys.argv[1:]
    model = mode = None
    words, i = [], 0
    while i < len(argv):
        flag = argv[i]
        if flag in ("-h", "--help"):
            print(USAGE)
            raise SystemExit(0)
        if flag in ("--model", "--mode"):
            if i + 1 >= len(argv) or argv[i + 1].startswith("--") or not argv[i + 1].strip():
                raise SystemExit(f"{flag} requires a value\n{USAGE}")
            if flag == "--model":
                model = argv[i + 1]
            else:
                mode = argv[i + 1]
            i += 2
            continue
        if flag.startswith("--"):
            raise SystemExit(f"unrecognized argument {flag!r}\n{USAGE}")
        words.append(flag)
        i += 1

    if not words:
        q = DEMO_QUERY
        print(f"no query given; running the demo query: {DEMO_QUERY}", file=sys.stderr)
    else:
        q = " ".join(words)
        if not q.strip():
            raise SystemExit("empty query: pass the command to pre-map")

    try:
        mode = normalize_mode(mode)
    except ValueError as exc:
        raise SystemExit(str(exc))
    here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "version0_6")
    try:
        services, meta = premap_from_version_dir(q, here, model=model, mode=mode)
    except (OSError, ValueError, TypeError) as exc:
        raise SystemExit(str(exc))

    if model and "dense_model_load_failed" in (meta.get("fallback") or ""):
        raise SystemExit(
            f"--model {model!r} did not load, and the ranking would come from "
            f"the sparse half under its name: {meta['fallback']}\n" + USAGE)
    print(json.dumps(meta, ensure_ascii=False, indent=2))
