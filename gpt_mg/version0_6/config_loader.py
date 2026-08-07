import json
import os

def _read_text(path):
    """Read one UTF-8 prompt asset."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"prompt asset not found: {path}")
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return handle.read()
    except OSError as exc:
        reason = exc.strerror or exc
        raise OSError(f"prompt asset is not readable: {path} ({reason})") from exc
    except UnicodeDecodeError as exc:
        raise ValueError(f"{path} is not valid UTF-8: {exc}") from exc


def _read_json(path):
    """Read one JSON prompt asset."""
    try:
        return json.loads(_read_text(path))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} is not valid JSON: {exc}") from exc


def parse_selected_device_simple(connected_devices: dict):
    if not connected_devices:
        return None, [], {}

    categories = set()
    extra_tags = set()
    for device in connected_devices.values():
        category = device.get("category")
        if category:
            categories.add(category)
        extra_tags.update(
            tag for tag in device.get("tags", []) if tag != category
        )

    category_tags = "[" + ", ".join(
        f"#{category}" for category in sorted(categories)
    ) + "]"
    return category_tags, list(extra_tags), categories


def load_version_config(
    user_input,
    connected_devices: dict = None,
    other_params: dict = None,
    base_path: str = ".",
    premap_services: dict = None,
):
    version_dir = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), base_path)
    )

    config = _read_json(os.path.join(version_dir, "model_config.json"))
    model_input = config["model_input"]

    grammar = _read_text(os.path.join(version_dir, "grammar_ver1.5.10.md"))
    service_prompt = _read_text(os.path.join(version_dir, "service_prompt_10.md"))
    value_catalog = _read_json(
        os.path.join(version_dir, "service_list_ver1.5.4_value.json")
    )
    function_catalog = _read_json(
        os.path.join(version_dir, "service_list_ver1.5.4_function.json")
    )

    if premap_services:
        no_match = set(premap_services.get("no_match") or ())
        if "value" not in no_match:
            value_catalog = premap_services.get("value", [])
        if "function" not in no_match:
            function_catalog = premap_services.get("function", [])

    tempo = _read_text(os.path.join(version_dir, "tempo_prompt_9.md"))
    caution = _read_text(os.path.join(version_dir, "caution_prompt_8.md"))
    response_prompt = _read_text(
        os.path.join(version_dir, "response_prompt_6.md")
    )

    category_tags, _, _ = parse_selected_device_simple(connected_devices)
    connected_context = (
        f"\n\n---\n[connected_devices]\n {category_tags}"
        if category_tags
        else ""
    )

    if other_params:
        compact_params = json.dumps(
            other_params,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        user_context = f"\n\n---\n[userinfo]\n {compact_params}"
        user_context = (
            user_context.replace('\\"', '"')
            .replace("\\n", "")
            .replace("    ", "")
            .replace("   ", "")
            .strip()
        )
    else:
        user_context = ""

    system_prompt = f"""
You are a JOILang programmer. JOILang is a programming language used to control IoT devices.
Use the following knowledge to convert natural language into valid JOILang code.

Make sure to follow syntax rules strictly. Only use allowed keywords:
if, else if, else, >=, <=, ==, !=, not, and, or, wait until, (#Clock).clock_delay() 
The delay function (#Clock).clock_delay() only accepts values in milliseconds (ms).
Do not use while or any unlisted constructs. 
**Never use `while` in code**
[Incorrect Example]
while (blinkCount < 10)

---

[Device and Service Mapping]
IMPORTANT: You MUST extract **all device tags mentioned as subjects or objects in the input sentence**, including those connected by conjunctions such as "and" or "with".  
For each extracted device tag, retrieve **all associated services** (both value and function names) exactly as defined in the [Service List].  
**Do not omit any device or service even if their names overlap or repeat.**  
If multiple devices share similar service names (e.g., "alarm" function on both Alarm and Siren devices), include the services for each device separately and comprehensively.  
{service_prompt}
[service_list_value]
{value_catalog}
[service_list_function]
{function_catalog}

---
[Grammar]
{grammar}


---
[Condition Combination Rules]
{tempo}


---
[Important Cautions]
{caution}
{connected_context}
{user_context}

---
{response_prompt}
- **Never use `while` in code**

--- JOILang Code Output Format Guide ---
Every scenario generated will follow this structure:
```json
{{
  "name": "<명령의 의도를 영어로 **축약하여**, 띄어쓰기 없이 간결한 형태로 작성하세요. 너무 길게 쓰지 말고, 조합된 단어로 의미만 담아내세요.>",
  "cron": "<Time-based trigger to start execution>",
  "period": <Execution interval in milliseconds or -1>,
  "code": "<Main logic block written in JOILang>"
}}
```

"""

    messages = []
    for message in model_input["messages"]:
        content_key = message.get("content")
        if content_key == "system_prompt":
            content = system_prompt
        elif content_key == "sentence":
            content = user_input
        else:
            content = ""
        messages.append({"role": message["role"], "content": content})

    model_input["messages"] = messages
    return config, model_input
