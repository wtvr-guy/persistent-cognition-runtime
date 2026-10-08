"""Explicit model management; catalog lookups and downloads require consent."""
from __future__ import annotations

from persistent_cognition.resource_limits import bounded_http_request

from html.parser import HTMLParser
import json
import re
from urllib.parse import urljoin, urlsplit

import httpx

from persistent_cognition.runtime_settings import ModelSelection
from persistent_cognition.model_parameters import parameter_catalog
from persistent_cognition.network_consent import NetworkPurpose, require_destination
from persistent_cognition.openai_transport import OPENAI_BASE_URL, OPENAI_ORIGIN

CATALOG_ORIGIN = "https://ollama.com"
DOWNLOAD_ORIGIN = "https://registry.ollama.ai"
MODEL_HTTP_TIMEOUT_SECONDS = 10
MODEL_PULL_TIMEOUT_SECONDS = 300
MAX_CATALOG_BYTES = 2097152
MAX_CATALOG_RESULTS = 120
# Ollama's own site renders a fixed page per query/sort with no further
# pagination (offset/page/cursor/limit params are all silently ignored); the
# only way to broaden coverage through its plain HTML search is to fan out
# across every sort order it actually supports and merge/dedupe the results.
CATALOG_SORTS = ("popular", "newest")
# Registered browsing categories for the Models page. These are independent of
# `model_admission.TASKS` (chat-response routing): a category here only needs
# to be a useful thing to browse for, not a text-completion specialist. Only
# entries with a `chat_task` correspond to an actual routable chat task.
CATALOG_CATEGORIES = {
    "general": {"label": "General", "keywords": ("",), "capability": None, "chat_task": "general"},
    "coding": {"label": "Coding", "keywords": ("coder", "code"), "capability": None, "chat_task": "coding"},
    "vision": {"label": "Vision", "keywords": ("",), "capability": "vision", "chat_task": "vision"},
    "tools": {"label": "Tool use", "keywords": ("",), "capability": "tools", "chat_task": None},
    "thinking": {"label": "Reasoning", "keywords": ("",), "capability": "thinking", "chat_task": None},
    "embedding": {"label": "Embedding", "keywords": ("",), "capability": "embedding", "chat_task": None},
    # Ollama's catalog has no "base"/foundation filter; a keyword match is a
    # heuristic over model descriptions, not a verified base-vs-instruct flag.
    "base": {"label": "Base / foundation", "keywords": ("base", "pretrained", "foundation"), "capability": None, "chat_task": None},
}


def ollama_request(base_url, path, *, method="GET", body=None):
    require_destination(base_url, NetworkPurpose.MODEL)
    with httpx.Client(base_url=base_url, trust_env=False, follow_redirects=False,
                      timeout=MODEL_HTTP_TIMEOUT_SECONDS) as client:
        response = bounded_http_request(client, method, path, json=body)
        response.raise_for_status()
        return response.json() if response.content else {}


def installed_models(base_url):
    result = ollama_request(base_url, "/api/tags")
    return result.get("models", [])


def model_details(base_url, name):
    ModelSelection(model=name)
    data = ollama_request(base_url, "/api/show", method="POST", body={"model": name})
    if data.get("remote_host") or data.get("remote_model"):
        raise ValueError("This model delegates to a cloud service; choose an explicit supported provider")
    return {"model": name, "details": data.get("details", {}),
            "capabilities": data.get("capabilities", []), "model_info": data.get("model_info", {}),
            "parameters": data.get("parameters", ""),
            "controls": parameter_catalog("ollama", name, data)}


def validate_local_selection(base_url, selection):
    details = model_details(base_url, selection.model)
    if "completion" not in details["capabilities"] and details["capabilities"]:
        raise ValueError(f"{selection.model} does not report text completion support")
    for name, value in selection.parameters.items():
        if name not in details["controls"]:
            raise ValueError(f"{selection.model} does not report support for {name}")
        maximum = details["controls"][name].get("maximum")
        if maximum is not None and isinstance(value, (int, float)) and value > maximum:
            raise ValueError(f"{name} exceeds this model's reported limit of {maximum}")
    return details


def delete_model(base_url, name):
    ModelSelection(model=name)
    return ollama_request(base_url, "/api/delete", method="DELETE", body={"model": name})


def pull_model(base_url, name, report):
    ModelSelection(model=name)
    if name.count("/") > 1 or ("/" in name and "." in name.split("/", 1)[0]):
        raise ValueError("Downloads support Ollama registry names only; install other sources in Ollama explicitly")
    require_destination(base_url, NetworkPurpose.MODEL)
    require_destination(DOWNLOAD_ORIGIN, NetworkPurpose.MODEL_DOWNLOAD)
    with httpx.Client(base_url=base_url, trust_env=False, follow_redirects=False,
                      timeout=MODEL_PULL_TIMEOUT_SECONDS) as client:
        with client.stream("POST", "/api/pull", json={"model": name, "stream": True}) as response:
            response.raise_for_status()
            completed = False
            for line in response.iter_lines():
                if not line:
                    continue
                data = json.loads(line)
                if data.get("error"):
                    raise ValueError(str(data["error"])[:500])
                report({key: data[key] for key in ("status", "digest", "total", "completed") if key in data})
                completed = data.get("status") == "success"
            if not completed:
                raise ValueError("Ollama download ended without a success receipt")
    return model_details(base_url, name)


class CatalogParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.names = []
        self.records = []
        self.current = None
        self.badge = False

    def handle_starttag(self, tag, attrs):
        if self.current and tag == "span":
            self.badge = "rounded-md" in dict(attrs).get("class", "").split()
        if tag != "a":
            return
        href = dict(attrs).get("href", "")
        parsed = urlsplit(urljoin(CATALOG_ORIGIN, href))
        if parsed.hostname != "ollama.com" or not parsed.path.startswith("/library/"):
            return
        name = parsed.path.removeprefix("/library/").strip("/")
        if "/" in name or name in self.names:
            return
        try:
            ModelSelection(model=name)
        except ValueError:
            return
        if len(self.names) < MAX_CATALOG_RESULTS:
            self.names.append(name)
            self.current = {"name": name, "url": CATALOG_ORIGIN + "/library/" + name,
                            "advertised_capabilities": [], "advertised_sizes": [], "catalog_text": ""}

    def handle_data(self, data):
        if self.current is None:
            return
        value = data.strip()
        self.current["catalog_text"] += " " + value
        if self.badge and value in {"vision", "tools", "thinking", "embedding", "audio", "cloud"}:
            self.current["advertised_capabilities"].append(value)
        elif self.badge and re.fullmatch(r"\d+(?:\.\d+)?[bm]", value):
            self.current["advertised_sizes"].append(value)

    def handle_endtag(self, tag):
        if tag == "span":
            self.badge = False
        if tag == "a" and self.current:
            self.records.append(self.current)
            self.current = None


def search_catalog(query, *, capability=None, sort=None):
    if capability not in {None, "vision", "tools", "thinking", "embedding"}:
        raise ValueError("Unregistered catalog capability filter")
    if sort not in {None, *CATALOG_SORTS}:
        raise ValueError("Unregistered catalog sort order")
    require_destination(CATALOG_ORIGIN, NetworkPurpose.MODEL_CATALOG)
    # An empty "q" is not the same as an absent one, and "o=popular" is not
    # the same as an absent "o": ollama.com 303-redirects both a
    # present-but-empty query and an explicit-but-default ("popular") sort to
    # its canonical URL without them. Omit each rather than forwarding it, so
    # the request lands on the real 200 page instead of an unfollowed redirect.
    params = {**({"q": query} if query else {}), **({"c": capability} if capability else {}),
              **({"o": sort} if sort and sort != "popular" else {})}
    with httpx.Client(trust_env=False, follow_redirects=False, timeout=MODEL_HTTP_TIMEOUT_SECONDS) as client:
        with client.stream("GET", CATALOG_ORIGIN + "/search", params=params) as response:
            response.raise_for_status()
            chunks, size = [], 0
            for chunk in response.iter_bytes():
                size += len(chunk)
                if size > MAX_CATALOG_BYTES:
                    raise ValueError("Catalog response exceeds its read budget")
                chunks.append(chunk)
    parser = CatalogParser()
    parser.feed(b"".join(chunks).decode("utf-8", errors="replace"))
    return sorted(parser.records, key=lambda item: item["name"])


def _broad_catalog_search(keywords, capability):
    """Merge/dedupe results across every registered keyword and sort order.

    Ollama's search page renders a fixed ~12-20 results per request and
    silently ignores every pagination parameter tried (page/p/offset/cursor/
    n/limit); fanning out across its supported sort orders ("popular",
    "newest") and, where a category has no dedicated capability filter,
    across a small set of related keywords, is the only way to see more than
    one page's worth through its plain HTML search.
    """
    merged = {}
    for keyword in keywords:
        for sort in CATALOG_SORTS:
            for record in search_catalog(keyword, capability=capability, sort=sort):
                merged.setdefault(record["name"], record)
    return sorted(merged.values(), key=lambda item: item["name"])[:MAX_CATALOG_RESULTS]


def _parameter_count_billions(label):
    """Parse a "<number><b|m>" parameter-count label (case-insensitive):
    matches both ollama.com's advertised size badges ("7b") and Ollama's own
    reported `parameter_size` ("8.9B", "270M")."""
    if not label:
        return None
    match = re.fullmatch(r"(\d+(?:\.\d+)?)\s*([bm])", label.strip(), re.IGNORECASE)
    if not match:
        return None
    value, unit = match.groups()
    count = float(value)
    return count / 1000 if unit.lower() == "m" else count


def _installed_weight_ratio(base_url):
    """Empirical MiB-per-billion-parameters ratio from every currently
    installed model's own exact, real reported weight size and parameter
    count -- no architecture or capability restriction, since raw weight
    bytes-per-parameter (i.e. quantization) is unrelated to either. Returns
    (ratio, sample_count), or None when nothing installed can calibrate it.
    Self-improving: installing more/larger models only makes this more exact.
    """
    try:
        installed = installed_models(base_url)
    except Exception:
        return None
    ratios = []
    for record in installed:
        billions = _parameter_count_billions(record.get("details", {}).get("parameter_size", ""))
        size_bytes = record.get("size")
        if not billions or not isinstance(size_bytes, (int, float)) or size_bytes <= 0:
            continue
        ratios.append((size_bytes / (1024 * 1024)) / billions)
    if not ratios:
        return None
    ratios.sort()
    middle = len(ratios) // 2
    median = ratios[middle] if len(ratios) % 2 else (ratios[middle - 1] + ratios[middle]) / 2
    return median, len(ratios)


def _host_memory_ceiling_mib(settings):
    """The same hard, headroom-adjusted total-RAM ceiling
    `model_admission.assess_model` uses to mark a model definitively
    unsupported (as opposed to merely temporarily blocked by transient
    load) -- reusing the resource observation the app already produces on
    every scan, not a new one.
    """
    from persistent_cognition.model_admission import capture_observation
    from persistent_cognition.attention_observation import HOST_MEMORY_RESOURCE_ID
    observation = capture_observation(settings)
    if not observation.healthy:
        return None
    pool = next((item for item in observation.capacities if item.resource_id == HOST_MEMORY_RESOURCE_ID), None)
    if pool is None:
        return None
    return pool.configured_capacity - pool.configured_headroom_units


def _apply_capacity_filter(records, settings):
    """Exclude only catalog families whose every advertised size estimate
    exceeds this host's measured capacity ceiling. The weight estimate comes
    from `_installed_weight_ratio` (self-calibrating from whatever is
    actually installed); the runtime/buffer margin reuses the operator's own
    configured headroom settings, the same fields `assess_model` already
    uses for that purpose. This is a coarse, catalog-time screen -- the
    byte-precise post-install check remains `assess_model`. Absent evidence
    (unparsed size, unmeasurable host, no calibration data yet) never
    excludes a result; it is only ever excluded on positive evidence that it
    cannot fit.
    """
    ceiling = _host_memory_ceiling_mib(settings)
    calibration = _installed_weight_ratio(settings.ollama_url)
    resource_filter = {"ceiling_mib": ceiling,
                        "calibration_mib_per_billion_parameters": round(calibration[0], 1) if calibration else None,
                        "calibration_sample_count": calibration[1] if calibration else 0}
    if ceiling is None or calibration is None:
        for record in records:
            record["resource_check"] = "unavailable"
        return records, resource_filter
    ratio, _ = calibration

    def required_mib(billions):
        weights = billions * ratio
        buffers = max(settings.resources.model_runtime_headroom_min_mib,
                      weights * settings.resources.model_runtime_headroom_percent / 100)
        worker = settings.resources.default_process_memory_mib
        return max(settings.resources.default_llm_process_memory_mib, weights + buffers + worker)

    kept = []
    for record in records:
        sizes = record.get("advertised_sizes") or []
        known = [(size, billions) for size in sizes if (billions := _parameter_count_billions(size)) is not None]
        if not known:
            record["resource_check"] = "unknown_size"
            kept.append(record)
            continue
        infeasible = [size for size, billions in known if required_mib(billions) > ceiling]
        record["resource_filtered_sizes"] = infeasible
        if len(infeasible) < len(known):
            record["resource_check"] = "fits" if not infeasible else "partial_fit"
            kept.append(record)
        else:
            record["resource_check"] = "exceeds_capacity"
    return kept, resource_filter


def search_specialists(category, *, settings=None):
    """Fixed public queries and exact capability filters, never the personal prompt."""
    from persistent_cognition.content_digest import content_digest
    from persistent_cognition.model_admission import TASKS
    if category not in CATALOG_CATEGORIES:
        raise ValueError("Unregistered catalog category")
    spec = CATALOG_CATEGORIES[category]
    records = _broad_catalog_search(spec["keywords"], spec["capability"])
    matches = []
    for record in records:
        advertised = set(record["advertised_capabilities"])
        if category != "embedding" and "embedding" in advertised:
            continue
        if "cloud" in advertised and not record["advertised_sizes"]:
            continue
        if spec["capability"] and spec["capability"] not in advertised:
            continue
        if category == "coding" and not re.search(r"\b(?:code|coding|coder|programming)\b", record["catalog_text"], re.I):
            continue
        verification = "Advertised catalog match; exact tag, capabilities and resource fit require installation and /api/show"
        if category == "base":
            verification = ("Keyword match on model descriptions, not a verified base/foundation flag: ollama.com "
                            "has no such filter. Review the exact tag before choosing.")
        matches.append({**record, "verification": verification})
    resource_filter = None
    if settings is not None:
        matches, resource_filter = _apply_capacity_filter(matches, settings)
    chat_task = spec["chat_task"]
    required_capabilities = TASKS[chat_task]["requires"] if chat_task else ([spec["capability"]] if spec["capability"] else [])
    openai_offer = bool(chat_task and TASKS[chat_task]["implemented"])
    result = {"policy_version": "specialist-search/v2", "task": category, "label": spec["label"],
              "keywords": list(spec["keywords"]), "capability_filter": spec["capability"],
              "required_capabilities": required_capabilities, "catalog_digest": content_digest(records),
              "models": matches, "openai_offer": openai_offer,
              "download_policy": "Only the operator chooses an exact model/tag to download"}
    if resource_filter is not None:
        result["resource_filter"] = resource_filter
    return result


def openai_models(key):
    require_destination(OPENAI_ORIGIN, NetworkPurpose.MODEL)
    if not key:
        raise ValueError("Connect an API key first")
    with httpx.Client(trust_env=False, follow_redirects=False, timeout=MODEL_HTTP_TIMEOUT_SECONDS) as client:
        response = client.get(OPENAI_BASE_URL + "/models", headers={"Authorization": "Bearer " + key})
    if response.is_error:
        raise ValueError(f"OpenAI model listing returned HTTP {response.status_code}; check key and project access")
    # The listing supplies IDs, not a reliable parameter/capability schema.
    return sorted([{"id": item["id"]} for item in response.json().get("data", [])
                   if isinstance(item.get("id"), str)], key=lambda item: item["id"])
