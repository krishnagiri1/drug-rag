"""Download prescription drug labels from the openFDA drug label API.

openFDA (https://open.fda.gov/apis/drug/label/) serves structured product labels as JSON.
Each section (indications, contraindications, ...) is an array of strings. Many products
share a label (repackagers, generics), so we keep one label per generic name: the most
recently updated one.

No API key is needed for this volume (~50 requests). With FDA_API_KEY set, the daily
limit is much higher.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Iterable, Iterator, Optional

from .config import LABELS_PATH, SECTIONS

API = "https://api.fda.gov/drug/label.json"
PAGE = 100
MAX_SKIP = 25_000  # openFDA's skip ceiling

SEARCH_STRICT = ('openfda.product_type:"HUMAN PRESCRIPTION DRUG" AND _exists_:indications_and_usage '
                 'AND _exists_:contraindications AND _exists_:openfda.generic_name')
SEARCH_LOOSE = 'openfda.product_type:"HUMAN PRESCRIPTION DRUG"'


def _get(params: dict, retries: int = 5) -> dict:
    key = os.environ.get("FDA_API_KEY")
    if key:
        params = {**params, "api_key": key}
    url = f"{API}?{urllib.parse.urlencode(params)}"
    for i in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code == 404:  # openFDA returns 404 for "no results"
                return {"results": []}
            if e.code in (429, 500, 502, 503, 504) and i < retries - 1:
                time.sleep(2 * (2**i))
                continue
            raise
        except urllib.error.URLError:
            if i < retries - 1:
                time.sleep(2 * (2**i))
                continue
            raise
    return {"results": []}


def iter_raw_labels(max_records: int, search: str = SEARCH_STRICT, sleep: float = 0.3) -> Iterator[dict]:
    """Page through the API (limit/skip)."""
    skip = 0
    while skip < min(max_records, MAX_SKIP):
        data = _get({"search": search, "limit": PAGE, "skip": skip})
        results = data.get("results") or []
        if not results:
            return
        yield from results
        skip += len(results)
        time.sleep(sleep)  # stay well under the unauthenticated rate limit


def clean_text(text: str) -> str:
    text = text.replace("•", "-").replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s*\n\s*", "\n", text)
    return text.strip()


def normalize_label(raw: dict) -> Optional[dict]:
    """Keep identifiers, names and the sections we index. Returns None if unusable."""
    ofda = raw.get("openfda") or {}
    generic = (ofda.get("generic_name") or [""])[0].strip()
    if not generic:
        return None
    sections = {}
    for key in SECTIONS:
        parts = raw.get(key)
        if parts:
            text = clean_text(" ".join(parts) if isinstance(parts, list) else str(parts))
            if len(text) > 40:
                sections[key] = text
    if "indications_and_usage" not in sections or len(sections) < 3:
        return None
    return {
        "set_id": raw.get("set_id") or raw.get("id"),
        "label_id": raw.get("id"),
        "effective_time": raw.get("effective_time", ""),
        "generic_name": generic.title(),
        "brand_name": (ofda.get("brand_name") or [""])[0].strip().title(),
        "manufacturer": (ofda.get("manufacturer_name") or [""])[0].strip(),
        "route": ", ".join(ofda.get("route") or []),
        "sections": sections,
    }


def dedupe_by_generic(labels: Iterable[dict]) -> list[dict]:
    """One label per generic name: the most recently updated one."""
    best: dict[str, dict] = {}
    for lab in labels:
        k = lab["generic_name"].lower()
        if k not in best or lab["effective_time"] > best[k]["effective_time"]:
            best[k] = lab
    return sorted(best.values(), key=lambda d: d["generic_name"])


def download_labels(n_labels: int = 1200, max_records: int = 8000, out: str | Path = LABELS_PATH,
                    progress: bool = True) -> list[dict]:
    """Fetch labels until `n_labels` unique generics are collected (or `max_records` scanned)."""
    best: dict[str, dict] = {}
    scanned = 0
    search = SEARCH_STRICT
    for attempt in range(2):
        try:
            for raw in iter_raw_labels(max_records, search=search):
                scanned += 1
                lab = normalize_label(raw)
                if lab:
                    k = lab["generic_name"].lower()
                    if k not in best or lab["effective_time"] > best[k]["effective_time"]:
                        best[k] = lab
                if progress and scanned % 500 == 0:
                    print(f"  scanned {scanned} labels, {len(best)} unique generics")
                if len(best) >= n_labels:
                    break
            break
        except urllib.error.HTTPError as e:
            if e.code == 400 and attempt == 0:  # query syntax not accepted: fall back, filter locally
                search = SEARCH_LOOSE
                continue
            raise
    labels = sorted(best.values(), key=lambda d: d["generic_name"])[:n_labels]
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for lab in labels:
            f.write(json.dumps(lab, ensure_ascii=False) + "\n")
    if progress:
        print(f"Saved {len(labels)} labels (scanned {scanned}) to {out}")
    return labels


def load_labels(path: str | Path = LABELS_PATH) -> list[dict]:
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
