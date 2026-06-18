"""Small JSONL helpers for qr_v2 experiment records."""

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


JsonRecord = Mapping[str, Any]


def dumps_record(record: JsonRecord) -> str:
    return json.dumps(record, sort_keys=True, separators=(",", ":"))


def append_jsonl(path: Path, record: JsonRecord) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(dumps_record(record))
        file.write("\n")


def write_jsonl(path: Path, records: Iterable[JsonRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(dumps_record(record))
            file.write("\n")
