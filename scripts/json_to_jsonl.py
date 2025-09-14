#!/usr/bin/env python3
"""
json_to_jsonl.py

Convert a JSON file containing a list of objects into JSONL (one JSON object per line).

Features
- Picks a list field or top-level array automatically
- Optionally include only selected keys
- Optionally rename keys (e.g., query->question, response->answer)
- Validates input and reports useful errors

Usage examples
- Keep original keys:
  python scripts/json_to_jsonl.py --input datasets/metamathqa.json --output datasets/metamathqa.jsonl

- Keep only query/response:
  python scripts/json_to_jsonl.py --input datasets/metamathqa.json --output datasets/metamathqa.jsonl \
    --include-keys query response

- Rename to question/answer:
  python scripts/json_to_jsonl.py --input datasets/metamathqa.json --output datasets/metamathqa.jsonl \
    --rename query=question --rename response=answer
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List


def _load_items(path: Path) -> List[Dict]:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        raise SystemExit(f"Failed to read JSON from {path}: {e}")

    if isinstance(data, list):
        if not all(isinstance(x, dict) for x in data):
            raise SystemExit("Top-level list must contain objects (dicts).")
        return data
    if isinstance(data, dict):
        # common patterns: {"data": [...]}, {"items": [...]}
        for key in ("data", "items", "examples"):
            val = data.get(key)
            if isinstance(val, list) and all(isinstance(x, dict) for x in val):
                return val
        # if dict has exactly one list-of-dicts value, take it
        list_fields = [v for v in data.values() if isinstance(v, list) and all(isinstance(x, dict) for x in v)]
        if len(list_fields) == 1:
            return list_fields[0]
        raise SystemExit("Could not find a list of objects in the JSON. Provide a JSON array or a dict with a single array of objects.")
    raise SystemExit("Unsupported JSON structure; expected array or dict.")


def _parse_renames(items: Iterable[str]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for s in items:
        if "=" not in s:
            raise SystemExit(f"Invalid --rename '{s}', expected OLD=NEW")
        old, new = s.split("=", 1)
        old, new = old.strip(), new.strip()
        if not old or not new:
            raise SystemExit(f"Invalid --rename '{s}', empty key")
        mapping[old] = new
    return mapping


def main() -> None:
    p = argparse.ArgumentParser(description="Convert JSON array to JSONL")
    p.add_argument("--input", required=True, help="Path to input .json file")
    p.add_argument("--output", required=True, help="Path to output .jsonl file")
    p.add_argument("--include-keys", nargs="*", help="Only include these keys in each object")
    p.add_argument("--rename", action="append", default=[], help="Rename fields OLD=NEW (repeatable)")
    args = p.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    include = set(args.include_keys or [])
    renames = _parse_renames(args.rename)

    items = _load_items(in_path)

    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as w:
            for obj in items:
                if not isinstance(obj, dict):
                    continue
                row = obj.copy()
                # filter
                if include:
                    row = {k: row[k] for k in include if k in row}
                # rename
                for old, new in renames.items():
                    if old in row:
                        row[new] = row.pop(old)
                # write line
                w.write(json.dumps(row, ensure_ascii=False) + "\n")
    except Exception as e:
        raise SystemExit(f"Failed to write JSONL to {out_path}: {e}")

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

