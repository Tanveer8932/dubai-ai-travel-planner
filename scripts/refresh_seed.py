"""Enrich curated seed coordinates / opening hours / wheelchair tags from OpenStreetMap.

Dry run by default (prints what would change). ``--write`` updates the JSON files in
place and re-validates them. Respects the Nominatim policy: ≤ 1 request/second,
descriptive ``HTTP_USER_AGENT`` (set a contact address in ``.env``), cached responses.

    python scripts/refresh_seed.py --file attractions.json --limit 5
    python scripts/refresh_seed.py --write
"""

from __future__ import annotations

import argparse
import json
import sys

import dubai_planner
from dubai_planner.data import seed
from dubai_planner.data.enrichment import EnrichStatus, enrich_record, nominatim_http

VENUE_FILES = (seed.ATTRACTIONS, seed.HOTELS, seed.RESTAURANTS)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--file",
        choices=VENUE_FILES,
        action="append",
        help="seed file(s) to refresh (default: all venue files)",
    )
    parser.add_argument("--limit", type=int, default=None, help="max records per file")
    parser.add_argument("--write", action="store_true", help="write changes back to disk")
    args = parser.parse_args(argv)

    settings = dubai_planner.bootstrap()
    if settings.http_offline:
        print("HTTP_OFFLINE=true: nothing to do.")
        return 1
    areas = {a.id: a.name for a in seed.load_catalog().areas}
    http = nominatim_http()
    totals: dict[str, int] = {}
    for filename in args.file or VENUE_FILES:
        path = settings.seed_dir / filename
        doc = json.loads(path.read_text(encoding="utf-8"))
        items = doc["items"]
        for i, record in enumerate(items[: args.limit]):
            area = areas.get(record["area"], "Dubai")
            updated, outcome = enrich_record(record, f"{area}, United Arab Emirates", http)
            items[i] = updated
            totals[outcome.status] = totals.get(outcome.status, 0) + 1
            shift = f" ({outcome.shift_km} km)" if outcome.shift_km is not None else ""
            print(f"{filename}: {outcome.venue_id}: {outcome.status}{shift}")
        if args.write:
            path.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.write:
        seed.clear_cache()
        seed.load_catalog(settings.seed_dir)  # re-validate everything; raises if invalid
        print("Seed files updated and re-validated.")
    print("Summary:", ", ".join(f"{k}={v}" for k, v in sorted(totals.items())) or "nothing")
    return 0 if totals.get(EnrichStatus.UNAVAILABLE, 0) == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
