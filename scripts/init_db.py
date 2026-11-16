"""Create or migrate the application SQLite database (``DB_PATH``, default data/app.db).

python scripts/init_db.py
"""

from __future__ import annotations

import sys

import dubai_planner
from dubai_planner.data.seed import load_catalog
from dubai_planner.storage.repository import init_db


def main() -> int:
    settings = dubai_planner.bootstrap()
    repo = init_db(settings)
    print(f"Database ready: {repo.db_path} (schema version {repo.schema_version()})")
    catalog = load_catalog(settings.seed_dir)
    print(
        f"Seed data valid: {len(catalog.attractions)} attractions, {len(catalog.hotels)} "
        f"hotels, {len(catalog.restaurants)} restaurants, {len(catalog.areas)} areas"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
