#!/usr/bin/env python
"""Download (or verify) the data files the test suite needs.

Files are fetched into the pooch cache (``pooch.os_cache("astro-brutus")``)
and symlinked into ``data/DATAFILES``. With a warm cache nothing is
downloaded: pooch only hash-checks the cached files.

Harvard Dataverse intermittently answers bursts of concurrent requests from
CI runners with HTTP 403, so every fetch is retried with exponential backoff
(pooch's own retries are only ~1-3 s apart, too short to ride that out).

Usage: python .github/scripts/fetch_test_data.py [--no-retry]
"""

import os
import sys
import time

import pooch

from brutus.data import (
    fetch_dustmaps,
    fetch_grids,
    fetch_isos,
    fetch_nns,
    fetch_offsets,
    fetch_tracks,
)

DATA_DIR = "data/DATAFILES"
WAITS = (30, 60, 120, 240)  # seconds between attempts (~7.5 min total)

# (label, fetcher, kwargs, required)
FILES = [
    ("MIST grid (mist_v9)", fetch_grids, {"grid": "mist_v9"}, True),
    ("isochrones", fetch_isos, {}, True),
    ("EEP tracks", fetch_tracks, {}, True),
    ("neural network", fetch_nns, {}, True),
    ("dust map (bayestar19)", fetch_dustmaps, {}, False),
    ("photometric offsets (mist_v9)", fetch_offsets, {"grid": "mist_v9"}, False),
]


def fetch_with_backoff(label, fetcher, kwargs, waits):
    for attempt in range(len(waits) + 1):
        try:
            return fetcher(target_dir=DATA_DIR, **kwargs)
        except Exception as e:  # noqa: BLE001 - network/HTTP errors vary
            if attempt == len(waits):
                raise
            print(
                f"   ! {label}: attempt {attempt + 1} failed ({e}); "
                f"retrying in {waits[attempt]} s",
                flush=True,
            )
            time.sleep(waits[attempt])


def main():
    waits = () if "--no-retry" in sys.argv else WAITS
    cache_dir = pooch.os_cache("astro-brutus")
    print(f"Cache directory: {cache_dir} (exists: {os.path.exists(cache_dir)})")
    os.makedirs(DATA_DIR, exist_ok=True)

    failed_required = []
    for label, fetcher, kwargs, required in FILES:
        print(f"-> {label}", flush=True)
        try:
            path = fetch_with_backoff(label, fetcher, kwargs, waits)
            print(f"   ok: {path}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"   FAILED: {e}", flush=True)
            if required:
                failed_required.append(label)

    total = 0.0
    if os.path.exists(cache_dir):
        for f in sorted(os.listdir(cache_dir)):
            size = os.path.getsize(os.path.join(cache_dir, f)) / 1024**2
            total += size
            print(f"   cache: {f} ({size:.1f} MB)")
    print(f"Total cache size: {total / 1024:.2f} GB")

    if failed_required:
        print(f"ERROR: required data files unavailable: {failed_required}")
        sys.exit(1)


if __name__ == "__main__":
    main()
