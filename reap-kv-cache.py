#!/usr/bin/env python3

import argparse
import os
import stat
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import serve

GIB = 1024**3
DEFAULT_CACHE_DIR = Path("/workspace/kv_cache")
DEFAULT_INTERVAL_SECONDS = 60

def gib(byte_count):
    return byte_count // GIB

def free_bytes(cache_dir):
    try:
        status = os.statvfs(cache_dir)
    except OSError:
        return None
    return status.f_bavail * status.f_frsize

def blocks_by_access_time(cache_dir):
    blocks = []
    for path in Path(cache_dir).rglob("*.bin"):
        try:
            status = os.stat(path, follow_symlinks=False)
        except OSError:
            continue
        if stat.S_ISREG(status.st_mode):
            blocks.append((status.st_atime, status.st_size, path))
    blocks.sort()
    return blocks

def reap(cache_dir, min_free_bytes):
    free = free_bytes(cache_dir)
    if free is None:
        return
    needed = min_free_bytes - free
    if needed <= 0:
        return

    reclaimed = 0
    count = 0
    for _, size, path in blocks_by_access_time(cache_dir):
        try:
            path.unlink()
        except OSError:
            continue
        reclaimed += size
        count += 1
        if reclaimed >= needed:
            break

    if count == 0:
        print(f"Nothing to reap: free={gib(free)} GiB, minimum free={gib(min_free_bytes)} GiB", flush=True)
        return

    print(f"Reaped: blocks={count}, reclaimed={gib(reclaimed)} GiB, free before={gib(free)} GiB, minimum free={gib(min_free_bytes)} GiB", flush=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cache_dir", nargs="?", type=Path, default=DEFAULT_CACHE_DIR, help="root of the disk KV cache tier (default: %(default)s)")
    args = parser.parse_args()

    min_free_bytes = serve.kv_cache_min_free_gib(args.cache_dir) * GIB
    interval = float(os.environ.get("KV_CACHE_REAP_INTERVAL", DEFAULT_INTERVAL_SECONDS))

    print(f"Reaper for '{args.cache_dir}': interval={interval:g}s, minimum free={gib(min_free_bytes)} GiB", flush=True)
    while True:
        reap(args.cache_dir, min_free_bytes)
        time.sleep(interval)

if __name__ == "__main__":
    main()
