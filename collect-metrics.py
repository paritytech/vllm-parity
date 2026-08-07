#!/usr/bin/env python3
"""Sample vLLM's Prometheus endpoint on a fixed interval, append one JSON object per
sample to a JSONL file.

Prometheus counters are cumulative, which makes a raw scrape awkward to plot and
useless once the server restarts and resets them. Every line written here describes
the interval that just ended instead:

    timestamp         end of the interval, ISO 8601 UTC
    interval_seconds  time since the previous sample, measured on a monotonic clock
    scrape_seconds    how long the /metrics fetch took (itself a load signal)
    gauges            instantaneous values: queue depth, KV cache usage, ...
    counters          change over the interval, not the cumulative total
    histograms        count/sum/mean, quantiles, and `buckets` -- the raw per-bucket
                      counts for the interval. count/sum/mean are exact; the quantiles
                      are as coarse as vLLM's bucket boundaries. Keep the buckets:
                      quantiles cannot be merged across intervals, buckets can, so
                      they are the only way to recover a whole run's distribution
    info              static config; written on the first sample and whenever it changes.
                      Includes `models` (id -> max_model_len) read from /v1/models,
                      because /metrics names no model and states no context limit
    gpus              nvidia-smi readings, when nvidia-smi is available
    baseline          "first" or "restart": no interval data on this line
    error             the scrape failed; gauges/counters/histograms are absent

Counters and histograms with no activity in the interval are omitted rather than
written as zero, so an idle server costs a short line. Read a missing key as zero.

Bucket counts roughly double a busy line, to ~1.5 kB. At one sample a second that is
~120 MB a day, so for a long-running deployment prefer `--interval 5`; nothing in the
capacity analysis depends on second-by-second resolution.

The `model_name` and `engine` labels are aggregated away, since this serves one model
and data-parallel engines are one pool. Every other label is kept, so a series prints
as `vllm:request_success_total{finished_reason=stop}`.
"""

import argparse
import json
import math
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone

DEFAULT_URL = "http://127.0.0.1:9001/metrics"
DEFAULT_OUTPUT = "/workspace/metrics.jsonl"

QUANTILES = (0.5, 0.9, 0.99)
COLLAPSED_LABELS = frozenset({"model_name", "engine"})
STRUCTURAL_LABELS = frozenset({"le", "quantile"})
ESCAPES = {"n": "\n", "\\": "\\", '"': '"'}

GPU_QUERY = (
    "index,utilization.gpu,utilization.memory,memory.used,memory.total,"
    "power.draw,temperature.gpu,clocks.sm"
)
GPU_KEYS = (
    "index",
    "utilization_percent",
    "memory_utilization_percent",
    "memory_used_mib",
    "memory_total_mib",
    "power_watts",
    "temperature_celsius",
    "sm_clock_mhz",
)


# --- Prometheus text exposition parsing ---------------------------------------------


@dataclass
class Family:
    name: str
    kind: str = "untyped"
    samples: list = field(default_factory=list)


def parse_exposition(text):
    """Parse the text format into families, keyed by family name.

    Samples belong to the most recent HELP/TYPE comment: the format requires a family's
    samples to be contiguous, which is far more robust than matching name prefixes.
    """
    families = {}
    current = None

    for line in text.splitlines():
        if not line.strip():
            continue
        if line.startswith("#"):
            parts = line.split(" ", 3)
            if len(parts) < 3 or parts[1] not in ("HELP", "TYPE"):
                continue
            current = families.setdefault(parts[2], Family(name=parts[2]))
            if parts[1] == "TYPE" and len(parts) == 4:
                current.kind = parts[3].strip()
            continue

        sample = parse_sample(line)
        if sample is None:
            continue
        name, labels, value = sample
        family = current
        if family is None or not name.startswith(family.name):
            family = families.setdefault(name, Family(name=name))
        family.samples.append((name, labels, value))

    return families


def parse_sample(line):
    try:
        brace = line.find("{")
        space = line.find(" ")
        if brace != -1 and (space == -1 or brace < space):
            name = line[:brace]
            labels, position = parse_labels(line, brace)
        elif space != -1:
            name, labels, position = line[:space], {}, space
        else:
            return None
        return name, labels, float(line[position:].split()[0])
    except (ValueError, IndexError):
        return None


def parse_labels(line, start):
    """Read the `{...}` block beginning at `start`; return the labels and the index after it."""
    labels = {}
    position = start + 1

    while line[position] != "}":
        if line[position] in ", ":
            position += 1
            continue
        equals = line.index("=", position)
        key = line[position:equals].strip()
        position = line.index('"', equals) + 1
        characters = []
        while line[position] != '"':
            if line[position] == "\\":
                position += 1
                characters.append(ESCAPES.get(line[position], line[position]))
            else:
                characters.append(line[position])
            position += 1
        labels[key] = "".join(characters)
        position += 1

    return labels, position + 1


# --- Snapshots ----------------------------------------------------------------------


@dataclass
class Histogram:
    count: float = 0.0
    total: float = 0.0
    buckets: dict = field(default_factory=dict)


@dataclass
class Snapshot:
    gauges: dict
    counters: dict
    histograms: dict
    info: dict


def series_key(name, labels):
    kept = {
        key: value
        for key, value in labels.items()
        if key not in COLLAPSED_LABELS and key not in STRUCTURAL_LABELS
    }
    if not kept:
        return name
    return name + "{" + ",".join(f"{key}={kept[key]}" for key in sorted(kept)) + "}"


def is_fraction(key):
    return key.split("{", 1)[0].endswith(("_perc", "_ratio", "_rate"))


def build_snapshot(families):
    gauge_series = defaultdict(list)
    counters = defaultdict(float)
    histograms = {}
    info = {}

    for family in families.values():
        if family.name.endswith("_info"):
            for _, labels, _value in family.samples:
                info[family.name] = {
                    key: value
                    for key, value in labels.items()
                    if key not in COLLAPSED_LABELS
                }
            continue

        if family.kind in ("histogram", "summary"):
            for name, labels, value in family.samples:
                histogram = histograms.setdefault(series_key(family.name, labels), Histogram())
                if name.endswith("_bucket"):
                    bound = float(labels.get("le", "inf"))
                    histogram.buckets[bound] = histogram.buckets.get(bound, 0.0) + value
                elif name.endswith("_sum"):
                    histogram.total += value
                elif name.endswith("_count"):
                    histogram.count += value
            continue

        for name, labels, value in family.samples:
            if name.endswith("_created"):
                continue
            key = series_key(name, labels)
            if family.kind == "counter":
                counters[key] += value
            else:
                gauge_series[key].append(value)

    gauges = {
        key: sum(values) / len(values) if is_fraction(key) else sum(values)
        for key, values in gauge_series.items()
    }
    return Snapshot(gauges, dict(counters), histograms, info)


def has_restarted(previous, current):
    before = previous.gauges.get("process_start_time_seconds")
    after = current.gauges.get("process_start_time_seconds")
    if before is not None and after is not None and before != after:
        return True
    return any(
        value < previous.counters.get(key, 0.0) for key, value in current.counters.items()
    )


def counter_deltas(previous, current):
    deltas = {}
    for key, value in sorted(current.counters.items()):
        delta = value - previous.counters.get(key, 0.0)
        if delta > 0:
            deltas[key] = json_number(delta)
    return deltas


def histogram_intervals(previous, current):
    intervals = {}
    for key, histogram in sorted(current.histograms.items()):
        before = previous.histograms.get(key, Histogram())
        count = histogram.count - before.count
        if count <= 0:
            continue
        total = histogram.total - before.total
        interval = {
            "count": json_number(count),
            "sum": json_number(total),
            "mean": json_number(total / count),
        }
        bounds, counts = bucket_deltas(histogram.buckets, before.buckets)
        if sum(counts) > 0:
            interval.update(bucket_quantiles(bounds, counts))
            # The raw buckets are what makes a whole run's distribution recoverable.
            # Per-interval quantiles cannot be merged across intervals; these can.
            interval["buckets"] = {
                bound_label(bound): json_number(count)
                for bound, count in zip(bounds, counts)
                if count > 0
            }
        intervals[key] = interval
    return intervals


def bucket_deltas(current_buckets, previous_buckets):
    """Prometheus buckets are cumulative (`le`); return per-bucket counts for the interval."""
    bounds = sorted(current_buckets)
    if not bounds:
        return [], []
    cumulative = [
        max(0.0, current_buckets[bound] - previous_buckets.get(bound, 0.0)) for bound in bounds
    ]
    counts = [cumulative[0]] + [
        max(0.0, cumulative[index] - cumulative[index - 1]) for index in range(1, len(bounds))
    ]
    return bounds, counts


def bound_label(bound):
    if math.isinf(bound):
        return "inf"
    return str(int(bound)) if float(bound).is_integer() else repr(bound)


def bucket_quantiles(bounds, counts):
    total = sum(counts)
    return {
        f"p{round(quantile * 100)}": json_number(quantile_of(bounds, counts, total, quantile))
        for quantile in QUANTILES
    }


def quantile_of(bounds, counts, total, quantile):
    """Interpolate a quantile within the bucket it falls into, as histogram_quantile does.

    A quantile landing in the open-ended top bucket is reported as its lower bound, i.e.
    "at least this", because the observations there have no upper bound to interpolate to.
    """
    target = quantile * total
    seen = 0.0
    lower = 0.0

    for bound, count in zip(bounds, counts):
        if count > 0 and seen + count >= target:
            if math.isinf(bound):
                return lower
            return lower + (bound - lower) * (target - seen) / count
        seen += count
        lower = bound

    return bounds[-1] if bounds else None


def json_number(value):
    if value is None or not math.isfinite(value):
        return None
    if float(value).is_integer():
        return int(value)
    return round(value, 6)


# --- GPU ----------------------------------------------------------------------------


def read_gpus(timeout):
    """Per-GPU readings, or None if nvidia-smi is unavailable or misbehaving."""
    try:
        completed = subprocess.run(
            ["nvidia-smi", f"--query-gpu={GPU_QUERY}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None

    gpus = []
    for line in completed.stdout.strip().splitlines():
        values = [value.strip() for value in line.split(",")]
        if len(values) == len(GPU_KEYS):
            gpus.append(dict(zip(GPU_KEYS, (parse_gpu_value(value) for value in values))))
    return gpus


def parse_gpu_value(value):
    try:
        return json_number(float(value))
    except ValueError:
        return None


# --- Collection ---------------------------------------------------------------------


class Collector:
    def __init__(self, url, timeout, collect_gpus):
        self.url = url
        self.models_url = urllib.parse.urljoin(url, "/v1/models")
        self.timeout = timeout
        self.collect_gpus = collect_gpus
        self.previous = None
        self.previous_time = None
        self.info = None
        self.models = {}

    def fetch(self):
        with urllib.request.urlopen(self.url, timeout=self.timeout) as response:
            return response.read().decode("utf-8", "replace")

    def fetch_models(self):
        """/metrics names no model and states no context limit -- both live on /v1/models,
        and the context limit is what capacity planning is measured against."""
        try:
            with urllib.request.urlopen(self.models_url, timeout=self.timeout) as response:
                served = json.loads(response.read().decode("utf-8", "replace"))["data"]
        except Exception:
            return {}
        return {
            entry["id"]: str(entry["max_model_len"])
            for entry in served
            if entry.get("max_model_len") is not None
        }

    def sample(self):
        record = {"timestamp": now_iso()}
        started = time.monotonic()

        try:
            snapshot = build_snapshot(parse_exposition(self.fetch()))
            scraped = time.monotonic()
            record.update(self.summarise(snapshot, scraped))
            record["scrape_seconds"] = json_number(scraped - started)
        except Exception as error:  # neither a failed scrape nor a bug here may end the run
            record["error"] = f"{type(error).__name__}: {error}"

        if self.collect_gpus:
            record["gpus"] = read_gpus(self.timeout)
        return record

    def summarise(self, snapshot, scraped):
        fields = {}
        fresh = self.previous is None or has_restarted(self.previous, snapshot)
        if fresh:
            self.models = self.fetch_models()
        if self.models:
            snapshot.info["models"] = self.models

        if self.previous is None:
            fields["baseline"] = "first"
        elif fresh:
            fields["baseline"] = "restart"
        else:
            fields["interval_seconds"] = json_number(scraped - self.previous_time)
            fields["counters"] = counter_deltas(self.previous, snapshot)
            fields["histograms"] = histogram_intervals(self.previous, snapshot)

        fields["gauges"] = {
            key: json_number(value) for key, value in sorted(snapshot.gauges.items())
        }
        if snapshot.info and snapshot.info != self.info:
            fields["info"] = snapshot.info
            self.info = snapshot.info

        self.previous, self.previous_time = snapshot, scraped
        return fields


def now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


# --- Entry point --------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--url", default=DEFAULT_URL, help=f"default: {DEFAULT_URL}")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help=f"default: {DEFAULT_OUTPUT}")
    parser.add_argument("--interval", type=float, default=1.0, help="seconds between samples")
    parser.add_argument("--timeout", type=float, default=5.0, help="seconds before a scrape gives up")
    parser.add_argument("--no-gpu", action="store_true", help="skip nvidia-smi")
    return parser.parse_args()


def main():
    args = parse_args()

    directory = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(directory, exist_ok=True)

    collect_gpus = not args.no_gpu and read_gpus(args.timeout) is not None
    collector = Collector(args.url, args.timeout, collect_gpus)

    stop = threading.Event()
    for received in (signal.SIGINT, signal.SIGTERM):
        signal.signal(received, lambda *_: stop.set())

    print(
        f"Sampling {args.url} every {args.interval}s into {args.output}"
        f"{'' if collect_gpus else ' (no nvidia-smi)'}",
        file=sys.stderr,
        flush=True,
    )

    last_error = None
    next_tick = time.monotonic()

    with open(args.output, "a", buffering=1) as stream:
        while not stop.is_set():
            record = collector.sample()
            stream.write(json.dumps(record) + "\n")

            error = record.get("error")
            if error != last_error:
                print(error or "Scrape recovered.", file=sys.stderr, flush=True)
                last_error = error

            next_tick += args.interval
            delay = next_tick - time.monotonic()
            if delay > 0:
                stop.wait(delay)
            else:
                next_tick = time.monotonic()

    return 0


if __name__ == "__main__":
    sys.exit(main())
