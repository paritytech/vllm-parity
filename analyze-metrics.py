#!/usr/bin/env python3
"""Turn a collect-metrics.py JSONL file into a single self-contained HTML report.

    python3 analyze-metrics.py /workspace/metrics.jsonl --output report.html

The output references nothing external -- SVG charts, CSS and a little JavaScript are
inlined -- so it can be copied off the box and opened anywhere, offline.

Counters in the JSONL are totals over intervals that are only nominally one second, so
everything plotted here is divided by the interval it actually covered and shown as a
rate. Gaps, failed scrapes and vLLM restarts break the lines rather than being bridged,
because a straight line drawn across an outage is a lie about it.

Long runs are bucketed down to a fixed number of points. Bucketing happens once, over
record indices, so every chart shares an x position and the crosshair means the same
instant everywhere. Where a bucket covers more than one sample the chart draws a min-max
band behind the mean, so a spike that averaging would swallow stays visible.
"""

import argparse
import html
import json
import math
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime

MAX_BUCKETS = 720

WIDTH = 1080
PLOT_HEIGHT = 168
MARGIN_TOP = 10
MARGIN_RIGHT = 88
MARGIN_BOTTOM = 24
MARGIN_LEFT = 60
HEIGHT = MARGIN_TOP + PLOT_HEIGHT + MARGIN_BOTTOM
PLOT_LEFT = MARGIN_LEFT
PLOT_RIGHT = WIDTH - MARGIN_RIGHT
PLOT_BOTTOM = MARGIN_TOP + PLOT_HEIGHT

REQUEST_SUCCESS = "vllm:request_success_total"
MAX_REASON_SLOTS = 4
LABEL_CLEARANCE = 14
DURATION_UNITS = ((1.0, "s"), (1e3, "ms"), (1e6, "µs"))


# --- Loading ------------------------------------------------------------------------


def load_records(path):
    """Read the JSONL, tolerating a torn final line from a collector still writing."""
    records = []
    with open(path) as stream:
        for line in stream:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if not records:
        raise SystemExit(f"{path}: no usable records")

    origin = parse_timestamp(records[0])
    for record in records:
        record["_elapsed"] = (parse_timestamp(record) - origin).total_seconds()
    return records


def parse_timestamp(record):
    return datetime.fromisoformat(record["timestamp"].replace("Z", "+00:00"))


# --- Series extraction --------------------------------------------------------------


def gauge(key):
    return lambda record: record.get("gauges", {}).get(key)


def counter_rate(key):
    def extract(record):
        interval = record.get("interval_seconds")
        if not interval:
            return None
        return record.get("counters", {}).get(key, 0.0) / interval

    return extract


def summed_rate(keys):
    def extract(record):
        interval = record.get("interval_seconds")
        if not interval:
            return None
        counters = record.get("counters", {})
        return sum(counters.get(key, 0.0) for key in keys) / interval

    return extract


def histogram_stat(key, statistic):
    """A latency with no observations in the interval is absent, not zero: nothing was
    measured, so the line should break rather than drop to the floor."""

    def extract(record):
        if "histograms" not in record:
            return None
        return record["histograms"].get(key, {}).get(statistic)

    return extract


def counter_ratio(numerator, denominator, factor=100.0):
    def extract(record):
        counters = record.get("counters")
        if counters is None:
            return None
        total = counters.get(denominator, 0.0)
        if not total:
            return None
        return factor * counters.get(numerator, 0.0) / total

    return extract


def gpu_stat(field_name, aggregate=None):
    aggregate = aggregate or (lambda values: sum(values) / len(values))

    def extract(record):
        values = [
            gpu[field_name] for gpu in record.get("gpus") or [] if gpu.get(field_name) is not None
        ]
        return aggregate(values) if values else None

    return extract


def scaled(extract, factor):
    def rescale(record):
        value = extract(record)
        return None if value is None else value * factor

    return rescale


def scrape_seconds(record):
    return record.get("scrape_seconds")


# --- Chart definitions --------------------------------------------------------------


@dataclass
class SeriesSpec:
    name: str
    extract: object


@dataclass
class ChartSpec:
    key: str
    title: str
    note: str
    unit: str
    series: list
    duration: bool = False


@dataclass
class Series:
    name: str
    slot: int
    means: list = field(default_factory=list)
    lows: list = field(default_factory=list)
    highs: list = field(default_factory=list)


@dataclass
class Chart:
    key: str
    title: str
    note: str
    unit: str
    series: list
    scale: float
    ticks: list
    top: float
    decimals: int


def finished_reason_series(records):
    """Reasons come from the data, so fix their order alphabetically: a series must not
    change colour just because a run happened to see a different mix of them."""
    prefix = f"{REQUEST_SUCCESS}{{finished_reason="
    reasons = sorted(
        {
            key[len(prefix) : -1]
            for record in records
            for key in record.get("counters", {})
            if key.startswith(prefix)
        }
    )

    def series_for(reason):
        return SeriesSpec(reason, counter_rate(f"{prefix}{reason}}}"))

    if len(reasons) <= MAX_REASON_SLOTS:
        return [series_for(reason) for reason in reasons]

    # Never invent a fifth hue: everything past the cap folds into one "other" series.
    head, tail = reasons[: MAX_REASON_SLOTS - 1], reasons[MAX_REASON_SLOTS - 1 :]
    folded = summed_rate([f"{prefix}{reason}}}" for reason in tail])
    return [series_for(reason) for reason in head] + [SeriesSpec("other", folded)]


def chart_specs(records):
    return [
        # Prefill and decode share a unit but not a scale -- a single long prompt prefills
        # at ~100k tokens/s while decode runs at ~1k -- so one axis would flatten decode
        # into the floor. Two charts, never two y-scales.
        ChartSpec(
            "generation",
            "Generation throughput",
            "Output tokens per second: the decode work a caller is waiting on.",
            "tokens/s",
            [SeriesSpec("Generation", counter_rate("vllm:generation_tokens_total"))],
        ),
        ChartSpec(
            "prefill",
            "Prompt throughput",
            "Prompt tokens processed per second. Spikes are long prompts being prefilled.",
            "tokens/s",
            [SeriesSpec("Prompt", counter_rate("vllm:prompt_tokens_total"))],
        ),
        ChartSpec(
            "finished",
            "Requests finished",
            "Completions per second, split by why each request ended.",
            "requests/s",
            finished_reason_series(records),
        ),
        ChartSpec(
            "concurrency",
            "Concurrency",
            "Requests waiting is the signal that the GPU is saturated rather than merely slow.",
            "requests",
            [
                SeriesSpec("Running", gauge("vllm:num_requests_running")),
                SeriesSpec("Waiting", gauge("vllm:num_requests_waiting")),
            ],
        ),
        ChartSpec(
            "ttft",
            "Time to first token",
            "Arrival to first token, queueing included.",
            "s",
            [
                SeriesSpec("Mean", histogram_stat("vllm:time_to_first_token_seconds", "mean")),
                SeriesSpec("p90", histogram_stat("vllm:time_to_first_token_seconds", "p90")),
            ],
            duration=True,
        ),
        ChartSpec(
            "itl",
            "Inter-token latency",
            "Gap between successive output tokens once generation is under way.",
            "s",
            [
                SeriesSpec("Mean", histogram_stat("vllm:inter_token_latency_seconds", "mean")),
                SeriesSpec("p90", histogram_stat("vllm:inter_token_latency_seconds", "p90")),
            ],
            duration=True,
        ),
        ChartSpec(
            "queue",
            "Queue time",
            "Time spent waiting for a scheduler slot, before any work began.",
            "s",
            [SeriesSpec("Mean", histogram_stat("vllm:request_queue_time_seconds", "mean"))],
            duration=True,
        ),
        ChartSpec(
            "e2e",
            "End-to-end latency",
            "Arrival to final token, across the whole request.",
            "s",
            [
                SeriesSpec("Mean", histogram_stat("vllm:e2e_request_latency_seconds", "mean")),
                SeriesSpec("p90", histogram_stat("vllm:e2e_request_latency_seconds", "p90")),
            ],
            duration=True,
        ),
        ChartSpec(
            "kv",
            "KV cache usage",
            "Cache pressure. Sustained high usage is what precedes preemption.",
            "%",
            [SeriesSpec("KV cache", scaled(gauge("vllm:kv_cache_usage_perc"), 100.0))],
        ),
        ChartSpec(
            "prefix",
            "Prefix cache hit rate",
            "Share of queried prompt blocks already cached, and so not processed again.",
            "%",
            [
                SeriesSpec(
                    "Hit rate",
                    counter_ratio(
                        "vllm:prefix_cache_hits_total", "vllm:prefix_cache_queries_total"
                    ),
                )
            ],
        ),
        ChartSpec(
            "preemptions",
            "Preemptions",
            "Requests evicted and recomputed. This is what a too-small KV cache looks like.",
            "/s",
            [SeriesSpec("Preemptions", counter_rate("vllm:num_preemptions_total"))],
        ),
        ChartSpec(
            "gpu",
            "GPU occupancy",
            "Shader and memory-bandwidth occupancy, averaged over GPUs. vLLM cannot report this.",
            "%",
            [
                SeriesSpec("Compute", gpu_stat("utilization_percent")),
                SeriesSpec("Memory", gpu_stat("memory_utilization_percent")),
            ],
        ),
        ChartSpec(
            "power",
            "GPU power",
            "Total board draw across all GPUs.",
            "W",
            [SeriesSpec("Power", gpu_stat("power_watts", aggregate=sum))],
        ),
        ChartSpec(
            "scrape",
            "Scrape duration",
            "How long each /metrics fetch took, which tracks API-server responsiveness.",
            "s",
            [SeriesSpec("Scrape", scrape_seconds)],
            duration=True,
        ),
    ]


# --- Bucketing ----------------------------------------------------------------------


def bucket_bounds(count, limit):
    if count <= limit:
        return [(index, index + 1) for index in range(count)]
    return [
        (round(index * count / limit), round((index + 1) * count / limit))
        for index in range(limit)
    ]


def bucket_series(records, bounds, extract):
    means, lows, highs = [], [], []
    for start, end in bounds:
        values = [value for value in map(extract, records[start:end]) if value is not None]
        means.append(sum(values) / len(values) if values else None)
        lows.append(min(values) if values else None)
        highs.append(max(values) if values else None)
    return means, lows, highs


def bucket_axis(records, bounds):
    """One x position per bucket, plus a flag where the clock jumped -- a collector that
    was stopped must show as a gap, not as a line drawn straight across the missing time."""
    positions = [
        sum(record["_elapsed"] for record in records[start:end]) / (end - start)
        for start, end in bounds
    ]
    stamps = [records[start]["timestamp"] for start, _ in bounds]

    steps = sorted(second - first for first, second in zip(positions, positions[1:]))
    typical = steps[len(steps) // 2] if steps else 0.0
    breaks = [False] + [
        typical > 0 and (second - first) > max(3 * typical, typical + 5)
        for first, second in zip(positions, positions[1:])
    ]
    return positions, stamps, breaks


# --- Scales -------------------------------------------------------------------------


def nice_ticks(top, count=4):
    """Round tick values, always anchored at zero so line heights stay comparable."""
    if top <= 0:
        return [0.0, 1.0], 1.0
    rough = top / count
    magnitude = 10 ** math.floor(math.log10(rough))
    step = next(
        multiple * magnitude
        for multiple in (1, 2, 2.5, 5, 10)
        if multiple * magnitude >= rough - 1e-12
    )
    ceiling = math.ceil(top / step) * step
    return [index * step for index in range(int(round(ceiling / step)) + 1)], step


def build_chart(spec, records, bounds):
    series = []
    for slot, member in enumerate(spec.series, start=1):
        means, lows, highs = bucket_series(records, bounds, member.extract)
        if any(value is not None for value in means):
            series.append(Series(member.name, slot, means, lows, highs))

    if not series:
        return None

    peak = max(
        (value for member in series for value in member.highs if value is not None), default=0.0
    )
    if peak <= 0:
        # Flat zero is one fact, and the stat tiles already state it. A chart would be
        # 200px of empty grid saying the same thing.
        return None

    scale, unit = 1.0, spec.unit
    if spec.duration:
        # Pick the unit that puts the peak above 1, so the axis never reads "0.080 ms".
        scale, unit = next(
            (step for step in DURATION_UNITS if peak * step[0] >= 1.0), DURATION_UNITS[-1]
        )

    ticks, step = nice_ticks(peak * scale)
    decimals = max(0, min(3, 1 - math.floor(math.log10(step))))
    return Chart(spec.key, spec.title, spec.note, unit, series, scale, ticks, ticks[-1], decimals)


# --- Distributions ------------------------------------------------------------------


def parse_bound(text):
    return math.inf if text == "inf" else float(text)


def distribution(records, key):
    """Sum the per-interval bucket counts into the whole run's distribution.

    This is why the collector keeps raw buckets: per-interval quantiles cannot be
    merged, so without these a run-level p99 simply is not recoverable.
    """
    totals = {}
    for record in records:
        buckets = record.get("histograms", {}).get(key, {}).get("buckets", {})
        for bound, count in buckets.items():
            edge = parse_bound(bound)
            totals[edge] = totals.get(edge, 0.0) + count
    bounds = sorted(totals)
    return bounds, [totals[bound] for bound in bounds]


def distribution_quantile(bounds, counts, quantile):
    total = sum(counts)
    if total <= 0:
        return None
    return quantile_of(bounds, counts, total, quantile)


def quantile_of(bounds, counts, total, quantile):
    """Interpolate within the bucket the quantile falls into, as histogram_quantile does.

    A quantile in the open-ended top bucket reports that bucket's lower bound -- "at
    least this" -- since there is no upper bound to interpolate towards.
    """
    target = quantile * total
    seen = 0.0
    lower = 0.0
    for bound, count in zip(bounds, counts):
        if count > 0 and seen + count >= target:
            return lower if math.isinf(bound) else lower + (bound - lower) * (target - seen) / count
        seen += count
        lower = bound
    return None


def largest_observed(bounds, counts):
    """Describe the highest bucket that saw anything.

    Buckets give a range, not a value, so this reads "at most 200K" rather than
    claiming a longest prompt that was never measured. The open-ended top bucket is
    the one case where only a lower bound is known.
    """
    for index in reversed(range(len(bounds))):
        if counts[index] > 0:
            if math.isinf(bounds[index]):
                return f"> {format_tokens(bounds[index - 1])}" if index else "any"
            return f"≤ {format_tokens(bounds[index])}"
    return "--"


# --- Capacity -----------------------------------------------------------------------


CONTEXT_STEPS = (1024, 4096, 16384, 65536, 262144, 1048576)


def run_info(records):
    return next((record["info"] for record in records if record.get("info")), {})


def kv_cache_tokens(info):
    size = info.get("vllm:cache_config_info", {}).get("kv_cache_size_tokens")
    try:
        return int(float(size))
    except (TypeError, ValueError):
        return None


def context_limit(info):
    limits = [int(value) for value in info.get("models", {}).values() if value.isdigit()]
    return max(limits) if limits else None


def capacity_rows(capacity, limit):
    """How many requests of a given length fit in the KV cache at once.

    A floor, not a ceiling: prefix caching lets requests share blocks, so shared
    system prompts only ever raise this.
    """
    lengths = sorted({step for step in CONTEXT_STEPS if step <= limit} | {limit})
    return [
        (length, capacity // length if capacity >= length else 0, capacity / length)
        for length in lengths
    ]


# --- Load profile -------------------------------------------------------------------


CONCURRENCY_BINS = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256)


def concurrency_bin(running):
    chosen = CONCURRENCY_BINS[0]
    for edge in CONCURRENCY_BINS:
        if running >= edge:
            chosen = edge
    return chosen


def load_profile(records):
    """Group samples by how many requests were running, so throughput can be read
    against load rather than against time. This is the shape that says where the server
    saturates and what each caller experiences on the way there.

    Two measures, because the obvious one is confounded. Aggregate generation rate drops
    whenever long prompts are being prefilled -- prefill produces no output tokens -- so
    a dip can mean "long prompts", not "saturated". Decode speed comes from inter-token
    latency, which is only measured between output tokens, so prefill cannot depress it.
    """
    bins = {}
    for record in records:
        interval = record.get("interval_seconds")
        running = record.get("gauges", {}).get("vllm:num_requests_running")
        if not interval or not running or running < 1:
            continue
        entry = bins.setdefault(concurrency_bin(running), {"rates": [], "decode": []})

        rate = record.get("counters", {}).get("vllm:generation_tokens_total", 0.0) / interval
        if rate > 0:
            entry["rates"].append(rate)

        latency = record.get("histograms", {}).get("vllm:inter_token_latency_seconds", {})
        if latency.get("mean"):
            entry["decode"].append(1.0 / latency["mean"])
    return bins


def concurrency_time(records):
    """Seconds spent at each level of concurrency.

    A peak says what the server survived once; this says what it actually sat at, which
    is what headroom is judged from. Idle is kept as its own bin -- time at zero is a
    real and important part of the answer.
    """
    seconds = {}
    for record in records:
        interval = record.get("interval_seconds")
        running = record.get("gauges", {}).get("vllm:num_requests_running")
        if not interval or running is None:
            continue
        edge = 0 if running < 1 else concurrency_bin(running)
        seconds[edge] = seconds.get(edge, 0.0) + interval
    return seconds


def median(values):
    ordered = sorted(values)
    middle = len(ordered) // 2
    if not ordered:
        return None
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2


# --- Run summary --------------------------------------------------------------------


def sum_counter(records, key):
    return sum(record.get("counters", {}).get(key, 0.0) for record in records)


def sum_counter_prefix(records, prefix):
    return sum(
        value
        for record in records
        for key, value in record.get("counters", {}).items()
        if key.startswith(prefix)
    )


def histogram_mean(records, key):
    """Exact: summing the histogram sums and counts recovers the true mean, which
    averaging the per-interval means would not."""
    total = sum(record.get("histograms", {}).get(key, {}).get("sum", 0.0) for record in records)
    count = sum(record.get("histograms", {}).get(key, {}).get("count", 0.0) for record in records)
    return total / count if count else None


def peak_gauge(records, key, factor=1.0):
    values = [
        record["gauges"][key]
        for record in records
        if record.get("gauges", {}).get(key) is not None
    ]
    return max(values) * factor if values else None


def mean_of(records, extract):
    values = [value for value in map(extract, records) if value is not None]
    return sum(values) / len(values) if values else None


def status_totals(records):
    totals = {}
    for record in records:
        for key, value in record.get("counters", {}).items():
            if key.startswith("http_requests_total"):
                found = re.search(r"status=([^,}]+)", key)
                if found:
                    totals[found.group(1)] = totals.get(found.group(1), 0.0) + value
    return totals


def summarise(records, prompts, generations, latency):
    measured = sum(record.get("interval_seconds", 0.0) for record in records)
    generation = sum_counter(records, "vllm:generation_tokens_total")
    prompt = sum_counter(records, "vllm:prompt_tokens_total")
    queries = sum_counter(records, "vllm:prefix_cache_queries_total")
    hits = sum_counter(records, "vllm:prefix_cache_hits_total")
    occupancy = mean_of(records, gpu_stat("utilization_percent"))
    statuses = status_totals(records)
    served = sum(statuses.values())
    failed = sum(count for status, count in statuses.items() if not status.startswith("2"))

    workload = [
        ("Requests finished", format_number(sum_counter_prefix(records, REQUEST_SUCCESS)), ""),
        ("Request error rate", f"{100 * failed / served:.2f}" if served else "--",
         "%" if served else ""),
        ("Peak concurrent requests",
         format_number(peak_gauge(records, "vllm:num_requests_running")), ""),
        ("Median prompt tokens", format_tokens_value(distribution_quantile(*prompts, 0.5)), ""),
        ("p99 prompt tokens", format_tokens_value(distribution_quantile(*prompts, 0.99)), ""),
        ("Longest prompt seen", largest_observed(*prompts), ""),
        ("Median generation tokens",
         format_tokens_value(distribution_quantile(*generations, 0.5)), ""),
        ("Prefix cache hit rate", f"{100 * hits / queries:.1f}" if queries else "--",
         "%" if queries else ""),
    ]
    service = [
        ("Prompt tokens", format_number(prompt), ""),
        ("Generation tokens", format_number(generation), ""),
        ("Mean time to first token", format_duration_value(
            histogram_mean(records, "vllm:time_to_first_token_seconds")), ""),
        ("p99 time to first token",
         format_duration_value(distribution_quantile(*latency, 0.99)), ""),
        ("Mean inter-token latency", format_duration_value(
            histogram_mean(records, "vllm:inter_token_latency_seconds")), ""),
        ("Mean queue time", format_duration_value(
            histogram_mean(records, "vllm:request_queue_time_seconds")), ""),
        ("Peak KV cache usage",
         format_percent(peak_gauge(records, "vllm:kv_cache_usage_perc", 100)), ""),
        ("Peak requests waiting",
         format_number(peak_gauge(records, "vllm:num_requests_waiting")), ""),
        ("Preemptions", format_number(sum_counter(records, "vllm:num_preemptions_total")), ""),
        ("Mean GPU occupancy", f"{occupancy:.0f}" if occupancy is not None else "--",
         "%" if occupancy is not None else ""),
    ]
    return {
        "hero": format_rate(generation, measured),
        "measured": measured,
        "groups": [("Workload", workload), ("Service", service)],
    }


def format_tokens_value(value):
    return "--" if value is None else format_tokens(value)


def format_tokens(value):
    if value is None:
        return "--"
    if math.isinf(value):
        return "∞"
    if value >= 1e6:
        return f"{value / 1e6:,.1f}M".replace(".0M", "M")
    if value >= 1000:
        return f"{value / 1000:,.1f}K".replace(".0K", "K")
    return f"{value:,.0f}"


def format_rate(total, seconds):
    return f"{total / seconds:,.1f}" if seconds > 0 else "--"


def format_duration_value(seconds):
    if seconds is None:
        return "--"
    return f"{seconds * 1000:.0f} ms" if seconds < 1 else f"{seconds:.2f} s"


def format_percent(value):
    return "--" if value is None else f"{value:.1f}%"


def format_number(value):
    if value is None:
        return "--"
    if abs(value) >= 1e9:
        return f"{value / 1e9:,.2f}B"
    if abs(value) >= 1e6:
        return f"{value / 1e6:,.2f}M"
    if abs(value) >= 10_000:
        return f"{value / 1e3:,.1f}K"
    return f"{value:,.0f}" if float(value).is_integer() else f"{value:,.2f}"


def format_clock(seconds):
    seconds = int(round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes}:{seconds:02d}"


def format_tick(value, decimals):
    return f"{value:,.{decimals}f}"


# --- SVG ----------------------------------------------------------------------------


def x_pixel(position, first, last):
    if last <= first:
        return (PLOT_LEFT + PLOT_RIGHT) / 2
    return PLOT_LEFT + (position - first) / (last - first) * (PLOT_RIGHT - PLOT_LEFT)


def y_pixel(value, top):
    return PLOT_BOTTOM - min(value, top) / top * PLOT_HEIGHT if top > 0 else PLOT_BOTTOM


def runs_of(values, breaks):
    """Indices grouped into stretches of consecutive present values, so gaps stay gaps."""
    groups, current = [], []
    for index, value in enumerate(values):
        if value is None or breaks[index]:
            if current:
                groups.append(current)
            current = []
        if value is not None:
            current.append(index)
    if current:
        groups.append(current)
    return groups


def render_chart(chart, xs, breaks, restarts):
    first, last = (xs[0], xs[-1]) if xs else (0.0, 1.0)
    pixels = [x_pixel(position, first, last) for position in xs]
    parts = [
        f'<svg class="plot" viewBox="0 0 {WIDTH} {HEIGHT}" role="img" '
        f'aria-label="{html.escape(chart.title)}">'
    ]

    for value in chart.ticks:
        y = y_pixel(value, chart.top)
        parts.append(
            f'<line class="grid" x1="{PLOT_LEFT}" y1="{y:.1f}" x2="{PLOT_RIGHT}" y2="{y:.1f}"/>'
            f'<text class="tick y" x="{PLOT_LEFT - 8}" y="{y + 4:.1f}">'
            f"{format_tick(value, chart.decimals)}</text>"
        )

    for position in restarts:
        x = x_pixel(position, first, last)
        parts.append(
            f'<line class="restart" x1="{x:.1f}" y1="{MARGIN_TOP}" x2="{x:.1f}" y2="{PLOT_BOTTOM}"/>'
        )

    for index in x_tick_indices(len(xs)):
        parts.append(
            f'<text class="tick x" x="{pixels[index]:.1f}" y="{PLOT_BOTTOM + 16}">'
            f"{format_clock(xs[index])}</text>"
        )

    bucketed = any(
        low != high
        for member in chart.series
        for low, high in zip(member.lows, member.highs)
        if low is not None
    )

    for member in chart.series:
        values = [None if value is None else value * chart.scale for value in member.means]
        if bucketed:
            parts.append(render_band(member, chart, pixels, breaks))
        elif len(chart.series) == 1:
            parts.append(render_area(values, chart, pixels, breaks, member.slot))
        parts.append(render_line(values, chart, pixels, breaks, member.slot))

    parts.append(render_end_labels(chart, pixels))
    parts.append(
        f'<line class="axis" x1="{PLOT_LEFT}" y1="{PLOT_BOTTOM}" '
        f'x2="{PLOT_RIGHT}" y2="{PLOT_BOTTOM}"/>'
    )
    parts.append(f'<g class="focus"><line class="crosshair" y1="{MARGIN_TOP}" y2="{PLOT_BOTTOM}"/>')
    parts.extend(f'<circle class="focus-dot s{member.slot}" r="3.5"/>' for member in chart.series)
    parts.append("</g></svg>")
    return "".join(parts)


def render_line(values, chart, pixels, breaks, slot):
    shapes = []
    for group in runs_of(values, breaks):
        points = " ".join(f"{pixels[index]:.1f},{y_pixel(values[index], chart.top):.1f}" for index in group)
        if len(group) == 1:
            index = group[0]
            shapes.append(
                f'<circle class="dot s{slot}" cx="{pixels[index]:.1f}" '
                f'cy="{y_pixel(values[index], chart.top):.1f}" r="2.5"/>'
            )
        else:
            shapes.append(f'<polyline class="line s{slot}" points="{points}"/>')
    return "".join(shapes)


def render_band(member, chart, pixels, breaks):
    lows = [None if value is None else value * chart.scale for value in member.lows]
    highs = [None if value is None else value * chart.scale for value in member.highs]
    shapes = []
    for group in runs_of(lows, breaks):
        if len(group) < 2:
            continue
        upper = " ".join(f"{pixels[i]:.1f},{y_pixel(highs[i], chart.top):.1f}" for i in group)
        lower = " ".join(f"{pixels[i]:.1f},{y_pixel(lows[i], chart.top):.1f}" for i in reversed(group))
        shapes.append(f'<polygon class="band s{member.slot}" points="{upper} {lower}"/>')
    return "".join(shapes)


def render_area(values, chart, pixels, breaks, slot):
    shapes = []
    for group in runs_of(values, breaks):
        if len(group) < 2:
            continue
        points = " ".join(f"{pixels[i]:.1f},{y_pixel(values[i], chart.top):.1f}" for i in group)
        shapes.append(
            f'<polygon class="area s{slot}" points="{pixels[group[0]]:.1f},{PLOT_BOTTOM} '
            f'{points} {pixels[group[-1]]:.1f},{PLOT_BOTTOM}"/>'
        )
    return "".join(shapes)


def render_end_labels(chart, pixels):
    """Direct labels ride the line ends, but only where they cannot collide: nudging them
    apart detaches a label from its line, and the legend already carries identity."""
    ends = []
    for member in chart.series:
        last = next(
            (index for index in reversed(range(len(member.means))) if member.means[index] is not None),
            None,
        )
        if last is not None:
            ends.append((member.slot, pixels[last], member.means[last] * chart.scale))

    heights = sorted(y_pixel(value, chart.top) for _, _, value in ends)
    if any(second - first < LABEL_CLEARANCE for first, second in zip(heights, heights[1:])):
        return ""

    labels = []
    for slot, x, value in ends:
        y = y_pixel(value, chart.top)
        labels.append(
            f'<line class="key s{slot}" x1="{x + 6:.1f}" y1="{y:.1f}" x2="{x + 16:.1f}" y2="{y:.1f}"/>'
            f'<text class="end-label" x="{x + 21:.1f}" y="{y + 4:.1f}">'
            f"{format_tick(value, chart.decimals)}</text>"
        )
    return "".join(labels)


@dataclass
class BarChart:
    key: str
    title: str
    note: str
    unit: str
    axis: str
    categories: list
    values: list
    detail: list
    decimals: int


def build_bars(key, title, note, unit, axis, categories, values, detail, decimals=0):
    if not any(value for value in values):
        return None
    return BarChart(key, title, note, unit, axis, categories, values, detail, decimals)


def render_bars(chart):
    ticks, _ = nice_ticks(max(chart.values))
    top = ticks[-1]
    parts = [
        f'<svg class="plot bars" viewBox="0 0 {WIDTH} {HEIGHT}" role="img" '
        f'aria-label="{html.escape(chart.title)}">'
    ]

    for value in ticks:
        y = y_pixel(value, top)
        parts.append(
            f'<line class="grid" x1="{PLOT_LEFT}" y1="{y:.1f}" x2="{PLOT_RIGHT}" y2="{y:.1f}"/>'
            f'<text class="tick y" x="{PLOT_LEFT - 8}" y="{y + 4:.1f}">'
            f"{format_tick(value, chart.decimals)}</text>"
        )

    band = (PLOT_RIGHT - PLOT_LEFT) / len(chart.values)
    # Cap the bar and let the leftover band be air; the 2px gap is what separates
    # neighbours, rather than a stroke drawn around each one.
    width = min(24.0, band - 2)
    peak = max(range(len(chart.values)), key=lambda index: chart.values[index])

    for index, value in enumerate(chart.values):
        centre = PLOT_LEFT + band * (index + 0.5)
        # A bucket holding three requests out of a thousand rounds to nothing at this
        # scale; give any non-zero count a sliver, since "rare" and "never" are the
        # distinction the reader is here for. Exact counts stay in the tooltip and table.
        height = max(2.0, PLOT_BOTTOM - y_pixel(value, top)) if value > 0 else 0.0
        y = PLOT_BOTTOM - height
        if height > 0:
            radius = min(4.0, height)
            parts.append(
                f'<path class="bar s1" d="M{centre - width / 2:.1f},{PLOT_BOTTOM} '
                f"V{y + radius:.1f} q0,-{radius:.1f} {radius:.1f},-{radius:.1f} "
                f"h{width - 2 * radius:.1f} q{radius:.1f},0 {radius:.1f},{radius:.1f} "
                f'V{PLOT_BOTTOM} Z"/>'
            )
        if index == peak and value > 0:
            parts.append(
                f'<text class="bar-label" x="{centre:.1f}" y="{y - 6:.1f}">'
                f"{format_tick(value, chart.decimals)}</text>"
            )
        parts.append(
            f'<text class="tick x" x="{centre:.1f}" y="{PLOT_BOTTOM + 16}">'
            f"{html.escape(chart.categories[index])}</text>"
        )
        parts.append(
            f'<rect class="hit" data-index="{index}" x="{PLOT_LEFT + band * index:.1f}" '
            f'y="{MARGIN_TOP}" width="{band:.1f}" height="{PLOT_HEIGHT}"/>'
        )

    parts.append(
        f'<line class="axis" x1="{PLOT_LEFT}" y1="{PLOT_BOTTOM}" '
        f'x2="{PLOT_RIGHT}" y2="{PLOT_BOTTOM}"/></svg>'
    )
    return "".join(parts)


def x_tick_indices(count):
    if count == 0:
        return []
    wanted = min(7, count)
    if wanted == 1:
        return [0]
    return sorted({round(index * (count - 1) / (wanted - 1)) for index in range(wanted)})


# --- Document -----------------------------------------------------------------------


STYLE = """
*, *::before, *::after { box-sizing: border-box; }
body.viz-root {
  color-scheme: light;
  --plane: #f9f9f7;
  --surface-1: #fcfcfb;
  --text-primary: #0b0b0b;
  --text-secondary: #52514e;
  --text-muted: #898781;
  --grid: #e1e0d9;
  --axis: #c3c2b7;
  --border: rgba(11,11,11,0.10);
  --series-1: #2a78d6;
  --series-2: #eb6834;
  --series-3: #1baf7a;
  --series-4: #eda100;
  --critical: #d03b3b;
  margin: 0;
  padding: 40px 24px 72px;
  background: var(--plane);
  color: var(--text-primary);
  font: 15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif;
}
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) body.viz-root {
    color-scheme: dark;
    --plane: #0d0d0d; --surface-1: #1a1a19;
    --text-primary: #ffffff; --text-secondary: #c3c2b7; --text-muted: #898781;
    --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10);
    --series-1: #3987e5; --series-2: #d95926; --series-3: #199e70; --series-4: #c98500;
  }
}
:root[data-theme="dark"] body.viz-root {
  color-scheme: dark;
  --plane: #0d0d0d; --surface-1: #1a1a19;
  --text-primary: #ffffff; --text-secondary: #c3c2b7; --text-muted: #898781;
  --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10);
  --series-1: #3987e5; --series-2: #d95926; --series-3: #199e70; --series-4: #c98500;
}
.wrap { max-width: 1080px; margin: 0 auto; }
header.run { display: flex; justify-content: space-between; align-items: flex-start; gap: 24px; }
h1 { font-size: 21px; font-weight: 600; margin: 0 0 6px; letter-spacing: -0.01em; }
.subtitle { color: var(--text-secondary); font-size: 13px; margin: 0; }
button {
  font: inherit; font-size: 13px; color: var(--text-secondary); background: var(--surface-1);
  border: 1px solid var(--border); border-radius: 7px; padding: 6px 12px; cursor: pointer;
}
button:hover { color: var(--text-primary); }
button:focus-visible { outline: 2px solid var(--series-1); outline-offset: 2px; }
.hero { margin: 34px 0 0; }
.hero .value { font-size: 52px; font-weight: 600; line-height: 1; letter-spacing: -0.02em; }
.hero .unit { font-size: 20px; font-weight: 500; color: var(--text-secondary); margin-left: 8px; }
.hero .label { color: var(--text-secondary); font-size: 13px; margin-top: 10px; }
.meta { display: flex; flex-wrap: wrap; gap: 6px 26px; margin: 18px 0 0;
  color: var(--text-secondary); font-size: 13px; }
.meta b { font-weight: 600; color: var(--text-primary); font-variant-numeric: tabular-nums; }
.meta .warn b, .meta .warn { color: var(--critical); }
h3.group { font-size: 12px; font-weight: 600; text-transform: uppercase;
  letter-spacing: 0.06em; color: var(--text-muted); margin: 26px 0 8px; }
.tiles { display: grid; grid-template-columns: repeat(auto-fill, minmax(172px, 1fr));
  gap: 8px; margin: 0; }
.capacity-table { margin: 12px 0 4px; }
.capacity-table td:first-child, .capacity-table th:first-child { text-align: left; }
.caveat { color: var(--text-secondary); font-size: 12.5px; margin: 12px 0 4px; max-width: 780px; }
.caveat b { color: var(--text-primary); }
.caveat code { font-size: 12px; background: var(--plane); padding: 1px 4px; border-radius: 4px; }
.bar { stroke: none; }
.bar.on { opacity: 0.75; }
.bar-label { fill: var(--text-secondary); font-size: 11.5px; text-anchor: middle;
  font-variant-numeric: tabular-nums; }
.hit { fill: transparent; }
.tile { background: var(--surface-1); padding: 14px 16px; display: flex;
  flex-direction: column; justify-content: space-between;
  border: 1px solid var(--border); border-radius: 9px; }
.tile .label { color: var(--text-secondary); font-size: 12px; margin-bottom: 6px; }
.tile .value { font-size: 22px; font-weight: 600; letter-spacing: -0.01em; }
.tile .value .unit { font-size: 13px; font-weight: 500; color: var(--text-secondary); margin-left: 3px; }
.card { background: var(--surface-1); border: 1px solid var(--border); border-radius: 10px;
  padding: 16px 18px 12px; margin-top: 16px; }
.card:focus-visible { outline: 2px solid var(--series-1); outline-offset: 2px; }
.card-head { display: flex; justify-content: space-between; align-items: flex-start; gap: 16px; }
.card h2 { font-size: 15px; font-weight: 600; margin: 0; }
.card .note { color: var(--text-secondary); font-size: 12.5px; margin: 3px 0 0; }
.head-right { display: flex; align-items: center; gap: 12px; flex: none; }
.unit-label { color: var(--text-muted); font-size: 12px; }
.legend { display: flex; flex-wrap: wrap; gap: 4px 14px; margin: 10px 0 0;
  color: var(--text-secondary); font-size: 12.5px; }
.legend span { display: inline-flex; align-items: center; gap: 6px; }
.legend i { width: 14px; height: 2px; border-radius: 1px; display: inline-block; }
.plot { display: block; width: 100%; height: auto; margin-top: 4px; touch-action: none; }
.grid { stroke: var(--grid); stroke-width: 1; }
.axis { stroke: var(--axis); stroke-width: 1; }
.restart { stroke: var(--critical); stroke-width: 1; stroke-opacity: 0.5; }
.tick { fill: var(--text-muted); font-size: 11px; font-variant-numeric: tabular-nums; }
.tick.y { text-anchor: end; }
.tick.x { text-anchor: middle; }
.end-label { fill: var(--text-secondary); font-size: 11.5px; font-variant-numeric: tabular-nums; }
.line { fill: none; stroke-width: 2; stroke-linejoin: round; stroke-linecap: round; }
.key { stroke-width: 2; stroke-linecap: round; }
.band { stroke: none; opacity: 0.16; }
.area { stroke: none; opacity: 0.10; }
.crosshair { stroke: var(--text-muted); stroke-width: 1; }
.s1 { stroke: var(--series-1); } polygon.s1, circle.s1, path.s1 { fill: var(--series-1); }
.s2 { stroke: var(--series-2); } polygon.s2, circle.s2, path.s2 { fill: var(--series-2); }
.s3 { stroke: var(--series-3); } polygon.s3, circle.s3, path.s3 { fill: var(--series-3); }
.s4 { stroke: var(--series-4); } polygon.s4, circle.s4, path.s4 { fill: var(--series-4); }
circle.dot, circle.focus-dot { stroke: var(--surface-1); stroke-width: 2; }
.focus { visibility: hidden; }
.focus.on { visibility: visible; }
.tip { position: fixed; z-index: 10; pointer-events: none; min-width: 156px;
  background: var(--surface-1); border: 1px solid var(--border); border-radius: 8px;
  padding: 9px 11px; font-size: 12.5px; box-shadow: 0 6px 22px rgba(0,0,0,0.16); }
.tip .when { color: var(--text-secondary); font-size: 11.5px; margin-bottom: 6px;
  font-variant-numeric: tabular-nums; }
.tip .row { display: flex; align-items: center; gap: 8px; margin-top: 3px; }
.tip .row i { width: 12px; height: 2px; border-radius: 1px; flex: none; }
.tip .row .name { color: var(--text-secondary); }
.tip .row .value { margin-left: auto; font-weight: 600; font-variant-numeric: tabular-nums; }
.table-wrap { max-height: 300px; overflow: auto; margin: 10px 0 4px; }
table { border-collapse: collapse; font-size: 12.5px; font-variant-numeric: tabular-nums; }
th, td { text-align: right; padding: 4px 18px 4px 0; border-bottom: 1px solid var(--border);
  white-space: nowrap; min-width: 92px; }
th:first-child, td:first-child { text-align: left; }
thead th { position: sticky; top: 0; background: var(--surface-1);
  color: var(--text-secondary); font-weight: 600; }
footer { color: var(--text-muted); font-size: 12px; margin-top: 28px; line-height: 1.7;
  max-width: 760px; }
@media print { button { display: none; } .card { break-inside: avoid; } }
"""


SCRIPT = """
const DATA = JSON.parse(document.getElementById("report-data").textContent);
const GEO = DATA.geometry;
const COUNT = DATA.elapsed.length;

const tip = document.createElement("div");
tip.className = "tip";
tip.hidden = true;
document.body.appendChild(tip);

document.getElementById("theme").addEventListener("click", () => {
  const root = document.documentElement;
  const dark = root.getAttribute("data-theme") === "dark"
    || (!root.hasAttribute("data-theme") && matchMedia("(prefers-color-scheme: dark)").matches);
  root.setAttribute("data-theme", dark ? "light" : "dark");
});

const xPixelOf = index =>
  GEO.left + (COUNT > 1 ? index / (COUNT - 1) : 0.5) * (GEO.right - GEO.left);

function nearestIndex(svg, clientX) {
  const box = svg.getBoundingClientRect();
  const local = (clientX - box.left) / box.width * GEO.width;
  const span = GEO.right - GEO.left;
  const fraction = span > 0 ? (local - GEO.left) / span : 0;
  return Math.max(0, Math.min(COUNT - 1, Math.round(fraction * (COUNT - 1))));
}

const format = (value, decimals) => value.toLocaleString(undefined, {
  minimumFractionDigits: decimals, maximumFractionDigits: decimals,
});

function show(chart, card, index, clientX, clientY) {
  const x = xPixelOf(index);

  for (const spec of DATA.charts) {
    const focus = document.querySelector('.card[data-key="' + spec.key + '"] .focus');
    focus.classList.add("on");
    const line = focus.querySelector(".crosshair");
    line.setAttribute("x1", x);
    line.setAttribute("x2", x);
    focus.querySelectorAll(".focus-dot").forEach((dot, slot) => {
      const value = spec.series[slot].values[index];
      dot.setAttribute("visibility", value === null ? "hidden" : "visible");
      if (value !== null) {
        dot.setAttribute("cx", x);
        dot.setAttribute("cy", GEO.bottom - Math.min(value, spec.top) / spec.top * GEO.plotHeight);
      }
    });
  }

  tip.textContent = "";
  const when = document.createElement("div");
  when.className = "when";
  when.textContent = DATA.stamps[index] + "  ·  +" + DATA.elapsed[index];
  tip.appendChild(when);

  for (const series of chart.series) {
    const value = series.values[index];
    const row = document.createElement("div");
    row.className = "row";
    const key = document.createElement("i");
    key.style.background = "var(--series-" + series.slot + ")";
    const name = document.createElement("span");
    name.className = "name";
    name.textContent = series.name;
    const shown = document.createElement("span");
    shown.className = "value";
    shown.textContent = value === null ? "no data" : format(value, chart.decimals) + " " + chart.unit;
    row.append(key, name, shown);
    tip.appendChild(row);
  }

  tip.hidden = false;
  const box = tip.getBoundingClientRect();
  const cardBox = card.getBoundingClientRect();
  const anchorX = clientX === null ? cardBox.left + cardBox.width / 2 : clientX;
  const anchorY = clientY === null ? cardBox.top + 48 : clientY;
  tip.style.left = Math.min(Math.max(8, anchorX + 16), innerWidth - box.width - 8) + "px";
  tip.style.top = Math.min(Math.max(8, anchorY - box.height - 14), innerHeight - box.height - 8) + "px";
}

function hide() {
  tip.hidden = true;
  document.querySelectorAll(".focus").forEach(focus => focus.classList.remove("on"));
}

let focused = 0;

for (const chart of DATA.charts) {
  const card = document.querySelector('.card[data-key="' + chart.key + '"]');
  const svg = card.querySelector("svg.plot");

  const track = event => {
    focused = nearestIndex(svg, event.clientX);
    show(chart, card, focused, event.clientX, event.clientY);
  };
  svg.addEventListener("pointermove", track);
  svg.addEventListener("pointerdown", track);
  svg.addEventListener("pointerleave", hide);
  card.addEventListener("blur", hide);

  card.addEventListener("keydown", event => {
    const step = event.key === "ArrowRight" ? 1 : event.key === "ArrowLeft" ? -1 : 0;
    if (!step && event.key !== "Home" && event.key !== "End") return;
    event.preventDefault();
    if (event.key === "Home") focused = 0;
    else if (event.key === "End") focused = COUNT - 1;
    else focused = Math.max(0, Math.min(COUNT - 1, focused + step));
    show(chart, card, focused, null, null);
  });

  card.querySelector(".table-toggle").addEventListener("click", event => {
    const wrap = card.querySelector(".table-wrap");
    if (!wrap.firstChild) wrap.appendChild(buildTable(chart));
    wrap.hidden = !wrap.hidden;
    event.currentTarget.textContent = wrap.hidden ? "Table" : "Hide table";
  });
}

for (const bars of DATA.bars) {
  const card = document.querySelector('.card[data-key="' + bars.key + '"]');
  const svg = card.querySelector("svg.bars");
  const painted = svg.querySelectorAll(".bar");

  svg.addEventListener("pointermove", event => {
    const hit = event.target.closest(".hit");
    if (!hit) return;
    const index = Number(hit.dataset.index);
    painted.forEach(bar => bar.classList.remove("on"));
    if (painted[index]) painted[index].classList.add("on");

    tip.textContent = "";
    const when = document.createElement("div");
    when.className = "when";
    when.textContent = bars.categories[index];
    tip.appendChild(when);

    const row = document.createElement("div");
    row.className = "row";
    const key = document.createElement("i");
    key.style.background = "var(--series-1)";
    const name = document.createElement("span");
    name.className = "name";
    name.textContent = bars.detail[index] || "";
    const shown = document.createElement("span");
    shown.className = "value";
    shown.textContent = format(bars.values[index], bars.decimals) + " " + bars.unit;
    row.append(key, name, shown);
    tip.appendChild(row);

    tip.hidden = false;
    const box = tip.getBoundingClientRect();
    tip.style.left = Math.min(Math.max(8, event.clientX + 16), innerWidth - box.width - 8) + "px";
    tip.style.top = Math.min(Math.max(8, event.clientY - box.height - 14),
      innerHeight - box.height - 8) + "px";
  });

  svg.addEventListener("pointerleave", () => {
    painted.forEach(bar => bar.classList.remove("on"));
    tip.hidden = true;
  });

  card.querySelector(".table-toggle").addEventListener("click", event => {
    const wrap = card.querySelector(".table-wrap");
    if (!wrap.firstChild) wrap.appendChild(buildBarTable(bars));
    wrap.hidden = !wrap.hidden;
    event.currentTarget.textContent = wrap.hidden ? "Table" : "Hide table";
  });
}

function buildBarTable(bars) {
  const table = document.createElement("table");
  const head = table.createTHead().insertRow();
  for (const title of ["Bucket", bars.unit, "Share"]) {
    const cell = document.createElement("th");
    cell.textContent = title;
    head.appendChild(cell);
  }
  const body = table.createTBody();
  bars.categories.forEach((category, index) => {
    const row = body.insertRow();
    row.insertCell().textContent = category;
    row.insertCell().textContent = format(bars.values[index], bars.decimals);
    row.insertCell().textContent = bars.detail[index] || "";
  });
  return table;
}

function buildTable(chart) {
  const table = document.createElement("table");
  const head = table.createTHead().insertRow();
  const titles = ["Time"].concat(chart.series.map(s => s.name + " (" + chart.unit + ")"));
  for (const title of titles) {
    const cell = document.createElement("th");
    cell.textContent = title;
    head.appendChild(cell);
  }
  const body = table.createTBody();
  for (let index = 0; index < COUNT; index++) {
    if (chart.series.every(series => series.values[index] === null)) continue;
    const row = body.insertRow();
    row.insertCell().textContent = DATA.elapsed[index];
    for (const series of chart.series) {
      const value = series.values[index];
      row.insertCell().textContent = value === null ? "--" : format(value, chart.decimals);
    }
  }
  return table;
}
"""


def distribution_bars(key, title, note, records, chart_key):
    bounds, counts = distribution(records, key)
    total = sum(counts)
    if not bounds or total <= 0:
        return None

    categories, detail, lower = [], [], 0.0
    for bound, count in zip(bounds, counts):
        categories.append(f"{format_tokens(lower)}+" if math.isinf(bound) else format_tokens(bound))
        detail.append(f"{100 * count / total:.1f}% of requests")
        lower = bound
    return build_bars(chart_key, title, note, "requests", "", categories, counts, detail)


def occupancy_bars(seconds):
    if not seconds:
        return None
    measured = sum(seconds.values())
    edges = sorted(seconds)
    return build_bars(
        "occupancy",
        "Time spent at each concurrency level",
        "How loaded the server actually was, rather than how loaded it ever got. Weight to "
        "the right is where headroom is running out; a tall idle bar is spare capacity.",
        "seconds",
        "",
        ["idle" if edge == 0 else str(edge) for edge in edges],
        [seconds[edge] for edge in edges],
        [f"{100 * seconds[edge] / measured:.1f}% of measured time" for edge in edges],
    )


def profile_bars(profile, field, chart_key, title, note, unit):
    bins = sorted(profile)
    if not bins:
        return None
    return build_bars(
        chart_key,
        title,
        note,
        unit,
        "",
        [str(edge) for edge in bins],
        [median(profile[edge][field]) or 0.0 for edge in bins],
        [f"{len(profile[edge][field])} samples" for edge in bins],
    )


def capacity_section(records, prompts):
    """The question this answers -- how many requests fit at a given context length --
    is arithmetic on the KV cache size, not something vLLM measures."""
    info = run_info(records)
    capacity = kv_cache_tokens(info)
    limit = context_limit(info)
    if not capacity or not limit:
        return ""

    rows = "".join(
        f"<tr><td>{format_tokens(length)}</td>"
        f"<td>{fits:,}</td>"
        f"<td>{100 * length / capacity:.2f}%</td></tr>"
        for length, fits, _ in capacity_rows(capacity, limit)
    )

    observed = distribution_quantile(*prompts, 0.99)
    footnote = ""
    if observed and observed > 0:
        footnote = (
            f" At the p99 prompt length actually seen — {format_tokens(observed)} tokens — "
            f"<b>{int(capacity // observed):,}</b> requests fit at once."
        )

    models = ", ".join(html.escape(name) for name in info.get("models", {}))
    return (
        '<section class="card capacity"><div class="card-head"><div>'
        "<h2>Capacity by context length</h2>"
        f'<p class="note">{html.escape(models)} · KV cache holds '
        f"<b>{format_tokens(capacity)}</b> tokens · context limit "
        f"<b>{format_tokens(limit)}</b> tokens.</p></div></div>"
        '<table class="capacity-table"><thead><tr><th>Context length</th>'
        "<th>Requests that fit at once</th><th>KV cache each</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
        f'<p class="caveat">A floor, not a ceiling: the figure assumes every request holds its '
        f"own blocks, and prefix caching lets requests sharing a prompt prefix share blocks. "
        f"Context length counts prompt <em>plus</em> generated tokens.{footnote}</p>"
        '<p class="caveat">vLLM counts requests, not people. It has no notion of a user or a '
        "session — no per-caller labels exist anywhere in <code>/metrics</code>, and the OpenAI "
        "<code>user</code> field is accepted and discarded. Concurrent <em>requests</em> here is "
        "what sizes the GPU; turning that into concurrent <em>users</em> needs LiteLLM's "
        "per-key request rates alongside it.</p>"
        "</section>"
    )


def config_notes(records):
    """A few facts about the server the run happened on, taken from the info block."""
    info = next((record["info"] for record in records if record.get("info")), {})
    cache = info.get("vllm:cache_config_info", {})
    notes = []
    if cache.get("kv_cache_size_tokens"):
        notes.append(f'<span>KV cache <b>{format_number(float(cache["kv_cache_size_tokens"]))} tokens</b></span>')
    if cache.get("block_size"):
        notes.append(f'<span>Block size <b>{html.escape(cache["block_size"])}</b></span>')
    if cache.get("enable_prefix_caching"):
        enabled = cache["enable_prefix_caching"] == "True"
        notes.append(f'<span>Prefix caching <b>{"on" if enabled else "off"}</b></span>')
    return notes


def build_payload(charts, bars, xs, stamps):
    return {
        "stamps": stamps,
        "bars": [
            {
                "key": chart.key,
                "unit": chart.unit,
                "decimals": chart.decimals,
                "categories": chart.categories,
                "values": [round(value, 4) for value in chart.values],
                "detail": chart.detail,
            }
            for chart in bars
        ],
        "elapsed": [format_clock(position) for position in xs],
        "geometry": {
            "width": WIDTH,
            "left": PLOT_LEFT,
            "right": PLOT_RIGHT,
            "bottom": PLOT_BOTTOM,
            "plotHeight": PLOT_HEIGHT,
        },
        "charts": [
            {
                "key": chart.key,
                "unit": chart.unit,
                "decimals": chart.decimals,
                "top": chart.top,
                "series": [
                    {
                        "name": member.name,
                        "slot": member.slot,
                        "values": [
                            None if value is None else round(value * chart.scale, 4)
                            for value in member.means
                        ],
                    }
                    for member in chart.series
                ],
            }
            for chart in charts
        ],
    }


def build_card(chart, xs, breaks, restarts):
    legend = ""
    if len(chart.series) > 1:
        keys = "".join(
            f'<span><i style="background:var(--series-{member.slot})"></i>'
            f"{html.escape(member.name)}</span>"
            for member in chart.series
        )
        legend = f'<div class="legend">{keys}</div>'

    return (
        f'<section class="card" data-key="{chart.key}" tabindex="0">'
        f'<div class="card-head"><div><h2>{html.escape(chart.title)}</h2>'
        f'<p class="note">{html.escape(chart.note)}</p></div>'
        f'<div class="head-right"><span class="unit-label">{html.escape(chart.unit)}</span>'
        f'<button class="table-toggle" type="button">Table</button></div></div>'
        f"{legend}"
        f"{render_chart(chart, xs, breaks, restarts)}"
        f'<div class="table-wrap" hidden></div></section>'
    )


def build_tile(label, value, unit):
    suffix = f'<span class="unit">{html.escape(unit)}</span>' if unit else ""
    return (
        f'<div class="tile"><div class="label">{html.escape(label)}</div>'
        f'<div class="value">{html.escape(value)}{suffix}</div></div>'
    )


def build_tile_group(title, tiles):
    return (
        f'<h3 class="group">{html.escape(title)}</h3>'
        f'<div class="tiles">{"".join(build_tile(*tile) for tile in tiles)}</div>'
    )


def build_bar_card(chart):
    return (
        f'<section class="card" data-key="{chart.key}">'
        f'<div class="card-head"><div><h2>{html.escape(chart.title)}</h2>'
        f'<p class="note">{html.escape(chart.note)}</p></div>'
        f'<div class="head-right"><span class="unit-label">{html.escape(chart.unit)}</span>'
        f'<button class="table-toggle" type="button">Table</button></div></div>'
        f"{render_bars(chart)}"
        f'<div class="table-wrap" hidden></div></section>'
    )


def render_document(source, records, charts, bars, xs, stamps, breaks, restarts,
                    summary, notes, capacity):
    groups = "".join(build_tile_group(title, tiles) for title, tiles in summary["groups"])
    cards = "".join(build_card(chart, xs, breaks, restarts) for chart in charts)
    bar_cards = "".join(build_bar_card(chart) for chart in bars)
    payload = json.dumps(build_payload(charts, bars, xs, stamps)).replace("<", "\\u003c")

    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        "<title>vLLM run report</title>"
        f'<style>{STYLE}</style></head><body class="viz-root"><div class="wrap">'
        '<header class="run"><div><h1>vLLM run report</h1>'
        f'<p class="subtitle">{html.escape(os.path.basename(source))} · '
        f'{html.escape(records[0]["timestamp"])} → {html.escape(records[-1]["timestamp"])}</p>'
        '</div><button id="theme" type="button">Theme</button></header>'
        f'<div class="hero"><div class="value">{summary["hero"]}'
        '<span class="unit">tokens/s</span></div>'
        '<div class="label">Mean generation throughput over the measured interval</div></div>'
        f'<div class="meta">{"".join(notes)}</div>'
        f"{groups}"
        f"{capacity}"
        f"{bar_cards}"
        f"{cards}"
        "<footer>Counters are shown as rates over the interval each sample actually "
        "covered, and gaps in a line are missing data rather than interpolation. "
        "Histogram means are exact; p90 is estimated from vLLM's bucket boundaries and is "
        "coarse when few requests finished within an interval. Red rules mark vLLM restarts."
        "</footer></div>"
        f'<script type="application/json" id="report-data">{payload}</script>'
        f"<script>{SCRIPT}</script></body></html>"
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("input", help="JSONL file written by collect-metrics.py")
    parser.add_argument("--output", default="report.html")
    parser.add_argument(
        "--buckets",
        type=int,
        default=MAX_BUCKETS,
        help=f"maximum plotted points per series (default {MAX_BUCKETS})",
    )
    args = parser.parse_args()

    records = load_records(args.input)
    bounds = bucket_bounds(len(records), max(1, args.buckets))
    xs, stamps, breaks = bucket_axis(records, bounds)

    charts = [
        chart
        for chart in (build_chart(spec, records, bounds) for spec in chart_specs(records))
        if chart
    ]
    if not charts:
        raise SystemExit(f"{args.input}: no plottable series")

    prompts = distribution(records, "vllm:request_prompt_tokens")
    generations = distribution(records, "vllm:request_generation_tokens")
    latency = distribution(records, "vllm:time_to_first_token_seconds")
    profile = load_profile(records)

    bars = [
        chart
        for chart in (
            distribution_bars(
                "vllm:request_prompt_tokens",
                "Prompt length distribution",
                "Context actually used, by request. Bars are vLLM's own bucket upper bounds, "
                "which stop at 200K -- anything longer only shows as the final bar.",
                records,
                "dist-prompt",
            ),
            distribution_bars(
                "vllm:request_generation_tokens",
                "Generation length distribution",
                "How much each request actually produced.",
                records,
                "dist-generation",
            ),
            occupancy_bars(concurrency_time(records)),
            profile_bars(
                profile, "rates", "load-throughput",
                "Throughput against load",
                "Median generation rate at each level of concurrency; where the bars stop "
                "rising is where the server saturated. Read dips carefully: a batch busy "
                "prefilling long prompts emits no output tokens, so mixed workloads dip "
                "here for reasons that are not saturation.",
                "tokens/s",
            ),
            profile_bars(
                profile, "decode", "load-decode",
                "Decode speed per request against load",
                "Output tokens per second that one request sees, derived from inter-token "
                "latency, so unlike the chart above it is not depressed by prefill. This is "
                "the speed a caller feels. Concurrency is not the only thing driving it — "
                "attention cost grows with context length — so bins mixing short and long "
                "prompts are not comparable. To read a clean curve, load one context size.",
                "tokens/s each",
            ),
        )
        if chart
    ]

    restarts = [record["_elapsed"] for record in records if record.get("baseline") == "restart"]
    summary = summarise(records, prompts, generations, latency)
    failures = sum(1 for record in records if "error" in record)

    notes = [
        f'<span>Samples <b>{len(records):,}</b></span>',
        f'<span>Measured <b>{format_clock(summary["measured"])}</b></span>',
        *config_notes(records),
    ]
    if failures:
        notes.append(f'<span class="warn">Failed scrapes <b>{failures:,}</b></span>')
    if restarts:
        notes.append(f'<span class="warn">vLLM restarts <b>{len(restarts):,}</b></span>')
    if len(bounds) < len(records):
        notes.append(f'<span>Bucketed to <b>{len(bounds):,}</b> points</span>')

    document = render_document(
        args.input, records, charts, bars, xs, stamps, breaks, restarts, summary, notes,
        capacity_section(records, prompts),
    )
    with open(args.output, "w") as stream:
        stream.write(document)

    print(
        f"Wrote {args.output} ({len(document) / 1024:.0f} KB, {len(charts)} charts, "
        f"{len(bounds):,} points)",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
