#!/usr/bin/env python3

import argparse
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
WORKSPACE = Path("/workspace")
KV_CACHE_DIR = WORKSPACE / "kv_cache"
MOUNTS_PATH = Path("/proc/mounts")
NETWORK_FILESYSTEMS = frozenset({"nfs", "nfs4", "lustre", "cifs", "smb3", "ceph", "glusterfs", "beegfs", "9p"})
FAMILIES = ("vllm", "sglang")
COMMANDS = ("launch", "get-environment", "check")
PORT = 9001
SEED = 234785107
PYTHON_HASH_SEED = "0"
BLACKWELL_MAJORS = (10, 12)
GIB = 1024**3
GB = 1000**3
KV_CACHE_MIN_FREE_FLOOR_GIB = 32
KV_CACHE_MIN_FREE_PERCENT = 2
MODEL_KEYS = frozenset({
    "extends", "hf_repo", "served_model_name", "trust_remote_code", "chat_template_kwargs", "generation_config", "kv_cache_offload",
    "gpu_memory", "parallelism", "default_engine", "validated_engines", "engine_args"
})

ENGINE_ARGS_KEYS = frozenset({"args", "blackwell_args", "env", "parallelism"})

class ServeError(Exception):
    pass

@dataclass
class Gpu:
    memory_mib: int
    compute_capability: str

    @property
    def is_blackwell(self):
        return int(self.compute_capability.split(".")[0]) in BLACKWELL_MAJORS

@dataclass
class Machine:
    gpus: list
    total_ram_bytes: int
    shm_free_bytes: int

@dataclass
class Resolution:
    model: str
    config: dict | None
    engine: str
    family: str
    engine_dir: Path

    @property
    def is_validated(self):
        return self.config is None or self.engine in self.config["validated_engines"]

@dataclass
class Launch:
    env: dict
    argv: list

def engine_family(engine):
    family = engine.split("-", 1)[0]
    if family not in FAMILIES:
        raise ServeError(f"unknown engine family '{family}' in '{engine}'; families: {', '.join(FAMILIES)}")

    return family

def available_engines(app_dir):
    return sorted(path.parent.name for path in (app_dir / "engines").glob("*/pyproject.toml"))

def installed_engines(app_dir):
    return sorted(path.parent.name for path in (app_dir / "engines").glob("*/.venv"))

def read_image_json(app_dir):
    path = app_dir / "image.json"
    if not path.is_file():
        return {}
    with open(path) as file:
        return json.load(file)

def model_path(model, app_dir):
    return app_dir / "models" / f"{model}.json"

def read_model_file(model, app_dir):
    path = model_path(model, app_dir)

    if not path.is_file():
        known = ", ".join(sorted(candidate.stem for candidate in path.parent.glob("*.json")))
        raise ServeError(f"unknown model '{model}'; models: {known}, none")
    try:
        with open(path) as file:
            config = json.load(file)
    except json.JSONDecodeError as error:
        raise ServeError(f"{path}: {error}") from error

    unknown = sorted(set(config) - MODEL_KEYS)
    if unknown:
        raise ServeError(f"{path}: unknown keys: {', '.join(unknown)}")

    for engine, section in config.get("engine_args", {}).items():
        try:
            engine_family(engine)
        except ServeError as error:
            raise ServeError(f"{path}: engine_args: {error}") from error
        unknown = sorted(set(section) - ENGINE_ARGS_KEYS)

        if unknown:
            raise ServeError(f"{path}: engine_args.{engine}: unknown keys: {', '.join(unknown)}")

    return config

def merge_config(base, overlay):
    merged = dict(base)
    for key, value in overlay.items():
        previous = merged.get(key)
        merged[key] = merge_config(previous, value) if isinstance(value, dict) and isinstance(previous, dict) else value

    return merged

def load_model_config(model, app_dir=APP_DIR):
    config = read_model_file(model, app_dir)
    parent = config.pop("extends", None)

    if parent is not None:
        base = read_model_file(parent, app_dir)
        if "extends" in base:
            raise ServeError(f"{model_path(model, app_dir)}: cannot extend '{parent}', which extends '{base['extends']}'")
        config = merge_config(base, config)

    for key in ("hf_repo", "validated_engines"):
        if key not in config:
            raise ServeError(f"{model_path(model, app_dir)}: missing '{key}'")

    config.setdefault("served_model_name", model)
    config.setdefault("kv_cache_offload", True)

    return config

def engine_version_key(engine):
    parts = engine.partition("-")[2].split(".")
    return [(0, int(part)) if part.isdigit() else (1, part) for part in parts]

def get_engine_name(engine, config, app_dir):
    if engine not in FAMILIES:
        return engine

    candidates = [name for name in available_engines(app_dir) if name.partition("-")[0] == engine]

    if not candidates:
        raise ServeError(f"no '{engine}' engine exists; engines: {', '.join(available_engines(app_dir)) or 'none'}")

    config = config or {}
    if config.get("default_engine") in candidates:
        return config["default_engine"]

    validated = [name for name in config.get("validated_engines", []) if name in candidates]

    return max(validated or candidates, key=engine_version_key)

def resolve(model=None, engine=None, app_dir=APP_DIR, environ=os.environ):
    image = read_image_json(app_dir)
    model = model or environ.get("MODEL") or image.get("default_model")

    if not model:
        raise ServeError("no model specified: pass --model, set MODEL, or build the image with a default model")

    config = None if model == "none" else load_model_config(model, app_dir)
    engine = engine or environ.get("ENGINE") or image.get("engine") or (config or {}).get("default_engine")

    if not engine:
        installed = installed_engines(app_dir)
        if len(installed) == 1:
            engine = installed[0]

    if not engine:
        raise ServeError("no engine specified: pass --engine, set ENGINE, or set default_engine in the model's JSON")

    engine = get_engine_name(engine, config, app_dir)
    family = engine_family(engine)
    engine_dir = app_dir / "engines" / engine

    if not (engine_dir / "pyproject.toml").is_file():
        raise ServeError(f"engine '{engine}' does not exist; engines: {', '.join(available_engines(app_dir))}")

    return Resolution(model, config, engine, family, engine_dir)

def warn_if_unvalidated(resolution):
    if resolution.is_validated:
        return
    validated = ", ".join(resolution.config["validated_engines"])
    print(f"WARNING: model '{resolution.model}' is not validated on engine '{resolution.engine}' (validated on: {validated})", file=sys.stderr)

def require_installed(resolution):
    if not (resolution.engine_dir / ".venv").is_dir():
        installed = ", ".join(installed_engines(resolution.engine_dir.parent.parent)) or "none"
        raise ServeError(f"engine '{resolution.engine}' is not installed (no {resolution.engine_dir}/.venv); installed: {installed}")

def shell_assignments(resolution):
    return f"MODEL={shlex.quote(resolution.model)}\nENGINE={shlex.quote(resolution.engine)}\nENGINE_FAMILY={resolution.family}\n"

def engine_args(config, engine, family):
    sections = config.get("engine_args", {})
    merged = {key: {} for key in ENGINE_ARGS_KEYS}
    for name in (family, engine):
        for key in merged:
            merged[key].update(sections.get(name, {}).get(key, {}))

    return merged

def compact_json(value):
    return json.dumps(value, separators=(",", ":"))

def render_env(env):
    return {key: str(value) for key, value in env.items() if value is not None}

def render_args(args):
    rendered = []
    for key, value in args.items():
        if value is None or value is False:
            continue
        if value is True:
            rendered.append(key)
        elif isinstance(value, (dict, list)):
            rendered.append(f"{key}={compact_json(value)}")
        else:
            rendered.append(f"{key}={value}")

    return rendered

def memory_utilization_flag(family):
    return "--gpu-memory-utilization" if family == "vllm" else "--mem-fraction-static"

def gpu_memory_percent(gpu_memory, gpu_memory_mib):
    headroom_mib = gpu_memory["startup_headroom_gib"] * 1024
    ceiling = round(gpu_memory["max_utilization"] * 100)
    return min(ceiling, (gpu_memory_mib - headroom_mib) * 100 // gpu_memory_mib)

def plan_parallelism(parallelism, gpu_count, usable_mib):
    weights_mib = (parallelism["weights_gib"] - parallelism.get("host_weights_gib", 0)) * 1024
    fixed_mib = (parallelism["runtime_overhead_gib"] + parallelism["min_kv_cache_gib"]) * 1024
    largest_tensor_parallel = 1
    tensor_parallel = 1

    while tensor_parallel <= gpu_count:
        if gpu_count % tensor_parallel == 0:
            largest_tensor_parallel = tensor_parallel
            if weights_mib // tensor_parallel + fixed_mib <= usable_mib:
                return tensor_parallel, gpu_count // tensor_parallel, True
        tensor_parallel *= 2

    return largest_tensor_parallel, gpu_count // largest_tensor_parallel, False

def launch_env(engine, compile_cache_root, kv_cache_min_free_gib):
    cache_dir = compile_cache_root / engine
    return {
        "PYTHONUNBUFFERED": "1",
        "PYTHONHASHSEED": PYTHON_HASH_SEED,
        "CUDA_HOME": "/usr/local/cuda",
        "HF_HUB_CACHE": str(WORKSPACE / "huggingface_cache"),
        "TORCHINDUCTOR_FX_GRAPH_CACHE": "1",
        "TORCHINDUCTOR_AUTOGRAD_CACHE": "1",
        "TORCHINDUCTOR_CACHE_DIR": str(cache_dir / "torch"),
        "TORCH_EXTENSIONS_DIR": str(cache_dir / "torch_extensions"),
        "TRITON_CACHE_DIR": str(cache_dir / "triton"),
        "FLASHINFER_WORKSPACE_BASE": str(cache_dir / "flashinfer"),
        "CUDA_CACHE_PATH": str(cache_dir / "cuda"),
        "TILELANG_CACHE_DIR": str(cache_dir / "tilelang"),
        "CUTE_DSL_CACHE_DIR": str(cache_dir / "cutedsl"),
        "VLLM_DO_NOT_TRACK": "1",
        "VLLM_CACHE_ROOT": str(cache_dir / "vllm"),
        "VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR": str(cache_dir / "flashinfer_autotune"),
        "SGLANG_CACHE_DIR": str(cache_dir / "sglang"),
        "SGLANG_DG_CACHE_DIR": str(cache_dir / "deep_gemm"),
        "SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR": str(KV_CACHE_DIR),
        "SGLANG_HICACHE_FILE_BACKEND_MIN_FREE_SPACE": f"{kv_cache_min_free_gib}Gi",
    }

def engine_command(resolution):
    return ["uv", "run", "--frozen", "--no-sync", "--project", str(resolution.engine_dir), resolution.family, "serve", resolution.config["hf_repo"]]

def filesystem_type(path, mounts_path):
    """The filesystem serving `path`, by the longest matching mount point, as the kernel picks."""
    try:
        mounts = mounts_path.read_text()
    except OSError:
        return ""

    best_mount = ""
    best_type = ""

    for line in mounts.splitlines():
        fields = line.split()
        if len(fields) < 3:
            continue
        mount_point, name = fields[1], fields[2]
        under_mount = path == mount_point or path.startswith(mount_point.rstrip("/") + "/")
        if under_mount and len(mount_point) >= len(best_mount):
            best_mount, best_type = mount_point, name

    return best_type

def filesystem_capacity_bytes(path, statvfs=os.statvfs):
    """The size of the filesystem `path` is on, or would be created on."""
    for candidate in (path, *path.parents):
        try:
            status = statvfs(candidate)
        except OSError:
            continue
        return status.f_blocks * status.f_frsize

    return 0

def kv_cache_min_free_gib(cache_dir, environ=os.environ, statvfs=os.statvfs):
    """How much of the volume the disk KV tier must leave alone.

    The engine evicts its own pages to hold this line, so it is the whole size of the tier:
    everything the checkpoint and the caches do not take. A fixed number is wrong at both ends
    of the range of volumes this runs on -- 32 GiB is a rounding error on 4 TB and most of a
    100 GiB disk -- so it scales with the volume, with the old default as the floor.
    """
    override = environ.get("KV_CACHE_MIN_FREE_GIB")

    if override:
        return int(override)

    capacity = filesystem_capacity_bytes(cache_dir, statvfs)

    return max(KV_CACHE_MIN_FREE_FLOOR_GIB, capacity * KV_CACHE_MIN_FREE_PERCENT // 100 // GIB)

def report_kv_cache_directory(cache_dir, min_free_gib):
    capacity = filesystem_capacity_bytes(cache_dir, os.statvfs)
    name = filesystem_type(str(cache_dir), MOUNTS_PATH)
    print(f"KV cache disk tier: {cache_dir} ({name or 'unknown'}, {capacity // GIB} GiB), keeping {min_free_gib} GiB of it free")

def report_checkpoint_directory(models_dir):
    name = filesystem_type(str(models_dir), MOUNTS_PATH)
    print(f"Checkpoint directory: {models_dir} ({name or 'unknown'})")

    if name.startswith("fuse") or name in NETWORK_FILESYSTEMS:
        print(f"WARNING: {models_dir} is on {name}; the whole checkpoint is read through it on every start. "
              f"Point MODELS_DIR at a local disk to load from one.", file=sys.stderr)

def base_args(family, host, models_dir):
    args = {
        "--host": host,
        "--port": PORT,
        "--download-dir": str(models_dir),
        "--load-format": "auto",
        "--enable-mfu-metrics": True,
    }

    if family == "vllm":
        return args | {
            "--seed": SEED,
            "--enable-prefix-caching": True,
            "--scheduling-policy": "priority",
            "--async-scheduling": True,
            "--kv-cache-metrics": True,
        }

    return args | {"--random-seed": SEED, "--enable-priority-scheduling": True, "--enable-metrics": True}

def model_args(config, family):
    args = {
        "--served-model-name": config["served_model_name"],
        "--trust-remote-code": config.get("trust_remote_code", False),
    }

    if "chat_template_kwargs" in config:
        args["--default-chat-template-kwargs"] = config["chat_template_kwargs"]

    if "generation_config" in config:
        flag = "--override-generation-config" if family == "vllm" else "--preferred-sampling-params"
        args[flag] = config["generation_config"]

    return args

def sizing_args(config, family, machine):
    if "gpu_memory" not in config and "parallelism" not in config:
        return {}, 1

    if not machine.gpus:
        raise ServeError("no GPUs detected, but the model's gpu_memory/parallelism settings require them")

    args = {}
    gpu_count = len(machine.gpus)
    smallest_mib = min(gpu.memory_mib for gpu in machine.gpus)
    smallest_gib = smallest_mib // 1024
    usable_mib = smallest_mib

    if "gpu_memory" in config:
        percent = gpu_memory_percent(config["gpu_memory"], smallest_mib)
        usable_mib = smallest_mib * percent // 100
        args[memory_utilization_flag(family)] = f"{percent / 100:.2f}"

    tensor_parallel_label = "tensor parallel" if family == "vllm" else "attention tensor parallel"
    tensor_parallel = 1
    data_parallel = 1

    if "parallelism" in config:
        parallelism = config["parallelism"]
        expert_parallel = bool(parallelism.get("expert_parallel"))
        tensor_parallel, data_parallel, fits = plan_parallelism(parallelism, gpu_count, usable_mib)

        if not fits:
            print(f"WARNING: the model may not fit in {gpu_count} x {smallest_gib} GiB; continuing with {tensor_parallel_label}: {tensor_parallel}")

        if family == "vllm":
            args["--enable-expert-parallel"] = expert_parallel
            args |= {"--tensor-parallel-size": tensor_parallel, "--data-parallel-size": data_parallel}
        else:
            args["--tp-size"] = gpu_count
            args["--ep-size"] = gpu_count if expert_parallel else None
            if data_parallel > 1:
                args |= {"--dp-size": data_parallel, "--enable-dp-attention": True, "--enable-dp-lm-head": True}

    print(f"GPUs: {gpu_count} x {smallest_gib} GiB; {tensor_parallel_label}: {tensor_parallel}, data parallel: {data_parallel}")
    return args, data_parallel

def reserve_host_weights(config, machine, data_parallel):
    """Weights the engine pins in host memory, before the KV tier is offered what is left."""
    pinned_gib = config.get("parallelism", {}).get("host_weights_gib", 0) * data_parallel
    total_gib = machine.total_ram_bytes // GIB

    if pinned_gib >= total_gib:
        raise ServeError(f"{data_parallel} replicas pin {pinned_gib} GiB of weights in host memory, more than the {total_gib} GiB this machine has")

    if pinned_gib:
        print(f"Weights pinned in host memory: {pinned_gib} GiB across {data_parallel} replica(s), out of {total_gib} GiB of RAM")

    return pinned_gib

def offload_args(family, machine, data_parallel, pinned_gib):
    ram_budget = (machine.total_ram_bytes - pinned_gib * GIB) // 2

    if family == "vllm":
        shm_cap = machine.shm_free_bytes * 99 // 100
        per_replica = min(ram_budget, shm_cap) // data_parallel
        return {"--kv-transfer-config": {
            "kv_connector": "OffloadingConnector",
            "kv_role": "kv_both",
            "kv_connector_extra_config": {
                "spec_name": "TieringOffloadingSpec",
                "cpu_bytes_to_use": per_replica,
                "blocks_per_chunk": 4,
                "eviction_policy": "lru",
                "secondary_tiers": [{
                    "type": "fs",
                    "root_dir": str(KV_CACHE_DIR),
                    "n_read_threads": 32,
                    "n_write_threads": 16,
                }],
            },
        }}

    per_group_gb = ram_budget // GB // data_parallel

    return {"--enable-hierarchical-cache": True, "--hicache-size": per_group_gb, "--hicache-storage-backend": "file"}

def report_kv_offload(family, args, machine):
    """The tier the engine is told to build, after the model's own arguments have replaced any of it."""
    if family == "vllm":
        transfer = args.get("--kv-transfer-config")
        if transfer is not None:
            cpu_bytes = transfer["kv_connector_extra_config"]["cpu_bytes_to_use"]
            print(f"KV cache CPU offload: {cpu_bytes // GIB} GiB per replica, out of {machine.shm_free_bytes // GIB} GiB free in /dev/shm")
        return

    if not args.get("--enable-hierarchical-cache"):
        return

    size_gb = args.get("--hicache-size")

    if size_gb is None:
        print(f"KV cache host offload: {args.get('--hicache-ratio')} x the device pool per attention group, which no RAM budget bounds")
    else:
        print(f"KV cache host offload: {size_gb} GB per attention group, out of {machine.total_ram_bytes // GB} GB of RAM")

def build_launch(resolution, machine, *, host, compile_cache_root, kv_cache_min_free_gib, models_dir, extra_args=()):
    family = resolution.family
    sections = engine_args(resolution.config, resolution.engine, family)
    config = merge_config(resolution.config, {"parallelism": sections["parallelism"]}) if sections["parallelism"] else resolution.config

    report_checkpoint_directory(models_dir)
    args = base_args(family, host, models_dir) | model_args(config, family)
    sizing, data_parallel = sizing_args(config, family, machine)
    args |= sizing

    pinned_gib = reserve_host_weights(config, machine, data_parallel)

    if config["kv_cache_offload"]:
        args |= offload_args(family, machine, data_parallel, pinned_gib)
        report_kv_cache_directory(KV_CACHE_DIR, kv_cache_min_free_gib)

    args |= sections["args"]
    utilization = args.get(memory_utilization_flag(family))

    if utilization is not None:
        print(f"GPU memory utilization: {utilization}")

    report_kv_offload(family, args, machine)

    if sections["blackwell_args"]:
        blackwell_count = sum(gpu.is_blackwell for gpu in machine.gpus)
        all_blackwell = bool(machine.gpus) and blackwell_count == len(machine.gpus)
        blackwell_args_state = "enabled" if all_blackwell else "disabled"
        print(f"Blackwell GPUs detected: {blackwell_count} of {len(machine.gpus)}; Blackwell-only arguments: {blackwell_args_state}")

        if all_blackwell:
            args |= sections["blackwell_args"]

    argv = engine_command(resolution) + render_args(args) + list(extra_args)
    env = launch_env(resolution.engine, compile_cache_root, kv_cache_min_free_gib) | render_env(sections["env"])

    return Launch(env, argv)

def detect_gpus():
    command = ["nvidia-smi", "--query-gpu=memory.total,compute_cap", "--format=csv,noheader,nounits"]
    try:
        output = subprocess.run(command, capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return []
    gpus = []
    for line in output.splitlines():
        if line.strip():
            memory_mib, compute_capability = (part.strip() for part in line.split(","))
            gpus.append(Gpu(memory_mib=int(memory_mib), compute_capability=compute_capability))
    return gpus

def total_ram_bytes():
    try:
        limit = Path("/sys/fs/cgroup/memory.max").read_text().strip()
    except OSError:
        limit = "max"
    if limit != "max":
        return int(limit)
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemTotal:"):
            return int(line.split()[1]) * 1024
    raise ServeError("cannot determine the total RAM")

def detect_machine():
    return Machine(gpus=detect_gpus(), total_ram_bytes=total_ram_bytes(), shm_free_bytes=shutil.disk_usage("/dev/shm").free)

def idle_forever():
    print("Model 'none': idling without an engine.", flush=True)
    while True:
        signal.pause()

def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=COMMANDS, help="launch: run the engine (default); get-environment: print MODEL/ENGINE/ENGINE_FAMILY; check: validate the model/engine pair")
    parser.add_argument("--model", help="models/<name>.json, or 'none' to idle; default: $MODEL, then image.json")
    parser.add_argument("--engine", help="engine directory under engines/, or a family name; default: $ENGINE, then image.json, then default_engine")
    parser.add_argument("--dry-run", action="store_true", help="print the environment and command instead of running")
    parser.add_argument("engine_args", nargs="*")
    argv = list(argv)

    if not argv or argv[0] not in COMMANDS:
        argv.insert(0, "launch")

    return parser.parse_args(argv)

def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        resolution = resolve(args.model, args.engine)

        if args.command == "get-environment":
            sys.stdout.write(shell_assignments(resolution))
            return 0

        warn_if_unvalidated(resolution)

        if args.command == "check":
            require_installed(resolution)
            print(f"OK: model '{resolution.model}' on engine '{resolution.engine}'")
            return 0

        if not args.dry_run:
            require_installed(resolution)

        if resolution.config is None:
            idle_forever()

        print(f"Model: {resolution.model} ({resolution.config['hf_repo']}); engine: {resolution.engine}")
        host = "0.0.0.0" if os.environ.get("DEBUG_EXPOSE_VLLM_PUBLICLY") == "1" else "127.0.0.1"
        launch = build_launch(
            resolution,
            detect_machine(),
            host=host,
            extra_args=args.engine_args,
            compile_cache_root=Path(os.environ.get("COMPILE_CACHE_ROOT_DIR", str(WORKSPACE / "compile_cache"))),
            kv_cache_min_free_gib=kv_cache_min_free_gib(KV_CACHE_DIR),
            models_dir=Path(os.environ.get("MODELS_DIR", str(WORKSPACE / "models"))),
        )

    except ServeError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    if args.dry_run:
        for key, value in launch.env.items():
            print(f"export {key}={shlex.quote(value)}")

    print(shlex.join(launch.argv), flush=True)

    if args.dry_run:
        return 0

    if resolution.config["kv_cache_offload"]:
        KV_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    os.environ.update(launch.env)
    os.execvp(launch.argv[0], launch.argv)

if __name__ == "__main__":
    sys.exit(main())
