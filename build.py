#!/usr/bin/env python3

import argparse
import shlex
import subprocess
import sys

import serve

def main():
    parser = argparse.ArgumentParser(description="Build the Docker image with one engine and a default model.", epilog="Arguments after '--' are passed to docker build.")
    parser.add_argument("model", help="default model for the image: models/<name>.json, or 'none'")
    parser.add_argument("--engine", help="engine directory under engines/, or a family name; default: the model's default_engine")
    parser.add_argument("--tag", default="vllm-parity", help="image tag (default: %(default)s)")
    parser.add_argument("--allow-unvalidated", action="store_true", help="build even though the model is not validated on the engine")
    parser.add_argument("docker_args", nargs="*")
    args = parser.parse_args()

    try:
        resolution = serve.resolve(model=args.model, engine=args.engine, environ={})
    except serve.ServeError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    serve.warn_if_unvalidated(resolution)
    if not resolution.is_validated and not args.allow_unvalidated:
        print("ERROR: the default model is not validated on this engine; pass --allow-unvalidated to build anyway", file=sys.stderr)
        return 1
    for required in ("uv.lock", "sync.sh"):
        if not (resolution.engine_dir / required).is_file():
            print(f"ERROR: {resolution.engine_dir}/{required} does not exist", file=sys.stderr)
            return 1
    if not (resolution.engine_dir / "compile-cache.tgz").is_file():
        print(f"WARNING: {resolution.engine_dir}/compile-cache.tgz does not exist; the image will start with a cold compile cache", file=sys.stderr)

    command = [
        "docker", "build",
        "--build-arg", f"ENGINE={resolution.engine}",
        "--build-arg", f"MODEL={resolution.model}",
        "--tag", args.tag,
        *args.docker_args,
        ".",
    ]

    print(shlex.join(command), flush=True)
    return subprocess.run(command, cwd=serve.APP_DIR).returncode

if __name__ == "__main__":
    sys.exit(main())
