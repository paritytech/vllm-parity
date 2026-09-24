# Building and uploading

```
$ ./build.py deepseek-v4-flash-0731
$ docker tag vllm-parity $USERNAME/vllm-parity:$VERSION
$ docker push $USERNAME/vllm-parity:$VERSION
```

`--engine` picks the engine; without it the model's own default is used. The image build installs
the engine by running `engines/<engine>/sync.sh`, so each engine states how it is built. Most
of them install from a published wheel in about a minute. `sglang-2026-09-17-84cba45bf` is pinned
to a git commit instead, because SGLang has not released DeepSeek-V4.1 support yet, so its script
fetches a Rust toolchain, compiles SGLang (about six minutes) and deletes the toolchain again.

# Environment variables

- `MODEL` (optional) -- the model to serve; default: the image's default model
    - `none` -- launches nothing; will loop forever, good for testing
    - `smollm2-135m-instruct` -- tiny, dumb model; good for testing
    - `deepseek-v4-flash-0731`
    - `qwen3.8-flash-next`
    - `deepseek-v4.1-flash` -- text and images; needs 614 GB of VRAM (one GB200 NVL4 tray, or an 8-GPU H200 node); serves on vLLM, and `ENGINE=sglang` selects the SGLang build of it, which nobody has run yet
    - `deepseek-v4.1-flash-text-only` -- the same checkpoint without the vision tower, which frees its VRAM for KV cache
- `HF_TOKEN` (optional, recommended for production) -- HuggingFace token key (only needs read-only permissions); speeds up downloads
- `CADDY_API_KEY` (optional) -- the API key required to access the vLLM instance publicly (arbitrary, you generate this yourself) through the exposed HTTPS reverse proxy; proxy doesn't start at all when not set
- `TS_AUTHKEY` (optional) -- the auth key to automatically join a Tailscale network (only if you want to use Tailscale); Tailscale doesn't start at all when not set
- `TS_HOSTNAME` (optional) -- the hostname with which to join a Tailscale network
- `SSH_TUNNEL_HOST` (optional) -- the host to open a persistent reverse SSH tunnel to; passed to `ssh` as-is, so `host`, `user@host` and `ssh://user@host:2222` all work; the tunnel doesn't start at all when not set
- `SSH_TUNNEL_REMOTE_VLLM_PORT` (optional) -- the port (or interface address *and* port, separated by a colon) to open on the remote host, forwarded straight to vLLM (bypassing the Caddy proxy, so no API key is checked)
- `SSH_TUNNEL_PRIVATE_KEY` (optional) -- the private SSH key to authenticate to that host with, base64-encoded (a raw PEM works too); must not have a passphrase
- `SSH_TUNNEL_HOST_KEY` (optional) -- the remote's public host key, so that it doesn't have to be trusted on first connect; when not set the first connection's key is remembered and required afterwards
- `KV_CACHE_MIN_FREE_GIB` (optional, default 32) -- how much space to keep free on `/workspace` for the disk KV cache tier
- `KV_CACHE_REAP_INTERVAL` (optional, default 300) -- how often (in seconds) the KV cache reaper (`reap-kv-cache.py`) should trigger (vLLM only)
- `DEBUG_KEEP_CONTAINER_ALIVE` (optional) -- when set to `1` will not exit when an error is encountered and/or the engine process is stopped
- `DEBUG_EXPOSE_VLLM_PUBLICLY` (optional, UNSAFE) -- when set to `1` the engine will listen on `0.0.0.0:9001` instead of `127.0.0.1:9001`

# Testing out locally

```
$ ./build.py smollm2-135m-instruct --engine vllm
$ mkdir -p /tmp/workspace
$ ./run-local.sh -v /tmp/workspace:/workspace -e CADDY_API_KEY=hello
```

The `run-local.sh` script will launch the container's default entrypoint.

The `-e` is used to pass environment variables into the container; `-e MODEL=...` overrides the image's default model.

The `-v /tmp/workspace:/workspace` is optional. The image stores its cache and the model weights in `/workspace`,
so this argument will make it persist between invocations, skipping unnecessary redownloads/recalculations.

`-e MODELS_DIR=...` puts the checkpoints somewhere other than `/workspace/models`. They are read in full on every
start, so a network volume makes startup as slow as that volume; `serve.py` prints which filesystem serves the
directory it was given and warns when that is a network or FUSE mount.

While the container is running you can connect to it with `ssh -p 10003 root@127.0.0.1`,
just make sure to first add your public SSH key to `authorized_keys` and rebuild the image.

The engine is launched under `screen`. You can attach to it by running `screen -r engine`
inside the container. You can detach from it by pressing `Ctrl+A` and then `D`.

You can also run `run-local-interactive.sh` to launch an interactive shell inside the container.

```
$ mkdir -p /tmp/workspace
$ ./run-local-interactive.sh -v /tmp/workspace:/workspace
```

From there you can manually run `entrypoint.sh` (which is what the non-interactive script runs by default)
or you can directly launch the engine with `python3 serve.py` with the image's default model.

These scripts expose the following ports (deliberately diffent on the host and in the container to prevent accidentally running a command meant for inside on the outside, or the other way around):

- 10001 -> 9001 -- raw vLLM OpenAI-compatible server (needs `DEBUG_EXPOSE_VLLM_PUBLICLY` to be set to work)
- 10002 -> 9002 -- Caddy-based HTTPS proxy over LLM; safe to expose to the Internet, requires API key, only starts if `CADDY_API_KEY` was set
- 10003 -> 53267 -- SSH server

You can test out that vLLM works by doing this on the container: (note that it takes a while for vLLM to start)

```
$ curl http://localhost:9001/v1/chat/completions -H "Content-Type: application/json" -d '{"messages":[{"role":"user","content":"hello"}]}'
```

...or this on the host: (make sure the API key matches with what you've given to `run-local.sh`)

```
$ curl -k https://localhost:10002/v1/chat/completions -H "Authorization: Bearer hello" -H "Content-Type: application/json" -d '{"messages":[{"role":"user","content":"hello"}]}'
```

# Setting up a Runpod template

Add your SSH key to `authorized_keys` (if you want to connect to the pod through SSH). Then build and upload the image.

(Optional, but highly recommended): if you don't have a HuggingFace token: install HuggingFace CLI on your PC with `uv tool install hf`, then run `hf auth login`, then print out your token with `hf auth token`.

Then create a new template on Runpod:

- Container image: the tag you've pushed to Dockerhub
- Temporary storage: 5GB (the minimum, we're not going to store anything there)
- Persistent storage: set appropriate size in GB (should at least be enough to hold the model you're going to run, plus extra space for KV cache disk offload)
- Environment variables:
    - Add `MODEL` (optional) if you want a model other than the image's default
    - Add `HF_TOKEN` and put its value through the little key icon as a secret (not strictly necessary to add, but highly recommended if you want to launch a big model as it'll download faster)
    - Add `CADDY_API_KEY` (optional); generate a random key with `openssl rand -hex 32` (should also be saved as a secret)
    - Add `TS_AUTHKEY` (optional) if you want to connect through Tailscale (should also be saved as a secret)
    - Add `TS_HOSTNAME` (optional) if you want it to have a preset hostname by default on your Tailscale network
    - Add `SSH_TUNNEL_HOST` (optional) if you want to create a reverse SSH tunnel for the vLLM instance directly; will connect to this address through SSH
    - Add `SSH_TUNNEL_HOST_KEY` (optional) the public key of the remote host
    - Add `SSH_TUNNEL_REMOTE_VLLM_PORT` (optional) with the port on which vLLM will be accessible on the *remote* host
    - Add `SSH_TUNNEL_PRIVATE_KEY` (optional) a private key encoded through `base64 -w0`, so that the remote host can authorize the pod (should also be saved as a secret)
- Networking configuration (TCP ports):
    - Label "SSH", port "53267" (it's running on a non-standard port to not get spammed by automated Internet-wide scans in case it gets directly exposed)
    - Label "vLLM Caddy Proxy", port "9002" (optional, add only if you've set the `CADDY_API_KEY` environment variable)

So, for example, assuming you named the secrets `HF_TOKEN` and `CADDY_API_KEY` you'll have:

```
HF_TOKEN = {{ RUNPOD_SECRET_HF_TOKEN }}
CADDY_API_KEY = {{ RUNPOD_SECRET_CADDY_API_KEY }}
SSH_TUNNEL_HOST = tunnel@vps.example.com
SSH_TUNNEL_HOST_KEY = ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAA
SSH_TUNNEL_REMOTE_VLLM_PORT = 9101
SSH_TUNNEL_PRIVATE_KEY = {{ RUNPOD_SECRET_SSH_TUNNEL_PRIVATE_KEY }}
```

Important: on Runpod you should only use the exposed Caddy proxy server for one-off access! Runpod's networking is not very good, so it's not appropriate to use directly in production.
It is perfectly safe to expose, however once you have a substantial amount of connections it will start randomly dropping them and/or new connections will spuriously fail.

# Metrics

There's a metrics collector that will automatically start and harvest some metrics periodically.
You can use the `analyze-metrics.py` to analyze them. Unlike the rest of the repository, both are mostly vibe-coded.

# Baking the cache into the image

Launching an engine takes a very long time on a cold cache (over half an hour). The image's set up so that its cache is located in `/workspace/compile_cache/<engine>`.
You can run `gather-cache.sh` to pack the contents of that directory, copy the resulting archive to `engines/<engine>`, and then rebuild the image.
On the very next startup this baked-in cache will be copied into the workspace before the engine is started, significantly speeding up the startup time.
Make sure to also warm it up with actual requests before copying the caches, because some kernels are only JIT compiled while there are pending requests.

FlashInfer's part of that cache is a ninja build tree whose inputs are the `.cu` files in the engine's `.venv`, and ninja rebuilds any object older than its sources.
`uv` stamps every file it installs with the install time, so an image build that reruns `uv sync` would otherwise leave the baked cache stale and cost ~20 minutes of `nvcc` at the next startup.
`backdate-venv.sh` gives every file under the `.venv` a constant mtime, so the cache stays valid for as long as the engine directory does.
It has to run in the same `RUN` as `uv sync`, which is why the two are chained: a changed mtime puts the file in that layer's changeset, and a layer stores whole files, so back-dating from a later `RUN` writes a second copy of the whole venv into the image.
Back-dating 63823 entries takes 0.8 seconds; doing it one layer too late costs minutes and 4.8 GB.
Patches apply after that and keep their build time, so a patched source still rebuilds what depends on it.
