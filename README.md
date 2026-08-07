# Building and uploading

```
$ ./build.sh
$ docker tag vllm-parity $USERNAME/vllm-parity:$VERSION
$ docker push $USERNAME/vllm-parity:$VERSION
```

# Environment variables

- `MODEL` (required) -- the model to serve/script to launch (will launch: `launch-vllm-$MODEL.sh`)
    - `none` -- launches nothing; will loop forever, good for testing
    - `smollm2-135m-instruct` -- tiny, dumb model; good for testing
    - `deepseek-v4-flash-0731`
- `HF_TOKEN` (optional, recommended for production) -- HuggingFace token key (only needs read-only permissions); speeds up downloads
- `CADDY_API_KEY` (optional) -- the API key required to access the vLLM instance publicly (arbitrary, you generate this yourself) through the exposed HTTPS reverse proxy; proxy doesn't start at all when not set
- `TS_AUTHKEY` (optional) -- the auth key to automatically join a Tailscale network (only if you want to use Tailscale); Tailscale doesn't start at all when not set
- `TS_HOSTNAME` (optional) -- the hostname with which to join a Tailscale network
- `DEBUG_KEEP_CONTAINER_ALIVE` (optional) -- when set to `1` will not exit when an error is encountered and/or the vLLM process is stopped
- `DEBUG_EXPOSE_VLLM_PUBLICLY` (optional, UNSAFE) -- when set to `1` the vLLM process will listen on `0.0.0.0:9001` instead of `127.0.0.1:9001`

# Testing out locally

First run `build.sh`, then:

```
$ mkdir -p /tmp/workspace
$ ./run-local.sh -v /tmp/workspace:/workspace -e CADDY_API_KEY=hello -e MODEL=smollm2-135m-instruct
```

The `run-local.sh` script will launch the container's default entrypoint.

The `-e` is used to pass environment variables into the container.

The `-v /tmp/workspace:/workspace` is optional. The image stores its cache and the model weights in `/workspace`,
so this argument will make it persist between invocations, skipping unnecessary redownloads/recalculations.

While the container is running you can connect to it with `ssh -p 10003 root@127.0.0.1`,
just make sure to first add your public SSH key to `authorized_keys` and rebuild the image.

The vLLM instance is launched under `screen`. You can attach to it by running `screen -r vllm` inside the container.
You can detach from it by pressing `Ctrl+A` and then `D`.

You can also run `run-local-interactive.sh` to launch an interactive shell inside the container.

```
$ mkdir -p /tmp/workspace
$ ./run-local-interactive.sh -v /tmp/workspace:/workspace
```

From there you can manually run `entrypoint.sh` (which is what the non-interactive script runs by default)
or you can directly launch vLLM with e.g. `bash launch-vllm-smollm2-135m-instruct.sh`.

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

Add your SSH key to `authorized_keys`. Then build and upload the image.

(Optional, but highly recommended): if you don't have a HuggingFace token: install HuggingFace CLI on your PC with `uv tool install hf`, then run `hf auth login`, then print out your token with `hf auth token`.

Then create a new template on Runpod:

- Container image: the tag you've pushed to Dockerhub
- Temporary storage: 5GB (the minimum, we're not going to store anything there)
- Persistent storage: set appropriate size in GB (should at least be enough to hold the model you're going to run, plus extra space for KV cache disk offload)
- Environment variables:
    - Add `MODEL` and set it to whichever model you want to run
    - Add `HF_TOKEN` and put its value through the little key icon as a secret (not strictly necessary to add, but highly recommended if you want to launch a big model as it'll download faster)
    - Add `CADDY_API_KEY` (optional); generate a random key with `openssl rand -hex 32` (should also be saved as a secret)
    - Add `TS_AUTHKEY` (optional) if you want to connect through Tailscale (should also be saved as a secret)
    - Add `TS_HOSTNAME` (optional) if you want it to have a preset hostname by default on your Tailscale network
- Networking configuration (TCP ports):
    - Label "SSH", port "53267" (it's running on a non-standard port to not get spammed by automated Internet-wide scans in case it gets directly exposed)
    - Label "vLLM Caddy Proxy", port "9002" (optional, add only if you've set the `CADDY_API_KEY` environment variable)

So, for example, assuming you named the secrets `HF_TOKEN` and `CADDY_API_KEY` you'll have:

```
MODEL = smollm2-135m-instruct
HF_TOKEN = {{ RUNPOD_SECRET_HF_TOKEN }}
CADDY_API_KEY = {{ RUNPOD_SECRET_CADDY_API_KEY }}
```

Important: on Runpod you should only use the exposed Caddy proxy server for one-off access! Runpod's networking is not very good, so it's not appropriate to use directly in production.
It is perfectly safe to expose, however once you have a substantial amount of connections it will start randomly dropping them and/or new connections will spuriously fail.

# Metrics

There's a metrics collector that will automatically start and harvest some metrics periodically.
You can use the `analyze-metrics.py` to analyze them. Unlike the rest of the repository, both are mostly vibe-coded.

# Baking the cache into the image

Launch vLLM takes a very long time on a cold cache (over half an hour). The image's set up so that its cache is located in `/workspace/compile_cache`.
You can run `gather-cache.sh` to pack the contents of that directory, copy the resulting archive here, and then rebuild the image.
On the very next startup this baked-in cache will be copied into `/workspace/compile_cache` before vLLM is started, significantly speeding up the startup time.
Make sure to also warm it up with actual requests before copying the caches, because some kernels are only JIT compiled while there are pending requests.
