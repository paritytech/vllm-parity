# Needs the `-devel` image instead of `-base` due to FlashInfer needing to JIT compile its kernels.
FROM nvidia/cuda:13.3.1-devel-ubuntu26.04

WORKDIR /app

ENV PATH="/root/.local/bin:${PATH}"
RUN mkdir -p /root/.ssh

# Note: we delete the SSH keys so that they're not built into the Docker image and will regenerate them later.
RUN DEBIAN_FRONTEND="noninteractive" apt-get update && DEBIAN_FRONTEND="noninteractive" apt-get install -y --no-install-recommends curl git ffmpeg build-essential libssl-dev ninja-build openssh-server screen mc caddy python3 && rm -f /etc/ssh/ssh_host_*
RUN curl -LsSf https://astral.sh/uv/install.sh -o install-uv.sh && bash install-uv.sh
ENV UV_HTTP_TIMEOUT=500

RUN mkdir -p --mode=0755 /usr/share/keyrings && curl -fsSL https://pkgs.tailscale.com/stable/ubuntu/resolute.noarmor.gpg -o /usr/share/keyrings/tailscale-archive-keyring.gpg
RUN curl -fsSL https://pkgs.tailscale.com/stable/ubuntu/resolute.tailscale-keyring.list -o /etc/apt/sources.list.d/tailscale.list
RUN DEBIAN_FRONTEND="noninteractive" apt-get update && DEBIAN_FRONTEND="noninteractive" apt-get install -y --no-install-recommends tailscale

ARG ENGINE
RUN test -n "$ENGINE" || { echo "missing 'ENGINE'" >&2; exit 1; }

COPY engines/${ENGINE}/pyproject.toml engines/${ENGINE}/uv.lock engines/${ENGINE}/sync.sh /app/engines/${ENGINE}/
COPY backdate-venv.sh /app/
RUN bash "engines/${ENGINE}/sync.sh" && bash backdate-venv.sh "engines/${ENGINE}"

COPY engines/${ENGINE}/ /app/engines/${ENGINE}/
COPY apply-patches.sh /app/
RUN bash apply-patches.sh "engines/${ENGINE}"

COPY authorized_keys /root/.ssh/authorized_keys
RUN chmod 700 /root/.ssh && chmod 600 /root/.ssh/authorized_keys && echo "Port 53267" > /etc/ssh/sshd_config.d/port.conf
COPY bash_profile /root/.bash_profile

COPY Caddyfile* initialize-* *-cache.sh engine-environment.sh with-logging.sh entrypoint* serve.py launch-* collect-metrics.* reap-kv-cache.py /app
COPY services /app/services
COPY models /app/models
RUN chmod +x /app/entrypoint* && mkdir -p /workspace

ARG MODEL=none
RUN python3 serve.py check --engine "$ENGINE" --model "$MODEL" && printf '{"engine": "%s", "default_model": "%s"}\n' "$ENGINE" "$MODEL" > /app/image.json

EXPOSE 53267 9002 80

CMD ["/app/entrypoint.sh"]
