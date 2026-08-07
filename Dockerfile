# Needs the `-devel` image instead of `-base` due to FlashInfer needing to JIT compile its kernels.
FROM nvidia/cuda:13.3.1-devel-ubuntu26.04

WORKDIR /app

ENV PATH="/root/.local/bin:${PATH}"
RUN mkdir -p /root/.ssh

# Note: we delete the SSH keys so that they're not built into the Docker image and will regenerate them later.
RUN DEBIAN_FRONTEND="noninteractive" apt-get update && DEBIAN_FRONTEND="noninteractive" apt-get install -y --no-install-recommends curl build-essential ninja-build openssh-server screen mc caddy && rm -f /etc/ssh/ssh_host_*
RUN curl -LsSf https://astral.sh/uv/install.sh -o install-uv.sh && bash install-uv.sh

COPY pyproject.toml uv.lock /app
RUN uv sync --frozen --no-cache

RUN mkdir -p --mode=0755 /usr/share/keyrings && curl -fsSL https://pkgs.tailscale.com/stable/ubuntu/resolute.noarmor.gpg -o /usr/share/keyrings/tailscale-archive-keyring.gpg
RUN curl -fsSL https://pkgs.tailscale.com/stable/ubuntu/resolute.tailscale-keyring.list -o /etc/apt/sources.list.d/tailscale.list
RUN DEBIAN_FRONTEND="noninteractive" apt-get update && DEBIAN_FRONTEND="noninteractive" apt-get install -y --no-install-recommends tailscale

COPY compile-cache.tgz* /app/

COPY apply-patches.sh *.patch /app
RUN bash apply-patches.sh

COPY authorized_keys /root/.ssh/authorized_keys
RUN chmod 700 /root/.ssh && chmod 600 /root/.ssh/authorized_keys && echo "Port 53267" > /etc/ssh/sshd_config.d/port.conf
COPY bash_profile /root/.bash_profile

COPY Caddyfile* initialize-* *-cache.sh with-logging.sh entrypoint* launch-* collect-metrics.* /app
RUN chmod +x /app/entrypoint* && mkdir -p /workspace

EXPOSE 53267 9002 80

CMD ["/app/entrypoint.sh"]
