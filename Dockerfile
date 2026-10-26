# Revive web app. Runs on Render, Hugging Face Spaces, Nebius or any container host.
#
# The server runs as root only so that it can start every notebook kernel as the
# unprivileged "sandbox" user (see revive/sandbox.py): uploaded notebooks run there
# and cannot read the API keys in the server's environment. On hosts that force a
# non-root user, the sandbox is unavailable and uploads switch themselves off.
FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=7860 \
    UV_PYTHON_INSTALL_DIR=/opt/uv-python \
    REVIVE_SANDBOX_USER=sandbox \
    REVIVE_ALLOW_UPLOADS=1 \
    REVIVE_LIVE_PER_CLIENT=3 \
    REVIVE_LIVE_PER_DAY=20 \
    REVIVE_MAX_CONCURRENT=1 \
    REVIVE_KERNEL_STARTUP_TIMEOUT=180

RUN useradd --create-home --uid 1000 user \
 && useradd --create-home --uid 1001 --shell /usr/sbin/nologin sandbox
WORKDIR /app

COPY pyproject.toml README.md LICENSE requirements-demo.txt ./
COPY revive ./revive
RUN pip install -e . && pip install -r requirements-demo.txt

COPY benchmark ./benchmark
COPY examples ./examples

# The Time Machine rebuilds old environments with uv; fetch the old Pythons once,
# into a folder the sandbox user can read.
RUN python -m uv python install 3.8 3.10 && chmod -R a+rX /opt/uv-python

EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s CMD python -c "import urllib.request, os; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/healthz', timeout=4)"
CMD ["sh", "-c", "exec uvicorn revive.web.app:app --host 0.0.0.0 --port ${PORT} --proxy-headers --forwarded-allow-ips='*'"]
