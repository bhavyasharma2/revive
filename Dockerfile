# Revive web app. Runs on Hugging Face Spaces (port 7860), Nebius Serverless
# Endpoints, or any container host.
FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=7860 \
    REVIVE_LIVE_PER_CLIENT=3 \
    REVIVE_LIVE_PER_DAY=20 \
    REVIVE_MAX_CONCURRENT=1

# Hugging Face Spaces runs containers as uid 1000.
RUN useradd --create-home --uid 1000 user
WORKDIR /app

COPY --chown=user pyproject.toml README.md LICENSE requirements-demo.txt ./
COPY --chown=user revive ./revive
RUN pip install -e . && pip install -r requirements-demo.txt

COPY --chown=user benchmark ./benchmark
COPY --chown=user examples ./examples

USER user
ENV HOME=/home/user
# The Time Machine rebuilds old environments with uv; fetch the old Pythons once at build time.
RUN python -m uv python install 3.8 3.10
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s CMD python -c "import urllib.request, os; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/healthz', timeout=4)"
CMD ["sh", "-c", "uvicorn revive.web.app:app --host 0.0.0.0 --port ${PORT} --proxy-headers --forwarded-allow-ips='*'"]
