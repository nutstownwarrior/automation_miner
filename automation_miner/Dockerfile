ARG BUILD_FROM=ghcr.io/home-assistant/base:latest
# hadolint ignore=DL3006
FROM ${BUILD_FROM}

ENV LANG=C.UTF-8 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    AMMINER_STATE_DIR=/config

# Runtime + build dependencies. Build deps are dropped again in the same layer so
# the shipped image stays small and CPU-friendly.
# hadolint ignore=DL3018
RUN apk add --no-cache python3 py3-pip tzdata libstdc++ \
    && apk add --no-cache --virtual .build-deps build-base python3-dev

WORKDIR /opt/automation_miner

COPY requirements.txt ./
RUN pip3 install --break-system-packages --no-cache-dir -r requirements.txt \
    && apk del --no-network .build-deps \
    && find /usr/lib/python3*/site-packages -name '__pycache__' -type d -prune -exec rm -rf {} + \
    && find /usr/lib/python3*/site-packages -name 'tests' -type d -prune -exec rm -rf {} +

COPY amminer ./amminer
COPY rootfs /

RUN chmod -R +x /etc/s6-overlay/s6-rc.d

ENV PYTHONPATH=/opt/automation_miner

LABEL \
    io.hass.name="Automation Miner" \
    io.hass.description="Mines Home Assistant history for explainable automation suggestions" \
    io.hass.type="addon" \
    io.hass.version="0.1.0"

EXPOSE 8099
