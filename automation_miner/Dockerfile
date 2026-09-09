ARG BUILD_FROM=ghcr.io/home-assistant/base:latest
# hadolint ignore=DL3006
FROM ${BUILD_FROM}

ENV LANG=C.UTF-8 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    AMMINER_STATE_DIR=/config

# Every pinned dependency ships a musllinux wheel for both architectures (see
# requirements.txt), so no compiler is needed here at all - that is what keeps
# the image small and the build fast. libstdc++ is the one shared library the
# numpy/scipy/pandas wheels link against.
# hadolint ignore=DL3018
RUN apk add --no-cache python3 py3-pip tzdata libstdc++

WORKDIR /opt/automation_miner

COPY requirements.txt requirements-nodeps.txt ./
RUN pip3 install --break-system-packages --no-cache-dir --only-binary=:all: -r requirements.txt \
    && pip3 install --break-system-packages --no-cache-dir --no-deps -r requirements-nodeps.txt \
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
