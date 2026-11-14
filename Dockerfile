FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libgomp1 llvm \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY packages/mitsuba_video /app/packages/mitsuba_video
RUN python -m pip install "/app/packages/mitsuba_video[usd]"

WORKDIR /data
ENTRYPOINT ["mitsuba-video"]
CMD ["--help"]
