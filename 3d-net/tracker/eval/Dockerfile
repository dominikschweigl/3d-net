# Build context must be the repo root:
#   docker build -f detection/tracker/Dockerfile .
#
# hailo_platform is NOT installed inside the image.
# It is mounted from the RPi host at runtime via docker-compose volumes,
# so the container always uses the same version that's already on the system.

FROM python:3.13-slim

WORKDIR /app

# OpenCV headless runtime dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps (hailo_platform excluded — provided by host mount)
COPY detection/tracker/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy tracker eval code and the inference_service it imports from
COPY detection/tracker/          detection/tracker/
COPY detection/inference_service/ detection/inference_service/
COPY detection/__init__.py        detection/__init__.py

ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["python", "detection/tracker/eval_tracker.py"]
CMD ["--help"]
