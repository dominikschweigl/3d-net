# Build context must be the repo root (same as metadata_service):
#   docker build -f detection/inference_service/Dockerfile .
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
COPY detection/inference_service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the service and shared detection models
COPY detection/inference_service/ detection/inference_service/
COPY detection/__init__.py        detection/__init__.py

# ZMQ PUB port
EXPOSE 5556

CMD ["python", "-m", "detection.inference_service"]
