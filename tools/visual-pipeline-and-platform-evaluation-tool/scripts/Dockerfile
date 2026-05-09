# SPDX-License-Identifier: Apache-2.0
FROM python:3.12-slim

ARG http_proxy
ARG https_proxy
ARG no_proxy

ENV http_proxy=${http_proxy} \
    https_proxy=${https_proxy} \
    no_proxy=${no_proxy}

# libgl1 and libglib2.0-0 are required by opencv-python on slim base images.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY *.py .
COPY .env .

CMD ["python", "raised_hand_detector.py"]
