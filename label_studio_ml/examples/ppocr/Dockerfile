# syntax=docker/dockerfile:1
# PP-OCR ML Backend for Label Studio (CPU version)
#
# Uses the official PaddleX Docker image which includes
# PaddlePaddle and PaddleX pre-installed.
#
# Build arguments:
#   TEST_ENV: Set to "true" to install test dependencies

ARG TEST_ENV

FROM ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlex/paddlex:paddlex3.1.2-paddlepaddle3.0.0-cpu

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=9090 \
    PIP_CACHE_DIR=/.cache \
    WORKERS=1 \
    THREADS=8 \
    DEVICE=cpu

# Install base requirements (label-studio-ml and gunicorn)
COPY requirements-base.txt .
RUN --mount=type=cache,target=${PIP_CACHE_DIR},sharing=locked \
    pip install -r requirements-base.txt

# Install PaddleX OCR extra dependencies
RUN --mount=type=cache,target=${PIP_CACHE_DIR},sharing=locked \
    pip install "paddlex[ocr]"

# Install custom requirements (boto3, opencv, etc.)
COPY requirements.txt .
RUN --mount=type=cache,target=${PIP_CACHE_DIR},sharing=locked \
    pip install -r requirements.txt

# Install test requirements if needed
ARG TEST_ENV
COPY requirements-test.txt .
RUN --mount=type=cache,target=${PIP_CACHE_DIR},sharing=locked \
    if [ "$TEST_ENV" = "true" ]; then \
        pip install -r requirements-test.txt; \
    fi

# Copy application code
COPY . .

EXPOSE 9090

CMD gunicorn --preload --bind :$PORT --workers $WORKERS --threads $THREADS --timeout 0 _wsgi:app
