#!/bin/bash

# Default configuration
HTTP_PORT=${HTTP_PORT:-8000}
GRPC_PORT=${GRPC_PORT:-8001}
METRICS_PORT=${METRICS_PORT:-8002}
MODEL_REPOSITORY=${MODEL_REPOSITORY:-/models}
MODEL_CONTROL_MODE=${MODEL_CONTROL_MODE:-explicit}
LOG_VERBOSE=${LOG_VERBOSE:-0}
STRICT_MODEL_CONFIG=${STRICT_MODEL_CONFIG:-0}

# Start Triton server
echo "Starting Triton Inference Server with model repository: ${MODEL_REPOSITORY}"

tritonserver \
    --model-repository=${MODEL_REPOSITORY} \
    --http-port=${HTTP_PORT} \
    --grpc-port=${GRPC_PORT} \
    --metrics-port=${METRICS_PORT} \
    --model-control-mode=${MODEL_CONTROL_MODE} \
    --log-verbose=${LOG_VERBOSE} \
    --strict-model-config=${STRICT_MODEL_CONFIG} \
    --allow-gpu-metrics=true \
    --allow-metrics=true \
    "$@" 