#!/bin/bash
set -e

MODEL_NAME="${1:-toolgrad_1b}"

# Link CUDA shared libraries
ln -sf /usr/lib/x86_64-linux-gnu/libcuda.so.1 /usr/lib/x86_64-linux-gnu/libcuda.so
/sbin/ldconfig

# Install audio/libsndfile dependencies for vLLM
apt-get update && apt-get install -y libsndfile1
pip install soundfile

# Navigate to the BFCL repository submodule and install in editable mode
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/../../ext/gorilla-for-toolgrad/berkeley-function-call-leaderboard"
pip install -e .

# Run BFCL Generation
bfcl generate --model "$MODEL_NAME" --test-category single_turn --backend vllm --gpu-memory-utilization 0.95 --allow-overwrite "${@:2}"

# Run BFCL Evaluation Checker
bfcl evaluate --model "$MODEL_NAME" --test-category single_turn
