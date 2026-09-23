#!/bin/sh
# Restore the upstream DeepSeek reference checkout into .refs/ (shallow).
set -e
cd "$(dirname "$0")/.."
mkdir -p .refs
[ -d .refs/deepseek-harness ] || git clone --depth 1 https://github.com/deepseek-ai/deepseek-harness .refs/deepseek-harness
