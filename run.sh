#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD="$ROOT/build"

missing=()
command -v cmake >/dev/null 2>&1 || missing+=(cmake)
command -v c++ >/dev/null 2>&1 || missing+=(g++)
command -v glslc >/dev/null 2>&1 || command -v glslangValidator >/dev/null 2>&1 || missing+=(glslang-tools)

if ((${#missing[@]})); then
  echo "Missing build tools: ${missing[*]}"
  echo "On Ubuntu/Debian install:"
  echo "  sudo apt install build-essential cmake libvulkan-dev vulkan-tools mesa-vulkan-drivers libglfw3-dev libglm-dev glslang-tools"
  exit 1
fi

cmake -S "$ROOT" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release
cmake --build "$BUILD" -j"$(nproc)"
exec "$BUILD/vsb-clock"
