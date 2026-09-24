#!/usr/bin/env bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
build_dir="${BUILD_DIR:-${repo_root}/build}"

cmake -S "${repo_root}" -B "${build_dir}" \
  -DCMAKE_CXX_COMPILER="${CXX:-/opt/dtk/bin/hipcc}" \
  -DGPU_TARGETS="${GPU_TARGETS:-gfx936}" \
  -DBUILD_DEV=ON \
  -DBUILD_EXAMPLE=ON \
  -DBUILD_TEST=OFF

cmake --build "${build_dir}" \
  --target tile_example_gdn_prefill tile_example_gdn_decode \
  -j "${JOBS:-8}"
