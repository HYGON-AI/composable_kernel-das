# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# Modified by Hygon Information Technology Co., Ltd.

import subprocess


def __version__():
    hcu_baseline = "0.0.0+hcu"
    hash_width = 6
    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        )[:hash_width]
    except Exception:
        git_hash = "0" * hash_width
    return f"{hcu_baseline}.g{git_hash}"


_package_version = __version__()