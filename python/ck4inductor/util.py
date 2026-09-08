# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import functools
from importlib import resources
from pathlib import Path


@functools.lru_cache(None)
def library_path():
    editable_library = Path(__file__).resolve().parents[2] / "library"
    try:
        package_files = resources.files("ck4inductor.library")
        if isinstance(package_files, Path):
            return str(package_files)
        for candidate in getattr(package_files, "_paths", ()):
            candidate = Path(candidate)
            if (candidate / "src").is_dir():
                return str(candidate)
    except ModuleNotFoundError:
        pass
    if editable_library.is_dir():
        return str(editable_library)
    return str(resources.files("ck4inductor.library"))
