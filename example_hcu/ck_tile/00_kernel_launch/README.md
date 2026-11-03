# CK multi-die launch regression

The shared implementation is `include/ck/host_utility/multidie_kernel_launch.hpp`.
`CK_ENABLE_MULTIDIE=1` requires the gfx946 HIP SDK. Runtime HIP device properties
select the route. The default is 0, which compiles without multi-die API symbols.
Existing generic CK/CK Tile launchers remain independent of this opt-in interface.

Build in a separate directory using the CK development configuration:

```bash
ck=/path/to/composable_kernel
mkdir -p build && cd build
CK_GPU_TARGETS=gfx946 bash "$ck/script/cmake-ck-dev.sh" "$ck"
cmake --build . --target tile_test_multidie_launch -j2
./bin/tile_test_multidie_launch
```

Use gfx936/gfx938 for physical HCU builds. For Perf Model, configure its SDK
environment and select `GPU_CHIP=sb` or `sbx2` before launching the executable.
The test initializes HIP, then removes GPU_CHIP to verify routing uses HIP
topology. Each model run needs its own output directory.

The standalone C++ test needs only CK and HIP. It checks typed argument
conversion, non-default streams, dynamic LDS, multidimensional grids, default/
auto/off/linear modes, malformed settings, unsupported topology, empty sorting
grids, both public adapters and the unchanged generic CK Tile factory. It uses
small inputs and no benchmark loop. It does not validate Graph replay or speed.

LayerNorm/RMSNorm/W8A8 retain a total-grid auto threshold of two TGs; sorting
retains its grid.x threshold and empty-grid behavior. Their existing environment
variable names are unchanged. W8A8's opt-in legacy adapter still rejects internal
timing, even in off mode. Other legacy CK consumers retain their original timer
when the macro is 0.

Callers select a policy rather than including an operator-specific launch file:

```cpp
#include "ck/host_utility/multidie_kernel_launch.hpp"

// CK Tile: leave launch_kernel's timer and kentry bounds at the call site.
ck_tile::launch_kernel(stream_config,
    ck_tile::make_multidie_kernel<256, 1>(ck::launch_policy::layernorm,
        Kernel{}, grid, block, lds_bytes, kargs));

// Legacy CK: preserve its ordinary timer when the build does not opt in.
ck::launch_and_time_multidie_kernel(ck::launch_policy::w8a8,
    stream_config, kernel, grid, block, lds_bytes, kargs);
```

For a new operator, add a policy only after checking its grid threshold, empty
grid behavior and ordinary API requirements. Enable the common build flag on
that operator's gfx946 target; do not enable it globally for unrelated targets.
