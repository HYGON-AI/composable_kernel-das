// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
#include "ck/host_utility/multidie_kernel_launch.hpp"

struct Probe
{
    __device__ void operator()(int64_t* output, int64_t rows, double value) const
    {
        extern __shared__ int64_t scratch[];
        scratch[threadIdx.x] = rows + static_cast<int64_t>(value) + threadIdx.x;
        __syncthreads();
        if(threadIdx.x == 0)
            output[blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z)] =
                scratch[blockDim.x - 1];
    }
};

template <typename F>
void expect_error(F fn, const char* message)
{
    try { fn(); }
    catch(const std::exception& e)
    {
        if(std::string(e.what()).find(message) != std::string::npos)
            return;
        throw;
    }
    throw std::runtime_error(std::string("expected error: ") + message);
}

int main()
{
    try
    {
        const auto& info = ck_tile::get_hip_device_info();
        const bool multi = CK_ENABLE_MULTIDIE && info.arch_name == "gfx946" && info.num_xcds >= 2;
        unsetenv("GPU_CHIP"); // Once initialized, only the HIP device selects routing.
        hipStream_t stream;
        HIP_CHECK_ERROR(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
        int64_t* output = nullptr;
        HIP_CHECK_ERROR(hipMalloc(&output, 5 * sizeof(int64_t)));
        const auto kernel = ck_tile::kentry<64, 1, Probe, int64_t*, int64_t, double>;
        const int32_t rows = 0x12345678;
        const float value = 7.0f;
        int checks = 0;
        for(const auto& p : {ck::launch_policy::layernorm, ck::launch_policy::rmsnorm,
                            ck::launch_policy::w8a8, ck::launch_policy::moe_sorting})
        {
            auto launch = [&](dim3 grid) {
                ck::multidie_detail::launch(p, kernel, grid, dim3(64), 64 * sizeof(int64_t),
                                            stream, output, rows, value);
            };
            auto verify = [&](dim3 grid) {
                launch(grid);
                HIP_CHECK_ERROR(hipStreamSynchronize(stream));
                int64_t result[5]{};
                const unsigned groups = grid.x * grid.y * grid.z;
                HIP_CHECK_ERROR(hipMemcpy(result, output, groups * sizeof(int64_t), hipMemcpyDeviceToHost));
                for(unsigned i = 0; i < groups; ++i)
                    if(result[i] != int64_t(rows) + 7 + 63)
                        throw std::runtime_error("kernel ABI/dynamic LDS mismatch");
                ++checks;
            };
            unsetenv(p.mode_env);
            setenv(p.chunk_env, "1", 1); setenv(p.mask_env, "0", 1); setenv(p.log_env, "1", 1);
            verify(dim3(5));
            setenv(p.mode_env, "auto", 1);
            setenv(p.chunk_env, "invalid", 1); setenv(p.mask_env, "invalid", 1);
            verify(dim3(1)); // Unused settings are ignored for a single TG.
            if(multi && p.count_all_grid_dimensions)
            {
                expect_error([&] { launch(dim3(1, 2)); }, "invalid unsigned"); ++checks;
            }
            else
                verify(dim3(1, 2)); // Sorting auto threshold counts only grid.x.
            setenv(p.mode_env, "off", 1);
            verify(dim3(5));
            setenv(p.mode_env, "invalid", 1);
            expect_error([&] { launch(dim3(5)); }, "must be auto"); ++checks;
            if(p.skip_empty_grid)
            {
                expect_error([&] { launch(dim3(0)); }, "must be auto"); ++checks;
                setenv(p.mode_env, "off", 1);
                launch(dim3(0)); ++checks;
            }
            setenv(p.mode_env, "linear", 1);
            if(multi)
            {
                setenv(p.mask_env, "0", 1);
                for(const char* bad : {"", "-1", "4294967296", "1tail"})
                {
                    setenv(p.chunk_env, bad, 1);
                    expect_error([&] { launch(dim3(5)); }, "invalid unsigned"); ++checks;
                }
                setenv(p.chunk_env, "0", 1);
                expect_error([&] { launch(dim3(5)); }, "must be positive"); ++checks;
                setenv(p.chunk_env, "2", 1); setenv(p.mask_env, "1", 1);
                expect_error([&] { launch(dim3(5)); }, "all dies"); ++checks;
                setenv(p.mask_env, "0", 1);
                verify(dim3(5)); verify(dim3(1, 2));
            }
            else
            {
                expect_error([&] { launch(dim3(5)); }, "requires"); ++checks;
            }
            unsetenv(p.mode_env); unsetenv(p.chunk_env); unsetenv(p.mask_env); unsetenv(p.log_env);
        }
        // Exercise both public adapters, plus the unchanged generic CK Tile
        // factory, in this same translation unit. No AITER/Python dependency.
        auto verify_adapter = [&] {
            HIP_CHECK_ERROR(hipStreamSynchronize(stream));
            int64_t result[2]{};
            HIP_CHECK_ERROR(hipMemcpy(result, output, sizeof(result), hipMemcpyDeviceToHost));
            if(result[0] != int64_t(rows) + 70 || result[1] != int64_t(rows) + 70)
                throw std::runtime_error("adapter mismatch");
            HIP_CHECK_ERROR(hipMemsetAsync(output, 0, sizeof(result), stream));
            ++checks;
        };
        HIP_CHECK_ERROR(hipMemsetAsync(output, 0, 2 * sizeof(int64_t), stream));
        ck_tile::stream_config tile_config{stream, false};
        ck_tile::launch_kernel(tile_config,
            ck_tile::make_multidie_kernel<64, 1>(ck::launch_policy::rmsnorm, Probe{},
                dim3(2), dim3(64), 64 * sizeof(int64_t), output, int64_t(rows), double(value)));
        verify_adapter();
        ck_tile::launch_kernel(tile_config, ck_tile::make_kernel<64, 1>(Probe{},
            dim3(2), dim3(64), 64 * sizeof(int64_t), output, int64_t(rows), double(value)));
        verify_adapter();
        StreamConfig legacy_config{stream, false};
        ck::launch_and_time_multidie_kernel(ck::launch_policy::w8a8, legacy_config,
            kernel, dim3(2), dim3(64), 64 * sizeof(int64_t), output, int64_t(rows), double(value));
        verify_adapter();
#if CK_ENABLE_MULTIDIE
        legacy_config.time_kernel_ = true;
        setenv(ck::launch_policy::w8a8.mode_env, "off", 1);
        expect_error([&] {
            ck::launch_and_time_multidie_kernel(ck::launch_policy::w8a8, legacy_config,
                kernel, dim3(2), dim3(64), 64 * sizeof(int64_t), output, int64_t(rows), double(value));
        }, "timing is not implemented");
        ++checks;
#endif
        HIP_CHECK_ERROR(hipFree(output));
        HIP_CHECK_ERROR(hipStreamDestroy(stream));
        std::cout << "PASS checks=" << checks << " arch=" << info.arch_name
                  << " xcds=" << info.num_xcds << " multidie_build=" << CK_ENABLE_MULTIDIE << '\n';
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
