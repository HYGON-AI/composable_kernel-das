// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.
// Copyright (c) 2026 Hygon Info Technologies Ltd.

#include <limits>
#include <iostream>
#include <getopt.h>
#include <cmath>
#include <cstdint>
#include <cstring>

#include "ck/ck.hpp"
#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/utility/host_common_util.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_batchnorm_backward.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_batchnorm_backward_impl_V2.hpp"

static struct option long_options[] = {{"inOutLengths", required_argument, nullptr, 'D'},
                                       {"verify", required_argument, nullptr, 'v'},
                                       {"help", no_argument, nullptr, '?'},
                                       {nullptr, 0, nullptr, 0}};

class BatchNormBwdArg
{
    private:
    int option_index = 0;

    public:
    std::vector<size_t> inOutLengths;

    bool do_verification = false;

    bool haveSavedMeanInvVar;

    int data_type               = 0;
    int init_method             = 3;
    bool time_kernel            = false;
    bool use_multiblock_welford = false;

    public:
    void show_usage(const char* cmd)
    {
        // clang-format off
        std::cout << "Usage of " << cmd << std::endl;
        std::cout << "--inOutLengths or -D, comma separated list of input tensor dimension lengths, must have 4 integers for nchw" << std::endl;
        std::cout << "--verify or -v, 1/0 to indicate whether to verify the result by comparing with the host-based batch-normalization" << std::endl;
        std::cout << "Arg1: data type (0: fp16)" << std::endl;
        std::cout << "Arg2 -- 1/0 to indicate whether to use saved mean and invVariance" << std::endl;
        std::cout << "Arg3 -- init method used for dy and bnScale (0=no init, 1=single integer value, 2=scope integer value, 3=decimal value)" << std::endl;
        std::cout << "Arg4 -- time kernel (0=no, 1=yes)" << std::endl;
        std::cout << "Arg5: use multi-block welford (0=n0, 1=yes)" << std::endl;
        // clang-format on
    };

    int processArgs(int argc, char* argv[])
    {
        using ck::host_common::getTypeValuesFromString;

        int ch;

        while(1)
        {
            ch = getopt_long(argc, argv, "D:v:", long_options, &option_index);
            if(ch == -1)
                break;
            switch(ch)
            {
            case 'D':
                if(!optarg)
                    throw std::runtime_error("Invalid option format!");

                inOutLengths = getTypeValuesFromString<size_t>(optarg);

                if(inOutLengths.size() != 4)
                    throw std::runtime_error(
                        "NCHW tensor layout should have 4 length values specified!");
                break;
            case 'v':
                if(!optarg)
                    throw std::runtime_error("Invalid option format!");

                do_verification = static_cast<bool>(std::atoi(optarg));
                break;
            case '?':
                if(std::string(long_options[option_index].name) == "help")
                {
                    show_usage(argv[0]);
                    return (-1);
                };
                break;
            default: show_usage(argv[0]); return (-1);
            };
        };

        if(optind + 5 > argc)
            throw std::runtime_error("Invalid cmd-line arguments, more argumetns are needed!");

        data_type              = std::atoi(argv[optind++]);
        haveSavedMeanInvVar    = std::atoi(argv[optind++]);
        init_method            = std::atoi(argv[optind++]);
        time_kernel            = static_cast<bool>(std::atoi(argv[optind++]));
        use_multiblock_welford = static_cast<bool>(std::atoi(argv[optind]));
        if(use_multiblock_welford == false){
            printf("single_block not suppported.\n");
            return (-1);
        }

        return (0);
    };
};

using namespace ck;

template <typename T>
float host_to_float(T v)
{
    return static_cast<float>(v);
}

// CPU reference/verification 需要和 host_tensor_generator.hpp 使用同一种 fp16
// bits 解释方式，避免 CK example 的 host 生成/读取/check_err 路径与 device
// 看到的 half payload 不一致。
template <>
float host_to_float<ck::half_t>(ck::half_t v)
{
    uint16_t bits = 0;
    std::memcpy(&bits, &v, sizeof(bits));

    const int sign = (bits & 0x8000) ? -1 : 1;
    const int exp  = (bits >> 10) & 0x1f;
    const int mant = bits & 0x03ff;

    if(exp == 0)
        return sign * std::ldexp(static_cast<float>(mant), -24);
    if(exp == 0x1f)
        return mant == 0 ? sign * std::numeric_limits<float>::infinity()
                         : std::numeric_limits<float>::quiet_NaN();

    return sign * std::ldexp(static_cast<float>(1024 + mant), exp - 25);
}

template <typename T>
T host_from_float(float v)
{
    return static_cast<T>(v);
}

// 与 host_tensor_generator.hpp 中显式 fp16 编码保持一致，确保 dx_ref 的 bits
// 和 device 输出使用同一种表示。
template <>
ck::half_t host_from_float<ck::half_t>(float v)
{
    return host_float_to_half(v);
}

template <typename XDataType, typename AccDataType, bool UseMultiblockInK>
bool bnorm_bwd_nhwc_test(bool do_verification,
                         int init_method,
                         bool time_kernel,
                         const std::vector<size_t> inOutLengths,
                         bool haveSavedMeanInvVar,
                         double epsilon)
{
    // for NHWC BatchNorm calculation of mean and meansquare
    constexpr index_t Rank         = 4;
    constexpr index_t NumReduceDim = 3;

    using ScaleDataType = XDataType;

    const std::vector<size_t> scaleBiasMeanVarLengths = {inOutLengths[1]};

    // input data of the batchnorm backward algorithm
    Tensor<XDataType> x(inOutLengths);
    Tensor<XDataType> dy(inOutLengths);

    Tensor<ScaleDataType> bnScale(scaleBiasMeanVarLengths);

    Tensor<AccDataType> savedMean(scaleBiasMeanVarLengths);
    Tensor<AccDataType> savedInvVar(scaleBiasMeanVarLengths);
    // savedVariance is only used for initializing savedInvVar
    Tensor<AccDataType> savedVariance(scaleBiasMeanVarLengths);

    // output data of the batchnorm backward algorithm
    Tensor<XDataType> dx_ref(inOutLengths);
    Tensor<XDataType> dx(inOutLengths);

    Tensor<AccDataType> dscale(scaleBiasMeanVarLengths);
    Tensor<AccDataType> dbias(scaleBiasMeanVarLengths);

    Tensor<AccDataType> dscale_ref(scaleBiasMeanVarLengths);
    Tensor<AccDataType> dbias_ref(scaleBiasMeanVarLengths);

    auto inOutStrides            = dy.mDesc.GetStrides();
    auto scaleBiasMeanVarStrides = dscale.mDesc.GetStrides();

    std::size_t num_thread = std::thread::hardware_concurrency();

    if(haveSavedMeanInvVar)
    {
        const float x_mean       = 0.0f;
        const float x_stddev     = 1.0f;
        const float noise_stddev = 0.0001f;

        // input data in normal distribution
        x.GenerateTensorValue(GeneratorTensor_4<XDataType>{x_mean, x_stddev}, num_thread);

        // initialize the savedMean to be values with tiny variation to the mean of the x values
        savedMean.GenerateTensorValue(GeneratorTensor_4<AccDataType>{x_mean, noise_stddev},
                                      num_thread);

        // initialize the variance to be values with tiny variation to the variance of the x values
        savedVariance.GenerateTensorValue(
            GeneratorTensor_4<AccDataType>{x_stddev * x_stddev, noise_stddev}, num_thread);

        auto it_src       = savedVariance.mData.begin();
        auto it_dst       = savedInvVar.mData.begin();
        float tmp_epsilon = std::numeric_limits<float>::epsilon();

        while(it_src != savedVariance.mData.end())
        {
            *it_dst = type_convert<AccDataType>(
                1.0f / std::sqrtf(type_convert<float>(*it_src) + tmp_epsilon));

            it_src++;
            it_dst++;
        };
    }
    else
    {
        const float x_mean   = 0.0f;
        const float x_stddev = 1.0f;

        // input data in normal distribution
        x.GenerateTensorValue(GeneratorTensor_4<XDataType>{x_mean, x_stddev}, num_thread);
    };

    if(do_verification)
    {
        // x、dy、scale 必须走同一组确定性初始化路径。旧 NCHW example 中 x 固定走
        // GeneratorTensor_4，而 dy/scale 走 init_method，会导致 CPU reference
        // 对比的不是同一个问题。
        switch(init_method)
        {
        case 0:
            x.GenerateTensorValue(GeneratorTensor_0<XDataType>{}, num_thread);
            dy.GenerateTensorValue(GeneratorTensor_0<XDataType>{}, num_thread);
            bnScale.GenerateTensorValue(GeneratorTensor_0<ScaleDataType>{}, num_thread);
            break;
        case 1:
            x.GenerateTensorValue(GeneratorTensor_1<XDataType>{1}, num_thread);
            dy.GenerateTensorValue(GeneratorTensor_1<XDataType>{1}, num_thread);
            bnScale.GenerateTensorValue(GeneratorTensor_1<ScaleDataType>{1}, num_thread);
            break;
        case 2:
            x.GenerateTensorValue(GeneratorTensor_2<XDataType>{-2, 2}, num_thread);
            dy.GenerateTensorValue(GeneratorTensor_2<XDataType>{-2, 2}, num_thread);
            bnScale.GenerateTensorValue(GeneratorTensor_2<ScaleDataType>{-5, 5}, num_thread);
            break;
        default:
            x.GenerateTensorValue(GeneratorTensor_3<XDataType>{-0.2f, 0.2f}, num_thread);
            dy.GenerateTensorValue(GeneratorTensor_3<XDataType>{-0.2f, 0.2f}, num_thread);
            bnScale.GenerateTensorValue(GeneratorTensor_3<ScaleDataType>{-0.5f, 0.5f}, num_thread);
        }
    };

    // input data of the batchnorm backward algorithm
    DeviceMem x_dev(sizeof(XDataType) * x.mDesc.GetElementSpaceSize());
    DeviceMem dy_dev(sizeof(XDataType) * dy.mDesc.GetElementSpaceSize());

    DeviceMem bnScale_dev(sizeof(ScaleDataType) * bnScale.mDesc.GetElementSpaceSize());

    DeviceMem savedMean_dev(sizeof(AccDataType) * savedMean.mDesc.GetElementSpaceSize());
    DeviceMem savedInvVar_dev(sizeof(AccDataType) * savedInvVar.mDesc.GetElementSpaceSize());

    // output data of the batchnorm backward algorithm
    DeviceMem dx_dev(sizeof(XDataType) * dx.mDesc.GetElementSpaceSize());

    DeviceMem dscale_dev(sizeof(AccDataType) * dscale.mDesc.GetElementSpaceSize());
    DeviceMem dbias_dev(sizeof(AccDataType) * dbias.mDesc.GetElementSpaceSize());

    x_dev.ToDevice(x.mData.data());
    dy_dev.ToDevice(dy.mData.data());
    bnScale_dev.ToDevice(bnScale.mData.data());

    if(haveSavedMeanInvVar)
    {
        savedMean_dev.ToDevice(savedMean.mData.data());
        savedInvVar_dev.ToDevice(savedInvVar.mData.data());
    };

    std::array<index_t, Rank> i_inOutLengths;
    std::array<index_t, Rank> i_inOutStrides;
    std::array<index_t, Rank - NumReduceDim> i_scaleBiasMeanVarLengths;
    std::array<index_t, Rank - NumReduceDim> i_scaleBiasMeanVarStrides;

    std::copy(inOutLengths.begin(), inOutLengths.end(), i_inOutLengths.begin());
    std::copy(inOutStrides.begin(), inOutStrides.end(), i_inOutStrides.begin());
    std::copy(scaleBiasMeanVarLengths.begin(),
              scaleBiasMeanVarLengths.end(),
              i_scaleBiasMeanVarLengths.begin());
    std::copy(scaleBiasMeanVarStrides.begin(),
              scaleBiasMeanVarStrides.end(),
              i_scaleBiasMeanVarStrides.begin());

    using PassThroughOp = ck::tensor_operation::element_wise::PassThrough;

    using DeviceBatchNormBwdInstance =
        ck::tensor_operation::device::DeviceBatchNormBwdV2Impl<XDataType,
                                                               XDataType,
                                                               XDataType,
                                                               AccDataType,
                                                               ScaleDataType,
                                                               AccDataType,
                                                               AccDataType,
                                                               PassThroughOp,
                                                               Rank,
                                                               NumReduceDim,
                                                               true,
                                                               256,
                                                               1,
                                                               256,
                                                               2,
                                                               8,
                                                               2,
                                                               4, // vectorDim
                                                               8,
                                                               8,
                                                               8,
                                                               2,
                                                               2,
                                                               2,
                                                               32>;

    auto batchnorm_bwd = DeviceBatchNormBwdInstance{};

    auto argument_ptr = batchnorm_bwd.MakeArgumentPointer(
        i_inOutLengths,
        i_inOutStrides,
        i_inOutStrides,
        i_inOutStrides,
        {0, 2, 3},
        i_scaleBiasMeanVarLengths,
        i_scaleBiasMeanVarStrides,
        i_scaleBiasMeanVarStrides,
        i_scaleBiasMeanVarStrides,
        x_dev.GetDeviceBuffer(),
        dy_dev.GetDeviceBuffer(),
        bnScale_dev.GetDeviceBuffer(),
        haveSavedMeanInvVar ? savedMean_dev.GetDeviceBuffer() : nullptr,
        haveSavedMeanInvVar ? savedInvVar_dev.GetDeviceBuffer() : nullptr,
        epsilon,
        PassThroughOp{},
        dx_dev.GetDeviceBuffer(),
        dscale_dev.GetDeviceBuffer(),
        dbias_dev.GetDeviceBuffer());

    if(!batchnorm_bwd.IsSupportedArgument(argument_ptr.get()))
    {
        std::cout << "The runtime parameters seems not supported by the BatchNorm device instance, "
                     "exiting!"
                  << std::endl;
        return (false);
    };

    size_t workspace_sz = batchnorm_bwd.GetWorkSpaceSize(argument_ptr.get());

    DeviceMem workspace_dev(workspace_sz);

    batchnorm_bwd.SetWorkSpacePointer(argument_ptr.get(), workspace_dev.GetDeviceBuffer());

    auto invoker_ptr = batchnorm_bwd.MakeInvokerPointer();

    if(time_kernel)
    {
        float avg_time   = 0.0f;
        size_t num_bytes = 0;

        size_t total_length = inOutLengths[0] * inOutLengths[1] * inOutLengths[2] * inOutLengths[3];
        size_t invariant_length = inOutLengths[3];

        avg_time = invoker_ptr->Run(argument_ptr.get(), StreamConfig{nullptr, time_kernel});

        // inputing of x, dy, scale, outputing of dx, dscale, dbias
        num_bytes +=
            total_length * sizeof(XDataType) * 3 + invariant_length * sizeof(AccDataType) * 3;

        // outputing of mean, inv-variance
        num_bytes += haveSavedMeanInvVar ? invariant_length * sizeof(AccDataType) * 2 : 0;

        float gb_per_sec = num_bytes / 1.E6 / avg_time;

        std::cout << "Perf: " << avg_time << " ms, " << gb_per_sec << " GB/s" << std::endl;

        if(do_verification)
        {
            // 计时模式内部可能多次运行 kernel；verification 前再非计时运行一次，
            // 保证读取的是干净的最终输出。
            (void)invoker_ptr->Run(argument_ptr.get(), StreamConfig{nullptr, false});
        }
    }
    else
        (void)invoker_ptr->Run(argument_ptr.get(), StreamConfig{nullptr, time_kernel});

    bool pass = true;

    if(do_verification)
    {
        // generic ReferenceBatchNormBwd 路径偏 NHWC。此 example 传入 NCHW strides，
        // 并在 {N,H,W} 上规约，因此这里构造本地 NCHW reference，保证 dBias、
        // dScale、dx 都按同一 layout 对比。
        const auto N = static_cast<index_t>(inOutLengths[0]);
        const auto C = static_cast<index_t>(inOutLengths[1]);
        const auto H = static_cast<index_t>(inOutLengths[2]);
        const auto W = static_cast<index_t>(inOutLengths[3]);

        const auto n_stride = static_cast<index_t>(inOutStrides[0]);
        const auto c_stride = static_cast<index_t>(inOutStrides[1]);
        const auto h_stride = static_cast<index_t>(inOutStrides[2]);
        const auto w_stride = static_cast<index_t>(inOutStrides[3]);

        const auto reduce_size = static_cast<AccDataType>(N * H * W);

        for(index_t c = 0; c < C; ++c)
        {
            AccDataType mean = type_convert<AccDataType>(0.0f);
            AccDataType inv_var;

            if(haveSavedMeanInvVar)
            {
                mean    = type_convert<AccDataType>(savedMean.mData[c]);
                inv_var = type_convert<AccDataType>(savedInvVar.mData[c]);
            }
            else
            {
                AccDataType variance = type_convert<AccDataType>(0.0f);
                int32_t count        = 0;

                for(index_t n = 0; n < N; ++n)
                    for(index_t h = 0; h < H; ++h)
                        for(index_t w = 0; w < W; ++w)
                        {
                            const auto offset =
                                n * n_stride + c * c_stride + h * h_stride + w * w_stride;
                            const AccDataType x_val =
                                static_cast<AccDataType>(host_to_float(x.mData[offset]));

                            ++count;
                            const AccDataType delta = x_val - mean;
                            mean += delta / type_convert<AccDataType>(count);
                            const AccDataType delta2 = x_val - mean;
                            variance += delta * delta2;
                        }

                variance /= reduce_size;
                inv_var =
                    type_convert<AccDataType>(1.0f) /
                    ck::math::sqrt(type_convert<AccDataType>(epsilon) + variance);
            }

            AccDataType dscale_val = type_convert<AccDataType>(0.0f);
            AccDataType dbias_val  = type_convert<AccDataType>(0.0f);

            for(index_t n = 0; n < N; ++n)
                for(index_t h = 0; h < H; ++h)
                    for(index_t w = 0; w < W; ++w)
                    {
                        const auto offset =
                            n * n_stride + c * c_stride + h * h_stride + w * w_stride;
                        const AccDataType x_val =
                            static_cast<AccDataType>(host_to_float(x.mData[offset]));
                        AccDataType dy_val =
                            static_cast<AccDataType>(host_to_float(dy.mData[offset]));
                        const AccDataType norm_x = (x_val - mean) * inv_var;

                        PassThroughOp{}(dy_val, dy_val);
                        dbias_val += dy_val;
                        dscale_val += norm_x * dy_val;
                    }

            dscale_ref.mData[c] = type_convert<AccDataType>(dscale_val);
            dbias_ref.mData[c]  = type_convert<AccDataType>(dbias_val);

            const AccDataType scale = static_cast<AccDataType>(host_to_float(bnScale.mData[c]));
            const AccDataType multiplier =
                type_convert<AccDataType>(1.0f) / reduce_size * inv_var * scale;

            for(index_t n = 0; n < N; ++n)
                for(index_t h = 0; h < H; ++h)
                    for(index_t w = 0; w < W; ++w)
                    {
                        const auto offset =
                            n * n_stride + c * c_stride + h * h_stride + w * w_stride;
                        const AccDataType x_val =
                            static_cast<AccDataType>(host_to_float(x.mData[offset]));
                        AccDataType dy_val =
                            static_cast<AccDataType>(host_to_float(dy.mData[offset]));
                        const AccDataType norm_x = (x_val - mean) * inv_var;

                        PassThroughOp{}(dy_val, dy_val);
                        const AccDataType dx_val =
                            multiplier *
                            (reduce_size * dy_val - dbias_val - norm_x * dscale_val);

                        dx_ref.mData[offset] = host_from_float<XDataType>(dx_val);
                    }
        }

        dx_dev.FromDevice(dx.mData.data());
        dscale_dev.FromDevice(dscale.data());
        dbias_dev.FromDevice(dbias.data());

        // dx 是 half buffer，先按同一 fp16 bits 解释方式转成 float vector，
        // 再用适合 half 的容差比较；dscale/dbias 本身是 float buffer。
        std::vector<AccDataType> dx_check(dx.mData.size());
        std::vector<AccDataType> dx_ref_check(dx_ref.mData.size());
        for(std::size_t i = 0; i < dx.mData.size(); ++i)
        {
            dx_check[i]     = static_cast<AccDataType>(host_to_float(dx.mData[i]));
            dx_ref_check[i] = static_cast<AccDataType>(host_to_float(dx_ref.mData[i]));
        }

        // clang-format off
        pass = pass && ck::utils::check_err(dbias.mData, dbias_ref.mData, "dBias result:", 1e-3, 1e-3);
        pass = pass && ck::utils::check_err(dscale.mData, dscale_ref.mData, "dScale result:", 1e-3, 1e-3);
        pass = pass && ck::utils::check_err(dx_check, dx_ref_check, "dx result:", 1e-2, 1e-2);
        // clang-format on
    };

    return (pass);
};

static const double epsilon = std::numeric_limits<float>::epsilon();

int main(int argc, char* argv[])
{
    bool pass = true;

    if(argc == 1)
    {
        BatchNormBwdArg{}.show_usage(argv[0]);
        return 0;
    }

    if(argc > 1)
    {
        BatchNormBwdArg arg;

        if(arg.processArgs(argc, argv) < 0)
            return (-1);

        if(arg.data_type == 0)
        {
            if(arg.use_multiblock_welford)
                pass = bnorm_bwd_nhwc_test<ck::half_t, float, true>(arg.do_verification,
                                                                    arg.init_method,
                                                                    arg.time_kernel,
                                                                    arg.inOutLengths,
                                                                    arg.haveSavedMeanInvVar,
                                                                    epsilon);
            else
                pass = bnorm_bwd_nhwc_test<ck::half_t, float, false>(arg.do_verification,
                                                                     arg.init_method,
                                                                     arg.time_kernel,
                                                                     arg.inOutLengths,
                                                                     arg.haveSavedMeanInvVar,
                                                                     epsilon);
        }
        // else if(arg.data_type == 1)
        // {
        //     if(arg.use_multiblock_welford)
        //         pass = bnorm_bwd_nhwc_test<float, float, true>(arg.do_verification,
        //                                                        arg.init_method,
        //                                                        arg.time_kernel,
        //                                                        arg.inOutLengths,
        //                                                        arg.haveSavedMeanInvVar,
        //                                                        epsilon);
        //     else
        //         pass = bnorm_bwd_nhwc_test<float, float, false>(arg.do_verification,
        //                                                         arg.init_method,
        //                                                         arg.time_kernel,
        //                                                         arg.inOutLengths,
        //                                                         arg.haveSavedMeanInvVar,
        //                                                         epsilon);
        // }
        // else if(arg.data_type == 5)
        // {
        //     if(arg.use_multiblock_welford)
        //         pass = bnorm_bwd_nhwc_test<ck::bhalf_t, float, true>(arg.do_verification,
        //                                                              arg.init_method,
        //                                                              arg.time_kernel,
        //                                                              arg.inOutLengths,
        //                                                              arg.haveSavedMeanInvVar,
        //                                                              epsilon);
        //     else
        //         pass = bnorm_bwd_nhwc_test<ck::bhalf_t, float, false>(arg.do_verification,
        //                                                               arg.init_method,
        //                                                               arg.time_kernel,
        //                                                               arg.inOutLengths,
        //                                                               arg.haveSavedMeanInvVar,
        //                                                               epsilon);
        // }
        // else if(arg.data_type == 6)
        // {
        //     if(arg.use_multiblock_welford)
        //         pass = bnorm_bwd_nhwc_test<double, double, true>(arg.do_verification,
        //                                                          arg.init_method,
        //                                                          arg.time_kernel,
        //                                                          arg.inOutLengths,
        //                                                          arg.haveSavedMeanInvVar,
        //                                                          epsilon);
        //     else
        //         pass = bnorm_bwd_nhwc_test<double, double, false>(arg.do_verification,
        //                                                           arg.init_method,
        //                                                           arg.time_kernel,
        //                                                           arg.inOutLengths,
        //                                                           arg.haveSavedMeanInvVar,
        //                                                           epsilon);
        // }
    }
    else
    {
        pass = bnorm_bwd_nhwc_test<ck::half_t, float, true>(true,
                                                            3,
                                                            false, // don't time kernel
                                                            {128, 16, 6, 512},
                                                            false,
                                                            epsilon);

        pass = pass && bnorm_bwd_nhwc_test<ck::half_t, float, false>(true,
                                                                     3,
                                                                     false, // don't time kernel
                                                                     {128, 16, 3, 1024},
                                                                     false,
                                                                     epsilon);
    };

    return (pass ? 0 : 1);
}
