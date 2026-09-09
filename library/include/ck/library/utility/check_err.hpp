// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.
// Copyright (c) 2026 Hygon Information Technology Co., Ltd.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <iterator>
#include <limits>
#include <type_traits>
#include <vector>

#include "ck/ck.hpp"
#include "ck/utility/data_type.hpp"
#include "ck/utility/numeric_utils.hpp"
#include "ck/utility/type.hpp"
#include "ck/host_utility/io.hpp"

#include "ck/library/utility/ranges.hpp"

namespace ck {
namespace utils {

template <typename ComputeDataType, typename OutDataType, typename AccDataType = ComputeDataType>
double get_relative_threshold(const int number_of_accumulations = 1)
{
    using F16  = ck::half_t;
    using BF16 = ck::bhalf_t;
    using F32  = float;
    using TF32 = ck::tf32_t;
    using I8   = int8_t;
    using I32  = int32_t;

    static_assert(std::is_same_v<ComputeDataType, F16> ||
                      std::is_same_v<ComputeDataType, BF16> ||
                      std::is_same_v<ComputeDataType, F32> ||
                      std::is_same_v<ComputeDataType, TF32> ||
                      std::is_same_v<ComputeDataType, I8> ||
                      std::is_same_v<ComputeDataType, I32> ||
                      std::is_same_v<ComputeDataType, int>,
                  "Warning: Unhandled ComputeDataType for setting up the relative threshold!");

    double compute_error = 0;
    if constexpr(std::is_same_v<ComputeDataType, I8> || std::is_same_v<ComputeDataType, I32> ||
                 std::is_same_v<ComputeDataType, int>)
    {
        return 0;
    }
    else
    {
        compute_error = std::pow(2, -NumericUtils<ComputeDataType>::mant) * 0.5;
    }

    static_assert(std::is_same_v<OutDataType, F16> || std::is_same_v<OutDataType, BF16> ||
                      std::is_same_v<OutDataType, F32> ||
                      std::is_same_v<OutDataType, TF32> || std::is_same_v<OutDataType, I8> ||
                      std::is_same_v<OutDataType, I32> || std::is_same_v<OutDataType, int>,
                  "Warning: Unhandled OutDataType for setting up the relative threshold!");

    double output_error = 0;
    if constexpr(std::is_same_v<OutDataType, I8> || std::is_same_v<OutDataType, I32> ||
                 std::is_same_v<OutDataType, int>)
    {
        return 0;
    }
    else
    {
        output_error = std::pow(2, -NumericUtils<OutDataType>::mant) * 0.5;
    }

    const double midway_error = std::max(compute_error, output_error);

    static_assert(std::is_same_v<AccDataType, F16> || std::is_same_v<AccDataType, BF16> ||
                      std::is_same_v<AccDataType, F32> ||
                      std::is_same_v<AccDataType, TF32> || std::is_same_v<AccDataType, I8> ||
                      std::is_same_v<AccDataType, I32> || std::is_same_v<AccDataType, int>,
                  "Warning: Unhandled AccDataType for setting up the relative threshold!");

    double acc_error = 0;
    if constexpr(std::is_same_v<AccDataType, I8> || std::is_same_v<AccDataType, I32> ||
                 std::is_same_v<AccDataType, int>)
    {
        return 0;
    }
    else
    {
        acc_error = std::pow(2, -NumericUtils<AccDataType>::mant) * 0.5 * number_of_accumulations;
    }

    return std::max(acc_error, midway_error);
}

template <typename ComputeDataType, typename OutDataType, typename AccDataType = ComputeDataType>
double get_absolute_threshold(const double max_possible_num, const int number_of_accumulations = 1)
{
    using F16  = ck::half_t;
    using BF16 = ck::bhalf_t;
    using F32  = float;
    using TF32 = ck::tf32_t;
    using I8   = int8_t;
    using I32  = int32_t;

    static_assert(std::is_same_v<ComputeDataType, F16> ||
                      std::is_same_v<ComputeDataType, BF16> ||
                      std::is_same_v<ComputeDataType, F32> ||
                      std::is_same_v<ComputeDataType, TF32> ||
                      std::is_same_v<ComputeDataType, I8> ||
                      std::is_same_v<ComputeDataType, I32> ||
                      std::is_same_v<ComputeDataType, int>,
                  "Warning: Unhandled ComputeDataType for setting up the absolute threshold!");

    const auto expo     = std::log2(std::abs(max_possible_num));
    double compute_error = 0;
    if constexpr(std::is_same_v<ComputeDataType, I8> || std::is_same_v<ComputeDataType, I32> ||
                 std::is_same_v<ComputeDataType, int>)
    {
        return 0;
    }
    else
    {
        compute_error = std::pow(2, expo - NumericUtils<ComputeDataType>::mant) * 0.5;
    }

    static_assert(std::is_same_v<OutDataType, F16> || std::is_same_v<OutDataType, BF16> ||
                      std::is_same_v<OutDataType, F32> ||
                      std::is_same_v<OutDataType, TF32> || std::is_same_v<OutDataType, I8> ||
                      std::is_same_v<OutDataType, I32> || std::is_same_v<OutDataType, int>,
                  "Warning: Unhandled OutDataType for setting up the absolute threshold!");

    double output_error = 0;
    if constexpr(std::is_same_v<OutDataType, I8> || std::is_same_v<OutDataType, I32> ||
                 std::is_same_v<OutDataType, int>)
    {
        return 0;
    }
    else
    {
        output_error = std::pow(2, expo - NumericUtils<OutDataType>::mant) * 0.5;
    }

    const double midway_error = std::max(compute_error, output_error);

    static_assert(std::is_same_v<AccDataType, F16> || std::is_same_v<AccDataType, BF16> ||
                      std::is_same_v<AccDataType, F32> ||
                      std::is_same_v<AccDataType, TF32> || std::is_same_v<AccDataType, I8> ||
                      std::is_same_v<AccDataType, I32> || std::is_same_v<AccDataType, int>,
                  "Warning: Unhandled AccDataType for setting up the absolute threshold!");

    double acc_error = 0;
    if constexpr(std::is_same_v<AccDataType, I8> || std::is_same_v<AccDataType, I32> ||
                 std::is_same_v<AccDataType, int>)
    {
        return 0;
    }
    else
    {
        acc_error =
            std::pow(2, expo - NumericUtils<AccDataType>::mant) * 0.5 * number_of_accumulations;
    }

    return std::max(acc_error, midway_error);
}

template <typename Range,
          typename RefRange,
          typename ComputeDataType = ranges::range_value_t<Range>>
typename std::enable_if<
    std::is_same_v<ranges::range_value_t<Range>, ranges::range_value_t<RefRange>> &&
        std::is_same_v<ranges::range_value_t<Range>, float> &&
        std::is_same_v<ComputeDataType, ck::tf32_t>,
    bool>::type
check_err(const Range& out,
          const RefRange& ref,
          const std::string& msg = "Error: Incorrect results!",
          double rtol            = 5e-4,
          double atol            = 5e-4)
{
    if(out.size() != ref.size())
    {
        std::cerr << msg << " out.size() != ref.size(), :" << out.size() << " != " << ref.size()
                  << std::endl;
        return false;
    }

    bool res{true};
    int err_count  = 0;
    double err     = 0;
    double max_err = std::numeric_limits<double>::min();
    for(std::size_t i = 0; i < ref.size(); ++i)
    {
        const double o = *std::next(std::begin(out), i);
        const double r = *std::next(std::begin(ref), i);
        err            = std::abs(o - r);
        if(err > atol + rtol * std::abs(r) || !std::isfinite(o) || !std::isfinite(r))
        {
            max_err = err > max_err ? err : max_err;
            if(err_count < 5)
            {
                std::cerr << msg << std::setw(12) << std::setprecision(7) << " out[" << i
                          << "] != ref[" << i << "]: " << o << " != " << r << std::endl;
            }
            res = false;
            err_count++;
        }
    }
    if(!res)
    {
        const float error_percent =
            static_cast<float>(err_count) / static_cast<float>(out.size()) * 100.f;
        std::cerr << "max err: " << max_err;
        std::cerr << ", number of errors: " << err_count;
        std::cerr << ", " << error_percent << "% wrong values" << std::endl;
    }
    return res;
}

template <typename Range,
          typename RefRange,
          typename ComputeDataType = ranges::range_value_t<Range>>
typename std::enable_if<
    std::is_same_v<ranges::range_value_t<Range>, ranges::range_value_t<RefRange>> &&
        std::is_floating_point_v<ranges::range_value_t<Range>> &&
        !std::is_same_v<ranges::range_value_t<Range>, half_t> &&
        !std::is_same_v<ComputeDataType, ck::tf32_t>,
    bool>::type
check_err(const Range& out,
          const RefRange& ref,
          const std::string& msg = "Error: Incorrect results!",
          double rtol            = 1e-5,
          double atol            = 6e-5)
{
    if(out.size() != ref.size())
    {
        std::cerr << msg << " out.size() != ref.size(), :" << out.size() << " != " << ref.size()
                  << std::endl;
        return false;
    }

    bool res{true};
    int err_count  = 0;
    double err     = 0;
    double max_err = std::numeric_limits<double>::min();
    for(std::size_t i = 0; i < ref.size(); ++i)
    {
        const double o = *std::next(std::begin(out), i);
        const double r = *std::next(std::begin(ref), i);
        err            = std::abs(o - r);
        if(err > atol + rtol * std::abs(r) || !std::isfinite(o) || !std::isfinite(r))
        {
            max_err = err > max_err ? err : max_err;
            err_count++;
            if(err_count < 5)
            {
                std::cerr << msg << std::setw(12) << std::setprecision(7) << " out[" << i
                          << "] != ref[" << i << "]: " << o << " != " << r << std::endl;
            }
            res = false;
        }
    }
    if(!res)
    {
        std::cerr << std::setw(12) << std::setprecision(7) << "max err: " << max_err << std::endl;
    }
    return res;
}

template <typename Range, typename RefRange>
typename std::enable_if<
    std::is_same_v<ranges::range_value_t<Range>, ranges::range_value_t<RefRange>> &&
        std::is_same_v<ranges::range_value_t<Range>, bhalf_t>,
    bool>::type
check_err(const Range& out,
          const RefRange& ref,
          const std::string& msg = "Error: Incorrect results!",
          double rtol            = 1e-3,
          double atol            = 2e-1)
{
    if(out.size() != ref.size())
    {
        std::cerr << msg << " out.size() != ref.size(), :" << out.size() << " != " << ref.size()
                  << std::endl;
        return false;
    }

    bool res{true};
    int err_count = 0;
    double err    = 0;
    // TODO: This is a hack. We should have proper specialization for bhalf_t data type.
    double max_err = std::numeric_limits<float>::min();
    for(std::size_t i = 0; i < ref.size(); ++i)
    {
        const double o = type_convert<float>(*std::next(std::begin(out), i));
        const double r = type_convert<float>(*std::next(std::begin(ref), i));
        err            = std::abs(o - r);
        if(err > atol + rtol * std::abs(r) || !std::isfinite(o) || !std::isfinite(r))
        {
            max_err = err > max_err ? err : max_err;
            err_count++;
            if(err_count < 5)
            {
                std::cerr << msg << std::setw(12) << std::setprecision(7) << " out[" << i
                          << "] != ref[" << i << "]: " << o << " != " << r << std::endl;
            }
            res = false;
        }
    }
    if(!res)
    {
        std::cerr << std::setw(12) << std::setprecision(7) << "max err: " << max_err << std::endl;
    }
    return res;
}

template <typename Range, typename RefRange>
typename std::enable_if<
    std::is_same_v<ranges::range_value_t<Range>, ranges::range_value_t<RefRange>> &&
        std::is_same_v<ranges::range_value_t<Range>, half_t>,
    bool>::type
check_err(const Range& out,
          const RefRange& ref,
          const std::string& msg = "Error: Incorrect results!",
          double rtol            = 1e-3,
          double atol            = 1e-3)
{
    if(out.size() != ref.size())
    {
        std::cerr << msg << " out.size() != ref.size(), :" << out.size() << " != " << ref.size()
                  << std::endl;
        return false;
    }

    bool res{true};
    int err_count  = 0;
    double err     = 0;
    double max_err = std::numeric_limits<ranges::range_value_t<Range>>::min();
    for(std::size_t i = 0; i < ref.size(); ++i)
    {
        const double o = type_convert<float>(*std::next(std::begin(out), i));
        const double r = type_convert<float>(*std::next(std::begin(ref), i));
        err            = std::abs(o - r);
        if(err > atol + rtol * std::abs(r) || !std::isfinite(o) || !std::isfinite(r))
        {
            max_err = err > max_err ? err : max_err;
            err_count++;
            if(err_count < 16)
            {
                std::cerr << msg << std::setw(12) << std::setprecision(7) << " out[" << i
                          << "] != ref[" << i << "]: " << o << " != " << r << std::endl;
            }
            res = false;
        }
    }
    if(!res)
    {
        std::cerr << std::setw(12) << std::setprecision(7) << "max err: " << max_err << std::endl;
    }
    return res;
}

template <typename Range, typename RefRange>
std::enable_if_t<(std::is_same_v<ranges::range_value_t<Range>, ranges::range_value_t<RefRange>> &&
                  std::is_integral_v<ranges::range_value_t<Range>> &&
                  !std::is_same_v<ranges::range_value_t<Range>, bhalf_t>
#ifdef CK_EXPERIMENTAL_BIT_INT_EXTENSION
                  && !std::is_same_v<ranges::range_value_t<Range>, f8_t> &&
                  !std::is_same_v<ranges::range_value_t<Range>, bf8_t>
#endif
                  )
#ifdef CK_EXPERIMENTAL_BIT_INT_EXTENSION
                     || std::is_same_v<ranges::range_value_t<Range>, int4_t>
#endif
                 ,
                 bool>
check_err(const Range& out,
          const RefRange& ref,
          const std::string& msg = "Error: Incorrect results!",
          double                 = 0,
          double atol            = 0)
{
    if(out.size() != ref.size())
    {
        std::cerr << msg << " out.size() != ref.size(), :" << out.size() << " != " << ref.size()
                  << std::endl;
        return false;
    }

    bool res{true};
    int err_count   = 0;
    int64_t err     = 0;
    int64_t max_err = std::numeric_limits<int64_t>::min();
    for(std::size_t i = 0; i < ref.size(); ++i)
    {
        const int64_t o = *std::next(std::begin(out), i);
        const int64_t r = *std::next(std::begin(ref), i);
        err             = std::abs(o - r);

        if(err > atol)
        {
            max_err = err > max_err ? err : max_err;
            err_count++;
            if(err_count < 5)
            {
                std::cerr << msg << " out[" << i << "] != ref[" << i << "]: " << o << " != " << r
                          << std::endl;
            }
            res = false;
        }
    }
    if(!res)
    {
        std::cerr << "max err: " << max_err << std::endl;
    }
    return res;
}

#ifdef CK_EXPERIMENTAL_BIT_INT_EXTENSION
template <typename Range,
          typename RefRange,
          typename ComputeDataType = ranges::range_value_t<Range>>
std::enable_if_t<(std::is_same_v<ranges::range_value_t<Range>, ranges::range_value_t<RefRange>> &&
                  std::is_same_v<ranges::range_value_t<Range>, f8_t>),
                 bool>
check_err(const Range& out,
          const RefRange& ref,
          const std::string& msg = "Error: Incorrect results!",
          double rtol            = 1e-3,
          double atol            = 1e-3)
{
    if(out.size() != ref.size())
    {
        std::cerr << msg << " out.size() != ref.size(), :" << out.size() << " != " << ref.size()
                  << std::endl;
        return false;
    }

    bool res{true};
    int err_count  = 0;
    double err     = 0;
    double max_err = std::numeric_limits<float>::min();

    for(std::size_t i = 0; i < ref.size(); ++i)
    {
        const double o = type_convert<float>(*std::next(std::begin(out), i));
        const double r = type_convert<float>(*std::next(std::begin(ref), i));
        err            = std::abs(o - r);

        if(err > atol + rtol * std::abs(r) || !std::isfinite(o) || !std::isfinite(r))
        {
            max_err = err > max_err ? err : max_err;
            err_count++;
            if(err_count < 5)
            {
                std::cerr << msg << std::setw(12) << std::setprecision(7) << " out[" << i
                          << "] != ref[" << i << "]: " << o << " != " << r << std::endl;
            }
            res = false;
        }
    }

    if(!res)
    {
        std::cerr << std::setw(12) << std::setprecision(7) << "max err: " << max_err
                  << " number of errors: " << err_count << std::endl;
    }
    return res;
}

template <typename Range,
          typename RefRange,
          typename ComputeDataType = ranges::range_value_t<Range>>
std::enable_if_t<(std::is_same_v<ranges::range_value_t<Range>, ranges::range_value_t<RefRange>> &&
                  std::is_same_v<ranges::range_value_t<Range>, bf8_t>),
                 bool>
check_err(const Range& out,
          const RefRange& ref,
          const std::string& msg = "Error: Incorrect results!",
          double rtol            = 1e-3,
          double atol            = 1e-3)
{
    if(out.size() != ref.size())
    {
        std::cerr << msg << " out.size() != ref.size(), :" << out.size() << " != " << ref.size()
                  << std::endl;
        return false;
    }

    bool res{true};
    int err_count  = 0;
    double err     = 0;
    double max_err = std::numeric_limits<float>::min();
    for(std::size_t i = 0; i < ref.size(); ++i)
    {
        const double o = type_convert<float>(*std::next(std::begin(out), i));
        const double r = type_convert<float>(*std::next(std::begin(ref), i));
        err            = std::abs(o - r);
        if(err > atol + rtol * std::abs(r) || !std::isfinite(o) || !std::isfinite(r))
        {
            max_err = err > max_err ? err : max_err;
            err_count++;
            if(err_count < 5)
            {
                std::cerr << msg << std::setw(12) << std::setprecision(7) << " out[" << i
                          << "] != ref[" << i << "]: " << o << " != " << r << std::endl;
            }
            res = false;
        }
    }
    if(!res)
    {
        std::cerr << std::setw(12) << std::setprecision(7) << "max err: " << max_err << std::endl;
    }
    return res;
}
#endif

} // namespace utils
} // namespace ck
