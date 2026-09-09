// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.
// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// Modified by Hygon Information Technology Co., Ltd.

#pragma once

#include <vector>
#include <type_traits>
#include <memory>

#include "ck/utility/functional2.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

/**
 * @brief Register device operation instances from a tuple of pre-constructed instances.
 *
 * Iterates over a std::tuple of device operation instance objects, creates a
 * std::unique_ptr copy of each, and appends them to the provided vector.
 *
 * nullptr_t elements in the tuple are silently skipped, which enables the
 * trailing-comma style used in upstream registration lists.
 *
 * @tparam BaseOp The base class that all operation instances must derive from.
 * @tparam NewOpInstances A std::tuple of device operation instance objects.
 *         Each non-null element must be derived from BaseOp.
 */
template <typename BaseOp, typename NewOpInstances>
void add_device_operation_instances(std::vector<std::unique_ptr<BaseOp>>& op_instances,
                                    const NewOpInstances& new_op_instances)
{
    ck::static_for<0, std::tuple_size_v<NewOpInstances>, 1>{}([&](auto i) {
        const auto new_op_instance = std::get<i>(new_op_instances);

        using NewOpInstance = remove_cvref_t<decltype(new_op_instance)>;

        if constexpr(std::is_same_v<NewOpInstance, std::nullptr_t>)
        {
            return;
        }
        else
        {
            static_assert(std::is_base_of_v<BaseOp, NewOpInstance>,
                          "wrong! NewOpInstance should be derived from BaseOp");

            op_instances.push_back(std::make_unique<NewOpInstance>(new_op_instance));
        }
    });
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
