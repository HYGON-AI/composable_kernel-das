// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#pragma once

#include <stdexcept>
#include <string>
#include <vector>

namespace ck_tile {

// Only an omitted list is empty. Malformed or non-positive entries must never
// turn an explicitly requested shape into the default nq/nkv configuration.
inline std::vector<int> parse_query_lens_helper(const std::string& text,
                                             const char* argument = "query_lens")
{
    std::vector<int> lengths;
    if(text.empty()) return lengths;
    const std::string error = std::string(argument) +
                             " must be a comma-separated list of positive integers";
    std::size_t begin = 0;
    while(true)
    {
        const auto end = text.find(',', begin);
        const auto token = text.substr(begin, end == std::string::npos ? end : end - begin);
        std::size_t consumed = 0;
        int value = 0;
        try
        {
            value = std::stoi(token, &consumed);
        }
        catch(const std::invalid_argument&)
        {
            throw std::invalid_argument(error);
        }
        catch(const std::out_of_range&)
        {
            throw std::invalid_argument(error);
        }
        if(value <= 0 || token.find_first_not_of(" \t\r\n", consumed) != std::string::npos)
            throw std::invalid_argument(error);
        lengths.push_back(value);
        if(end == std::string::npos) break;
        begin = end + 1;
    }
    return lengths;
}

} // namespace ck_tile
