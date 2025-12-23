#pragma once

#include <cassert>
#include <cstddef>
#include <span>
#include <vector>
#include <cstdlib>

struct point2
{
    float x, y;
};

struct point3
{
    float x, y, z;
};

// Result type alias for convenience
using dbscan_result = std::pair<std::vector<std::vector<size_t>>, std::vector<int>>;

// 2D points
auto dbscan(const std::span<const point2>& data, float eps, int min_pts) -> dbscan_result;

// 3D points
auto dbscan(const std::span<const point3>& data, float eps, int min_pts) -> dbscan_result;

// Generic template for flat float arrays
template<size_t dim>
auto dbscan(const std::span<const float>& data, float eps, int min_pts) -> dbscan_result
{
    static_assert(dim == 2 || dim == 3, "This only supports either 2D or 3D points");
    assert(data.size() % dim == 0);
    
    if constexpr (dim == 2)
    {
        auto const* ptr = reinterpret_cast<point2 const*>(data.data());
        auto points = std::span<const point2>(ptr, data.size() / dim);
        return dbscan(points, eps, min_pts);
    }
    else // dim == 3
    {
        auto const* ptr = reinterpret_cast<point3 const*>(data.data());
        auto points = std::span<const point3>(ptr, data.size() / dim);
        return dbscan(points, eps, min_pts);
    }
}