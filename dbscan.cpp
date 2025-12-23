#include "dbscan.hpp"

#include <cstddef>
#include <nanoflann/nanoflann.hpp>

#include <type_traits>
#include <vector>

// Adaptor helper functions for accessing point coordinates

inline auto get_pt(const point2& p, std::size_t dim)
{
    if (dim == 0) return p.x;
    return p.y;
}

inline auto get_pt(const point3& p, std::size_t dim)
{
    if (dim == 0) return p.x;
    if (dim == 1) return p.y;
    return p.z;
}

// Dataset to kd-tree adaptor class
template<typename Point>
struct adaptor
{
    const std::span<const Point>& points;
    
    adaptor(const std::span<const Point>& points) : points(points) { }

    inline std::size_t kdtree_get_point_count() const { return points.size(); }

    inline float kdtree_get_pt(const std::size_t idx, const std::size_t dim) const
    {
        return get_pt(points[idx], dim);
    }

    template <class BBOX>
    bool kdtree_get_bbox(BBOX& /*bb*/) const { return false; }

    auto const* elem_ptr(const std::size_t idx) const
    {
        return &points[idx].x;
    }
};

// Core DBSCAN implementation
template<int n_cols, typename Adaptor>
auto dbscan_impl(const Adaptor& adapt, float eps, int min_pts)
{
    eps *= eps;  // Convert to squared distance for L2_Simple_Adaptor
    
    using namespace nanoflann;
    using my_kd_tree_t = KDTreeSingleIndexAdaptor<
        L2_Simple_Adaptor<float, decltype(adapt)>, 
        decltype(adapt), 
        n_cols
    >;

    auto index = my_kd_tree_t(n_cols, adapt, KDTreeSingleIndexAdaptorParams(10));
    index.buildIndex();

    const auto n_points = adapt.kdtree_get_point_count();
    auto visited = std::vector<bool>(n_points);
    auto clusters = std::vector<std::vector<size_t>>();
    auto cluster_indices = std::vector<int>(n_points, -1);  // -1 = noise
    auto matches = std::vector<std::pair<size_t, float>>();
    auto sub_matches = std::vector<std::pair<size_t, float>>();

    int cluster_id = 0;
    
    for (size_t i = 0; i < n_points; i++)
    {
        if (visited[i]) continue;

        index.radiusSearch(adapt.elem_ptr(i), eps, matches, SearchParams(32, 0.f, false));
        
        if (matches.size() < static_cast<size_t>(min_pts)) continue;
        
        visited[i] = true;
        cluster_indices[i] = cluster_id;

        auto cluster = std::vector<size_t>({i});

        while (!matches.empty())
        {
            auto nb_idx = matches.back().first;
            matches.pop_back();
            
            if (visited[nb_idx]) continue;
            visited[nb_idx] = true;

            index.radiusSearch(adapt.elem_ptr(nb_idx), eps, sub_matches, SearchParams(32, 0.f, false));

            if (sub_matches.size() >= static_cast<size_t>(min_pts))
            {
                std::copy(sub_matches.begin(), sub_matches.end(), std::back_inserter(matches));
            }
            
            cluster_indices[nb_idx] = cluster_id;
            cluster.push_back(nb_idx);
        }
        
        clusters.emplace_back(std::move(cluster));
        cluster_id++;
    }
    
    return std::make_pair(clusters, cluster_indices);
}

// Public API for 2D points
auto dbscan(const std::span<const point2>& data, float eps, int min_pts) 
    -> std::pair<std::vector<std::vector<size_t>>, std::vector<int>>
{
    const auto adapt = adaptor<point2>(data);
    return dbscan_impl<2>(adapt, eps, min_pts);
}

// Public API for 3D points
auto dbscan(const std::span<const point3>& data, float eps, int min_pts) 
    -> std::pair<std::vector<std::vector<size_t>>, std::vector<int>>
{
    const auto adapt = adaptor<point3>(data);
    return dbscan_impl<3>(adapt, eps, min_pts);
}