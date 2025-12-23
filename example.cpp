#include "dbscan.hpp"

#include <iostream>
#include <string>
#include <system_error>
#include <vector>
#include <utility>
#include <fstream>
#include <charconv>
#include <cassert>
#include <tuple>
#include <cstring>


auto check_from_chars_error(std::errc err, const std::string_view& line, int line_counter)
{
    if (err == std::errc())
        return;
    
    if (err == std::errc::invalid_argument)
    {
        std::cerr << "Error: Invalid value \"" << line
                  << "\" at line " << line_counter << "\n";
        std::exit(1);
    }

    if (err == std::errc::result_out_of_range)
    {
        std::cerr << "Error: Value \"" << line << "\" out of range at line " 
                  << line_counter << "\n";
        std::exit(1);
    }
}


auto push_values(std::vector<float>& store, const std::string_view& line, int line_counter)
{
    auto ptr = line.data();
    auto n_pushed = 0;

    do
    {
        float value;
        auto [p, ec] = std::from_chars(ptr, line.data() + line.size(), value);
        check_from_chars_error(ec, line, line_counter);
        
        store.push_back(value);
        n_pushed++;
        ptr = p + 1;  // Skip delimiter

    } while (ptr < line.data() + line.size());

    return n_pushed;
}


auto read_values(const std::string& filename)
{
    std::ifstream file(filename);

    if (!file.good())
    {
        std::perror(filename.c_str());
        std::exit(2);
    }

    auto count = 0;
    auto points = std::vector<float>();
    auto dim = 0;

    std::string line;
    while (std::getline(file, line))
    {
        count++;

        if (!line.empty())
        {
            auto n_pushed = push_values(points, line, count);

            if (dim != 0 && n_pushed != dim)
            {
                std::cerr << "Inconsistent number of dimensions at line " << count << "\n";
                std::exit(1);
            }
            dim = n_pushed;
        }
    }

    return std::tuple(points, dim);
}


template<typename T>
auto to_num(const std::string& str)
{
    T value = 0;
    auto [ptr, ec] = std::from_chars(str.data(), str.data() + str.size(), value);

    if (ec != std::errc())
    {
        std::cerr << "Error converting value '" << str << "'\n";
        std::exit(1);
    }
    return value;
}


// Noise points get label 0, clusters get labels 1, 2, 3, ...
auto flatten_clusters(const std::vector<std::vector<size_t>>& clusters, size_t n_points)
{
    auto labels = std::vector<size_t>(n_points, 0);  // 0 = noise

    for (size_t cluster_id = 0; cluster_id < clusters.size(); cluster_id++)
    {
        for (auto point_idx : clusters[cluster_id])
        {
            labels[point_idx] = cluster_id + 1;
        }
    }

    return labels;
}


void run_dbscan_2d(const std::span<const float>& data, float eps, int min_pts)
{
    const size_t n_points = data.size() / 2;
    
    auto points = std::vector<point2>(n_points);
    std::memcpy(points.data(), data.data(), sizeof(float) * data.size());

    auto [clusters, cluster_indices] = dbscan(std::span<const point2>(points), eps, min_pts);
    auto labels = flatten_clusters(clusters, n_points);

    for (size_t i = 0; i < n_points; i++)
    {
        std::cout << points[i].x << ',' << points[i].y << ',' << labels[i] << '\n';
    }
}


void run_dbscan_3d(const std::span<const float>& data, float eps, int min_pts)
{
    const size_t n_points = data.size() / 3;
    
    auto points = std::vector<point3>(n_points);
    std::memcpy(points.data(), data.data(), sizeof(float) * data.size());

    auto [clusters, cluster_indices] = dbscan(std::span<const point3>(points), eps, min_pts);
    auto labels = flatten_clusters(clusters, n_points);

    for (size_t i = 0; i < n_points; i++)
    {
        std::cout << points[i].x << ',' << points[i].y << ',' << points[i].z << ',' << labels[i] << '\n';
    }
}


int main(int argc, char** argv)
{
    if (argc != 4)
    {
        std::cerr << "Usage: " << argv[0] << " <csv file> <epsilon> <min_points>\n";
        std::cerr << "\n";
        std::cerr << "Input file format: CSV with 2 or 3 columns (x,y or x,y,z)\n";
        std::cerr << "Output: input coordinates + cluster label (0 = noise)\n";
        return 1;
    }

    auto epsilon = to_num<float>(argv[2]);
    auto min_pts = to_num<int>(argv[3]);
    auto [values, dim] = read_values(argv[1]);

    if (values.empty())
    {
        std::cerr << "Error: No points found in file\n";
        return 1;
    }

    if (dim == 2)
    {
        run_dbscan_2d(values, epsilon, min_pts);
    }
    else if (dim == 3)
    {
        run_dbscan_3d(values, epsilon, min_pts);
    }
    else
    {
        std::cerr << "Error: Only 2D and 3D points are supported (found " << dim << " dimensions)\n";
        return 1;
    }

    return 0;
}