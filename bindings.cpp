#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "dbscan.hpp"

namespace py = pybind11;

std::vector<point2> array2d_to_points(const py::array_t<float>& py_points) {
    auto buf = py_points.request();
    
    if (buf.ndim != 2 || buf.shape[1] != 2) {
        throw std::runtime_error("Input array must have shape (N, 2)");
    }
    
    auto* ptr = static_cast<float*>(buf.ptr);
    size_t num_points = buf.shape[0];
    
    std::vector<point2> points;
    points.reserve(num_points);
    
    for (size_t i = 0; i < num_points; ++i) {
        points.push_back(point2{ptr[2 * i], ptr[2 * i + 1]});
    }
    
    return points;
}

std::vector<point3> array3d_to_points(const py::array_t<float>& py_points) {
    auto buf = py_points.request();
    
    if (buf.ndim != 2 || buf.shape[1] != 3) {
        throw std::runtime_error("Input array must have shape (N, 3)");
    }
    
    auto* ptr = static_cast<float*>(buf.ptr);
    size_t num_points = buf.shape[0];
    
    std::vector<point3> points;
    points.reserve(num_points);
    
    for (size_t i = 0; i < num_points; ++i) {
        points.push_back(point3{ptr[3 * i], ptr[3 * i + 1], ptr[3 * i + 2]});
    }
    
    return points;
}

// Helper to convert result to Python types
py::tuple result_to_python(const dbscan_result& result) {
    const auto& clusters = result.first;
    const auto& cluster_indices = result.second;
    
    py::list py_clusters;
    for (const auto& cluster : clusters) {
        py::array_t<size_t> py_cluster(cluster.size(), cluster.data());
        py_clusters.append(py_cluster);
    }
    
    py::array_t<int> py_cluster_indices(cluster_indices.size(), cluster_indices.data());
    
    return py::make_tuple(py_clusters, py_cluster_indices);
}

PYBIND11_MODULE(fast_dbscan, m) {
    m.doc() = "Fast DBSCAN clustering using KD-tree";

    // Point classes
    py::class_<point2>(m, "Point2")
        .def(py::init<>())
        .def(py::init<float, float>(), py::arg("x"), py::arg("y"))
        .def_readwrite("x", &point2::x)
        .def_readwrite("y", &point2::y)
        .def("__repr__", [](const point2& p) {
            return "Point2(" + std::to_string(p.x) + ", " + std::to_string(p.y) + ")";
        });

    py::class_<point3>(m, "Point3")
        .def(py::init<>())
        .def(py::init<float, float, float>(), py::arg("x"), py::arg("y"), py::arg("z"))
        .def_readwrite("x", &point3::x)
        .def_readwrite("y", &point3::y)
        .def_readwrite("z", &point3::z)
        .def("__repr__", [](const point3& p) {
            return "Point3(" + std::to_string(p.x) + ", " + std::to_string(p.y) + ", " + std::to_string(p.z) + ")";
        });

    // DBSCAN for 2D points
    m.def("dbscan_2d", [](py::array_t<float, py::array::c_style | py::array::forcecast> py_points, 
                          float eps, int min_pts) {
        std::vector<point2> points = array2d_to_points(py_points);
        auto result = dbscan(std::span<const point2>(points), eps, min_pts);
        return result_to_python(result);
    }, 
    R"pbdoc(
        DBSCAN clustering for 2D points.
        
        Args:
            data: numpy array of shape (N, 2) with float32 dtype
            eps: maximum distance between points in the same neighborhood
            min_pts: minimum number of points to form a dense region
            
        Returns:
            Tuple of (clusters, labels):
                - clusters: list of numpy arrays, each containing point indices in a cluster
                - labels: numpy array of cluster labels for each point (-1 = noise)
    )pbdoc",
    py::arg("data"), py::arg("eps"), py::arg("min_pts"));

    // DBSCAN for 3D points
    m.def("dbscan_3d", [](py::array_t<float, py::array::c_style | py::array::forcecast> py_points, 
                          float eps, int min_pts) {
        std::vector<point3> points = array3d_to_points(py_points);
        auto result = dbscan(std::span<const point3>(points), eps, min_pts);
        return result_to_python(result);
    },
    R"pbdoc(
        DBSCAN clustering for 3D points.
        
        Args:
            data: numpy array of shape (N, 3) with float32 dtype
            eps: maximum distance between points in the same neighborhood
            min_pts: minimum number of points to form a dense region
            
        Returns:
            Tuple of (clusters, labels):
                - clusters: list of numpy arrays, each containing point indices in a cluster
                - labels: numpy array of cluster labels for each point (-1 = noise)
    )pbdoc",
    py::arg("data"), py::arg("eps"), py::arg("min_pts"));

    // Unified interface that auto-detects dimensionality
    m.def("dbscan", [](py::array_t<float, py::array::c_style | py::array::forcecast> py_points, 
                       float eps, int min_pts) {
        auto buf = py_points.request();
        
        if (buf.ndim != 2) {
            throw std::runtime_error("Input array must be 2D with shape (N, dim)");
        }
        
        if (buf.shape[1] == 2) {
            std::vector<point2> points = array2d_to_points(py_points);
            auto result = dbscan(std::span<const point2>(points), eps, min_pts);
            return result_to_python(result);
        } 
        else if (buf.shape[1] == 3) {
            std::vector<point3> points = array3d_to_points(py_points);
            auto result = dbscan(std::span<const point3>(points), eps, min_pts);
            return result_to_python(result);
        }
        else {
            throw std::runtime_error("Only 2D and 3D points are supported (shape must be (N, 2) or (N, 3))");
        }
    },
    R"pbdoc(
        DBSCAN clustering with automatic dimensionality detection.
        
        Args:
            data: numpy array of shape (N, 2) or (N, 3) with float32 dtype
            eps: maximum distance between points in the same neighborhood
            min_pts: minimum number of points to form a dense region
            
        Returns:
            Tuple of (clusters, labels):
                - clusters: list of numpy arrays, each containing point indices in a cluster
                - labels: numpy array of cluster labels for each point (-1 = noise)
    )pbdoc",
    py::arg("data"), py::arg("eps"), py::arg("min_pts"));
}