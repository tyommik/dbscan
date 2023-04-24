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
        points.emplace_back(ptr[2 * i], ptr[2 * i + 1]);
    }

    return points;
}


PYBIND11_MODULE(fast_dbscan, m) {
    m.doc() = "Python bindings for DBSCAN C++ library";

    m.def("dbscan_2d", [&](py::array_t<float> py_points, float eps, int min_pts) {
        std::vector<point2> points = array2d_to_points(py_points);
        auto result = dbscan(points, eps, min_pts);
        auto& clusters = result.first;
        auto& cluster_indices = result.second;

        py::list py_clusters;
        for (const auto& cluster : clusters) {
            py::array_t<size_t> py_cluster(cluster.size(), cluster.data());
            py_clusters.append(py_cluster);
        }

        py::array_t<int> py_cluster_indices(cluster_indices.size(), cluster_indices.data());

        return py::make_tuple(py_clusters, py_cluster_indices);
    }, "DBSCAN algorithm for 2D points",
        py::arg("data"), py::arg("eps"), py::arg("min_pts"));

    py::class_<point2>(m, "Point2")
        .def(py::init<>())
        .def(py::init<float, float>(), py::arg("x"), py::arg("y"))
        .def_readwrite("x", &point2::x)
        .def_readwrite("y", &point2::y);

    py::class_<point3>(m, "Point3")
        .def(py::init<>())
        .def(py::init<float, float, float>(), py::arg("x"), py::arg("y"), py::arg("z"))
        .def_readwrite("x", &point3::x)
        .def_readwrite("y", &point3::y)
        .def_readwrite("z", &point3::z);
}

