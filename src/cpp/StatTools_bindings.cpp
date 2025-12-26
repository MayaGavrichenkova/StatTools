#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cstring> // For memcpy
#include "StatTools_core.h"

namespace py = pybind11;

// pybind11 bindings
PYBIND11_MODULE(StatTools_bindings, m) {
    m.doc() = "Modern pybind11 bindings for StatTools C/C++ functions";

    // Waiting time calculation function
    m.def("get_waiting_time", [](py::array_t<double> input_vector, py::array_t<double> U, double C0_input) {
        // Convert numpy arrays to std::vector
        std::vector<double> input_vec(input_vector.data(), input_vector.data() + input_vector.size());
        std::vector<double> u_vec(U.data(), U.data() + U.size());

        // Call the model function
        std::vector<double> result = model(input_vec, u_vec, C0_input);

        // Return as numpy array
        return py::array_t<double>(result.size(), result.data(), py::none());
    }, "Calculate average waiting time curve for given input vector and utilization factors",
          py::arg("input_vector"), py::arg("U"), py::arg("C0_input") = -1.0);

    // Random value generators
    m.def("get_exponential_dist_value", &get_exponential_dist_value,
          "Generate a single exponential distribution value", py::arg("lambda"));

    m.def("get_gauss_dist_value", &get_gauss_dist_value,
          "Generate a single Gaussian distribution value");

    // Vector generators
    m.def("get_exp_dist_vector", [](double lambda, int size) {
        std::vector<double> result = get_exp_dist_vector(lambda, size);
        return py::array_t<double>(result.size(), result.data(), py::none());
    }, "Generate a vector of exponential distribution values",
          py::arg("lambda"), py::arg("size"));

    m.def("get_poisson_thread", [](py::array_t<double> input_vector, double divisor) {
        std::vector<double> input_vec(input_vector.data(), input_vector.data() + input_vector.size());
        std::vector<double> result = get_poisson_thread(input_vec, divisor);
        return py::array_t<double>(result.size(), result.data(), py::none());
    }, "Generate Poisson thread from input vector",
          py::arg("input_vector"), py::arg("divisor") = 1.0);

    // Utility functions
    m.def("cumsum", [](py::array_t<double> input_vector) {
        std::vector<double> input_vec(input_vector.data(), input_vector.data() + input_vector.size());
        std::vector<double> result = cumsum(input_vec);
        // Create a new numpy array with copied data to avoid memory issues
        py::array_t<double> output = py::array_t<double>(result.size());
        std::memcpy(output.mutable_data(), result.data(), result.size() * sizeof(double));
        return output;
    }, "Compute cumulative sum of input vector", py::arg("input_vector"));

    m.def("model", [](py::array_t<double> input_vector, py::array_t<double> U, double C0_global) {
        std::vector<double> input_vec(input_vector.data(), input_vector.data() + input_vector.size());
        std::vector<double> u_vec(U.data(), U.data() + U.size());
        std::vector<double> result = model(input_vec, u_vec, C0_global);
        return py::array_t<double>(result.size(), result.data(), py::none());
    }, "Main model function for queueing theory calculations",
          py::arg("input_vector"), py::arg("U"), py::arg("C0_global") = -1.0);
}
