#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/chrono.h>
#include <vector>
#include <iostream>
#include <cmath>
#include <algorithm>

//g++ -O2 -Wall -shared -std=c++20 -fPIC `python3.10 -m pybind11 --includes` svm.cpp -o svm`python3-config --extension-suffix` -I/path/to/eigen


namespace py = pybind11;

class SVM {
public:
    SVM(double learning_rate, double regularization_strength, int iterations) 
        : learning_rate(learning_rate), regularization_strength(regularization_strength), iterations(iterations) {}

    void fit(py::array_t<double> X, py::array_t<double> y) {
        py::buffer_info X_info = X.request();
        py::buffer_info y_info = y.request();

        double* X_ptr = static_cast<double*>(X_info.ptr);
        double* y_ptr = static_cast<double*>(y_info.ptr);

        size_t num_samples = X_info.shape[0];
        size_t num_features = X_info.shape[1];

        // Initialize weights to zero
        weights = std::vector<double>(num_features, 0.0);
        bias = 0.0;

        // Stochastic Gradient Descent
        for (int i = 0; i < iterations; ++i) {
            for (size_t j = 0; j < num_samples; ++j) {
                double linear_output = bias;
                for (size_t k = 0; k < num_features; ++k) {
                    linear_output += weights[k] * X_ptr[j * num_features + k];
                }
                double y_pred = y_ptr[j] * linear_output;

                // Hinge loss
                if (y_pred < 1) {
                    // Misclassified point, update the weights and bias
                    for (size_t k = 0; k < num_features; ++k) {
                        weights[k] = (1 - learning_rate * regularization_strength) * weights[k] + learning_rate * y_ptr[j] * X_ptr[j * num_features + k];
                    }
                    bias += learning_rate * y_ptr[j];
                } else {
                    // Correctly classified point, only regularize the weights
                    for (size_t k = 0; k < num_features; ++k) {
                        weights[k] = (1 - learning_rate * regularization_strength) * weights[k];
                    }
                }
            }

            // Optional: Print loss every 100 iterations
            if (i % 100 == 0) {
                double loss = compute_loss(X_ptr, y_ptr, num_samples, num_features);
                std::cout << "Iteration " << i << ", Loss: " << loss << std::endl;
            }
        }
    }

    py::array_t<double> predict(py::array_t<double> X) {
        py::buffer_info X_info = X.request();
        double* X_ptr = static_cast<double*>(X_info.ptr);
        size_t num_samples = X_info.shape[0];
        size_t num_features = X_info.shape[1];

        py::array_t<double> result(num_samples);
        py::buffer_info result_info = result.request();
        double* result_ptr = static_cast<double*>(result_info.ptr);

        for (size_t j = 0; j < num_samples; ++j) {
            double linear_output = bias;
            for (size_t k = 0; k < num_features; ++k) {
                linear_output += weights[k] * X_ptr[j * num_features + k];
            }
            result_ptr[j] = (linear_output >= 0) ? 1.0 : -1.0;
        }

        return result;
    }

private:
    std::vector<double> weights;
    double bias;
    double learning_rate;
    double regularization_strength;
    int iterations;

    double compute_loss(double* X_ptr, double* y_ptr, size_t num_samples, size_t num_features) {
        double loss = 0.0;
        for (size_t i = 0; i < num_samples; ++i) {
            double linear_output = bias;
            for (size_t j = 0; j < num_features; ++j) {
                linear_output += weights[j] * X_ptr[i * num_features + j];
            }
            loss += std::max(0.0, 1 - y_ptr[i] * linear_output);
        }

        // Regularization term
        double reg_term = 0.0;
        for (double weight : weights) {
            reg_term += weight * weight;
        }
        reg_term *= 0.5 * regularization_strength;

        return loss + reg_term;
    }
};

PYBIND11_MODULE(svm_lib, m) {
    py::class_<SVM>(m, "SVM")
        .def(py::init<double, double, int>())
        .def("fit", &SVM::fit)
        .def("predict", &SVM::predict);
}
