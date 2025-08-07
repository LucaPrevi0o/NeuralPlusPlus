#ifndef NEURAL_H
#define NEURAL_H

#include <math.h>
#include "tensor.h"
#include "function.h"

/**
 * @brief Namespace for neural network related classes and functions.
 */
namespace neural {

    /**
     * @brief Linear activation function.
     * 
     */
    class LINEAR : public activation {

        public:

            float f(float x) const override { return x; }
            float df(float x) const override { return 1; }
            activation *clone() const override { return new LINEAR(); }
    };

    /**
     * @brief Sigmoid activation function.
     * 
     */
    class SIGMOID : public activation {

        public:

            float f(float x) const override { return 1 / (1 + exp(-x)); }
            float df(float x) const override { return f(x) * (1 - f(x)); }
            activation *clone() const override { return new SIGMOID(); }
    };

    /**
     * @brief SiLU (sigmoid linear unit) activation function.
     * 
     */
    class SILU : public activation {

        public:

            float f(float x) const override { return x / (1 + exp(-x)); }
            float df(float x) const override { return (1 + exp(-x) * (1 + x)) / ((1 + exp(-x)) * (1 + exp(-x))); }
            activation *clone() const override { return new SILU(); }
    };
    
    /**
     * @brief Hyperbolic tangent activation function.
     * 
     */
    class TANH : public activation {

        public:

            float f(float x) const override { return (exp(x) - exp(-x)) / (exp(x) + exp(-x)); }
            float df(float x) const override { return 1 - f(x) * f(x); }
            activation *clone() const override { return new TANH(); }
    };

    /**
     * @brief Softplus activation function.
     * 
     */
    class SOFTPLUS : public activation {

        public:

            float f(float x) const override { return log(1 + exp(x)); }
            float df(float x) const override { return 1 / (1 + exp(-x)); }
            activation *clone() const override { return new SOFTPLUS(); }
    };

    /**
     * @brief Mean Squared Error loss function.
     * 
     */
    class MSE : public loss {

        public:

            float f(float x, float y) const override { return (x - y) * (x - y); }
            float df(float x, float y) const override { return 2 * (x - y); }
            loss *clone() const override { return new MSE(); }
    };

    /**
     * @brief Mean Absolute Error loss function.
     * 
     */
    class MAE : public loss {

        public:

            float f(float x, float y) const override { return abs(x - y); }
            float df(float x, float y) const override { return (x == y ? 0 : ((x > y) ? 1 : -1)); }
            loss *clone() const override { return new MAE(); }
    };

    /**
     * @brief Binary Cross-Entropy loss function.
     *
     */
    class BCE : public loss {

        public:

            float f(float x, float y) const override { return -(y * log(x) + (1 - y) * log(1 - x)); }
            float df(float x, float y) const override { return -(y / x) + (1 - y) / (1 - x); }
            loss *clone() const override { return new BCE(); }
    };

    class network {
        
        private:

            struct test {

                std::matrix<float> weights; // Weights of the neural network
                std::matrix<float> biases;  // Biases of the neural network
                std::matrix<float> neurons; // Layers of the neural network
                activation *function; // Activation functions for each layer

                // Constructor for the test structure
                test() : weights(0), biases(0), neurons(0), function(nullptr) {}

                test(std::matrix<float> w, std::matrix<float> b, std::matrix<float> z, activation *a)
                    : weights(w), biases(b), neurons(z), function(a) {}

                test(const test& other)
                    : weights(other.weights), biases(other.biases), neurons(other.neurons), function(other.function ? other.function -> clone() : nullptr) {}

                test operator=(const test& other) {

                    if (this == &other) return *this; // Check for self-assignment

                    // Clean up existing resources
                    if (function) delete function; // Delete existing function
                    function = other.function ? other.function -> clone() : nullptr; // Clone the new function

                    weights = other.weights;
                    biases = other.biases;
                    neurons = other.neurons;

                    return *this; // Return the current object
                }
            };

            test *t; // Test structure for the neural network
            int layers; // Number of layers in the neural network

        public:

            struct layer {

                int size;
                int batch;
                activation *function; // Activation function for the layer

                layer(int size, int batch, activation *function) : size(size), batch(batch), function(function) {}
                layer(int size, activation *function) : size(size), batch(1), function(function) {}
            };

            template<typename... Args>
            network(Args... args) {

                layer dims[] = {args...};
                layers = sizeof...(args);

                for (auto i = 0; i < layers; i++) {

                    t[i].neurons = std::matrix<float>(dims[i].size, dims[i].batch);
                    t[i].function = dims[i].function -> clone(); // Clone the activation function for the layer

                    if (i < layers - 1) {

                        t[i].weights = std::matrix<float>(dims[i + 1].size, dims[i].size);
                        t[i].biases = std::matrix<float>(dims[i + 1].size, 1);

                        // Initialize weights and biases
                        for (int j = 0; j < t[i].weights.size()[0]; j++)
                            for (int k = 0; k < t[i].weights.size()[1]; k++) t[i].weights(j, k) = rand() / RAND_MAX;

                        for (int j = 0; j < t[i].biases.size()[0]; j++) t[i].biases(j, 0) = rand() / RAND_MAX;
                    }
                }
            }

            network(const network& other) {

                t = new test[other.layers];
                for (int i = 0; i < other.layers; i++) t[i] = other.t[i];
            }

            network copy() const { return network(*this); } 

            int size() const { return layers; }
            test operator[](int index) const { return t[index]; }

            network forward(std::matrix<float> input) {

                if (input.size()[0] != t[0].weights.size()[0]) throw "Input size does not match first layer size";
                if (input.size()[1] != t[0].neurons.size()[1]) throw "Input batch size does not match first layer batch size";

                auto result(*this); // Create a copy of the current network

                // Set input values to the first layer
                for (auto i = 0; i < input.size()[0]; i++)
                    for (auto j = 0; j < input.size()[1]; j++) result.t[0].neurons(i, j) = input(i, j);

                // Propagate through each layer
                for (auto i = 0; i < layers - 1; i++) {

                    // Compute weighted sum: W * input
                    auto sum = result.t[i].weights * result.t[i].neurons;

                    // Add biases to each column (broadcast biases across batch)
                    for (auto col = 0; col < sum.size()[1]; col++)
                        for (auto row = 0; row < sum.size()[0]; row++) {

                            auto value = sum(row, col) + result.t[i].biases(row, 0);
                            sum(row, col) = value;
                        }

                    // Apply activation function element-wise and store in next layer
                    for (auto row = 0; row < sum.size()[0]; row++)
                        for (auto col = 0; col < sum.size()[1]; col++) {

                            auto activated_value = result.t[i + 1].function -> f(sum(row, col));
                            result.t[i + 1].neurons(row, col) = activated_value;
                        }
                }

                return result; // Return new network with updated layer values  
            }

            network backpropagate(loss *loss_function, float learning_rate, std::matrix<float> target) {

                if (target.size()[0] != t[layers - 1].neurons.size()[0]) throw "Expected output rows must match last layer size";
                if (target.size()[1] != t[layers - 1].neurons.size()[1]) throw "Expected output batch size must match layer batch size";

                auto result(*this); // Create a copy of the current network to modify
                std::matrix<float> deltas[layers]; // Arrays to store deltas for each layer
                auto batch = t[0].neurons.size()[1]; // Batch size

                // Calculate error for output layer
                auto current = t[layers - 1].neurons;
                deltas[layers - 1] = std::matrix<float>(current.size()[0], current.size()[1]);

                // Compute delta for output layer: loss_derivative * activation_derivative (element-wise for each sample)
                for (auto sample = 0; sample < batch; sample++)
                    for (auto i = 0; i < current.size()[0]; i++) {

                        auto loss = loss_function -> df(current(i, sample), target(i, sample));
                        auto activation = t[layers - 1].function -> df(current(i, sample));
                        deltas[layers - 1](i, sample) = loss * activation;
                    }

                // Backpropagate errors through hidden layers
                for (auto l = layers - 2; l >= 1; l--) {

                    deltas[l] = std::matrix<float>(t[l].neurons.size()[0], batch);

                    for (auto sample = 0; sample < batch; sample++)
                        for (auto i = 0; i < t[l].neurons.size()[0]; i++) {

                            auto error = 0.0f;
                            for (auto j = 0; j < t[l + 1].neurons.size()[0]; j++) error += t[l].weights(j, i) * deltas[l + 1](j, sample);
                            auto activation = t[l].function -> df(t[l].neurons(i, sample));
                            deltas[l](i, sample) = error * activation;
                        }
                }

                // Update weights and biases with averaged gradients over the batch
                for (auto l = 0; l < layers - 1; l++) {

                    // Update weights: average gradient over batch
                    for (auto i = 0; i < t[l].weights.size()[0]; i++) {

                        for (auto j = 0; j < t[l].weights.size()[1]; j++) {

                            auto gradient = 0.0f;
                            for (auto sample = 0; sample < batch; sample++) gradient += deltas[l + 1](i, sample) * t[l].neurons(j, sample);
                            gradient /= batch;

                            result.t[l].weights(i, j) = t[l].weights(i, j) - learning_rate * gradient;
                        }

                        // Update biases: average bias gradient over batch
                        auto bias_gradient = 0.0f;
                        for (auto sample = 0; sample < batch; sample++) bias_gradient += deltas[l + 1](i, sample);
                        bias_gradient /= batch;

                        result.t[l].biases(i, 0) = t[l].biases(i, 0) - learning_rate * bias_gradient;
                    }
                }

                return result; // Return new network with updated weights and biases
            }
    };
}

#endif