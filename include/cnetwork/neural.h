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

    class neural {
        
        private:

            std::matrix<float> *weights; // Weights of the neural network
            std::matrix<float> *biases;  // Biases of the neural network
            std::matrix<float> *layers; // Layers of the neural network
            activation **activations; // Activation functions for each layer

            int num_layers; // Number of layers in the neural network

        public:

            struct layer {

                int size;
                int batch;
                activation *function; // Activation function for the layer

                layer(int size, int batch, activation *function) : size(size), batch(batch), function(function) {}
                layer(int size, activation *function) : size(size), batch(1), function(function) {}
            };

            template<typename... Args>
            neural(Args... args) {

                layer dims[] = {args...};
                num_layers = sizeof...(args);

                layers = new std::matrix<float>[num_layers];
                weights = new std::matrix<float>[num_layers - 1];
                biases = new std::matrix<float>[num_layers - 1];
                activations = new activation[num_layers];

                for (auto i = 0; i < num_layers; ++i) {

                    layers[i] = std::matrix<float>(dims[i].size, dims[i].batch);
                    activations[i] = dims[i].function;

                    if (i < num_layers - 1) {

                        weights[i] = std::matrix<float>(dims[i + 1].size, dims[i].size);
                        biases[i] = std::matrix<float>(dims[i + 1].size, 1);

                        // Initialize weights and biases
                        for (int j = 0; j < weights[i].size()[0]; j++)
                            for (int k = 0; k < weights[i].size()[1]; k++) weights[i](j, k) = rand() / RAND_MAX;

                        for (int j = 0; j < biases[i].size()[0]; j++) biases[i](j, 0) = rand() / RAND_MAX;
                    }
                }
            }

            neural(const neural& other) {

                num_layers = other.num_layers;
                layers = new std::matrix<float>[num_layers];
                weights = new std::matrix<float>[num_layers - 1];
                biases = new std::matrix<float>[num_layers - 1];
                activations = new activation*[num_layers];

                for (int i = 0; i < num_layers; ++i) {

                    layers[i] = other.layers[i];
                    activations[i] = other.activations[i] -> clone(); // Assuming activation has a clone method

                    if (i < num_layers - 1) {

                        weights[i] = other.weights[i];
                        biases[i] = other.biases[i];
                    }
                }
            }

            neural copy() const { return neural(*this); } 

            neural forward(std::matrix<float> input) {

                if (input.size()[0] != layers[0].size()[0]) throw "Input size does not match first layer size";
                if (input.size()[1] != layers[0].size()[1]) throw "Input batch size does not match first layer batch size";

                auto result(*this); // Create a copy of the current network

                // Set input values to the first layer
                for (auto i = 0; i < input.size()[0]; i++)
                    for (auto j = 0; j < input.size()[1]; j++) result.layers[0](i, j) = input(i, j);

                // Propagate through each layer
                for (auto i = 0; i < num_layers - 1; i++) {

                    // Compute weighted sum: W * input
                    auto weighted_sum = result.weights[i] * result.layers[i];

                    // Add biases to each column (broadcast biases across batch)
                    for (auto col = 0; col < weighted_sum.size()[1]; col++)
                        for (auto row = 0; row < weighted_sum.size()[0]; row++) {

                            auto value = weighted_sum(row, col) + result.biases[i](row, 0);
                            weighted_sum(row, col) = value;
                        }

                    // Apply activation function element-wise and store in next layer
                    for (auto row = 0; row < weighted_sum.size()[0]; row++)
                        for (auto col = 0; col < weighted_sum.size()[1]; col++) {

                            auto activated_value = result.activations[i + 1] -> f(weighted_sum(row, col));
                            result.layers[i + 1](row, col) = activated_value;
                        }
                }

                return result; // Return new network with updated layer values  
            }

            neural backpropagate(loss *loss_function, float learning_rate, std::matrix<float> target) {

                if (target.size()[0] != layers[num_layers - 1].size()[0]) throw "Expected output rows must match last layer size";
                if (target.size()[1] != layers[num_layers - 1].size()[1]) throw "Expected output batch size must match layer batch size";

                auto result(*this); // Create a copy of the current network to modify
                std::matrix<float> deltas[num_layers]; // Arrays to store deltas for each layer
                auto batch = layers[0].size()[1]; // Batch size

                // Calculate error for output layer
                auto current = layers[num_layers - 1];
                deltas[num_layers - 1] = std::matrix<float>(current.size()[0], current.size()[1]);

                // Compute delta for output layer: loss_derivative * activation_derivative (element-wise for each sample)
                for (auto sample = 0; sample < batch; sample++)
                    for (auto i = 0; i < current.size()[0]; i++) {

                        auto loss = loss_function -> df(current(i, sample), target(i, sample));
                        auto activation = activations[num_layers - 1] -> df(current(i, sample));
                        deltas[num_layers - 1](i, sample) = loss * activation;
                    }

                // Backpropagate errors through hidden layers
                for (auto l = num_layers - 2; l >= 1; l--) {

                    deltas[l] = std::matrix<float>(layers[l].size()[0], batch);

                    for (auto sample = 0; sample < batch; sample++)
                        for (auto i = 0; i < layers[l].size()[0]; i++) {

                            auto error = 0.0f;
                            for (auto j = 0; j < layers[l + 1].size()[0]; j++) error += weights[l](j, i) * deltas[l + 1](j, sample);
                            auto activation = activations[l] -> df(layers[l](i, sample));
                            deltas[l](i, sample) = error * activation;
                        }
                }

                // Update weights and biases with averaged gradients over the batch
                for (auto l = 0; l < num_layers - 1; l++) {

                    // Update weights: average gradient over batch
                    for (auto i = 0; i < weights[l].size()[0]; i++) {

                        for (auto j = 0; j < weights[l].size()[1]; j++) {

                            auto gradient = 0.0f;
                            for (auto sample = 0; sample < batch; sample++) gradient += deltas[l + 1](i, sample) * layers[l](j, sample);
                            gradient /= batch;

                            result.weights[l](i, j) = weights[l](i, j) - learning_rate * gradient;
                        }

                        // Update biases: average bias gradient over batch
                        auto bias_gradient = 0.0f;
                        for (auto sample = 0; sample < batch; sample++) bias_gradient += deltas[l + 1](i, sample);
                        bias_gradient /= batch;

                        result.biases[l](i, 0) = biases[l](i, 0) - learning_rate * bias_gradient;
                    }
                }

                return result; // Return new network with updated weights and biases
            }
    };
}

#endif