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

    /**
     * @brief Class representing a feedforward neural network.
     * 
     * This class provides methods for creating, training, and using a feedforward neural network.
     */
    class network {
        
        private:

            /**
             * @brief Structure representing a layer in the neural network.
             * 
             */
            struct layer {

                tensor::matrix<float> weights; // Weights of the neural network
                tensor::matrix<float> biases;  // Biases of the neural network
                tensor::matrix<float> neurons; // Layers of the neural network
                activation *function; // Activation functions for each layer

                /**
                 * @brief Construct a new layer object.
                 */
                layer() : weights(0, 0), biases(0, 0), neurons(0, 0), function(0) {}

                /**
                 * @brief Construct a new layer object.
                 * 
                 * @param w Weights of the layer
                 * @param b Biases of the layer
                 * @param z Neurons of the layer
                 * @param a Activation function for the layer
                 */
                layer(tensor::matrix<float> w, tensor::matrix<float> b, tensor::matrix<float> z, activation *a)
                    : weights(w), biases(b), neurons(z), function(a) {}

                /**
                 * @brief Copy constructor for the layer.
                 * 
                 * @param other The layer to copy from
                 */
                layer(const layer& other)
                    : weights(other.weights), biases(other.biases), neurons(other.neurons), function(other.function -> clone()) {}

                /**
                 * @brief Assignment operator for the layer.
                 * 
                 * @param other The layer to copy from
                 * @return layer A reference to the current layer
                 */
                layer operator=(const layer& other) {

                    if (this == &other) return *this; // Check for self-assignment

                    weights = other.weights;
                    biases = other.biases;
                    neurons = other.neurons;
                    if (function) delete function; // Delete old function
                    function = other.function -> clone(); // Clone the new function

                    return *this; // Return the current object
                }

                /**
                 * @brief Destructor for the layer.
                 */
                ~layer() { if (function) delete function; }
            };

            layer *layers; // Layers of the neural network
            int size, batch; // Number of layers in the neural network

        public:

            /**
             * @brief Structure representing the shape of a layer in the neural network.
             * 
             */
            struct shape {

                int size;             // Size of the layer
                activation *function; // Activation function for the layer

                /**
                 * @brief Construct a new shape object.
                 * 
                 * @param size Size of the layer
                 * @param function Activation function for the layer
                 */
                shape(int size, activation *function) : size(size), function(function) {}
            };

            /**
             * @brief Constructor for the neural network.
             * 
             * @param batch_size Number of samples to process in parallel
             * @param args Shape of each layer (size and activation function)
             */
            template<typename... Args>
            network(int batch_size, Args... args) : size(sizeof...(args)), batch(batch_size) {

                if (batch < 1) throw "Batch size must be positive";

                shape dims[] = {args...};
                layers = new layer[size]; // Allocate memory for the layers

                for (auto i = 0; i < size; i++) {

                    // Create neurons matrix: features x batch
                    layers[i].neurons = tensor::matrix<float>(dims[i].size, batch);
                    layers[i].function = dims[i].function -> clone(); // Clone the activation function for the layer

                    if (i < size - 1) {

                        // Weights: output_neurons x input_neurons
                        layers[i].weights = tensor::matrix<float>(dims[i + 1].size, dims[i].size);

                        // Biases: output_neurons x 1 (will be broadcasted)
                        layers[i].biases = tensor::matrix<float>(dims[i + 1].size, 1);

                        // Initialize weights with Xavier/Glorot initialization
                        auto limit = sqrt(6.0f / (dims[i].size + dims[i + 1].size));
                        for (auto j = 0; j < layers[i].weights.size(0); j++)
                            for (auto k = 0; k < layers[i].weights.size(1); k++) 
                                layers[i].weights(j, k) = (2.0f * (rand() / float(RAND_MAX)) - 1.0f) * limit;

                        // Initialize biases to zero
                        for (auto j = 0; j < layers[i].biases.size(0); j++) layers[i].biases(j, 0) = 0.0f;
                    }
                }
            }

            /**
             * @brief Copy constructor for the neural network.
             * 
             * @param other The neural network to copy from
             */
            network(const network& other) : size(other.size), batch(other.batch) {

                layers = new layer[size]; // Allocate memory for the layers
                for (int i = 0; i < other.size; i++) layers[i] = other.layers[i];
            }

            /**
             * @brief Assignment operator for the neural network.
             * 
             * @param other The neural network to assign from
             * @return Reference to the current object
             */
            network& operator=(const network& other) {

                if (this == &other) return *this; // Check for self-assignment

                delete[] layers; // Delete the old layers

                size = other.size; // Copy the size of the network
                batch = other.batch; // Copy the batch size
                layers = new layer[size]; // Allocate memory for the new layers
                for (int i = 0; i < size; i++) layers[i] = other.layers[i];

                return *this; // Return the current object
            }

            /**
             * @brief Destructor for the neural network.
             */
            ~network() { delete[] layers; }

            /**
             * @brief Get the depth of the neural network.
             * 
             * @return The number of layers in the network
             */
            int depth() const { return size; }

            /**
             * @brief Get a specific layer of the neural network.
             * 
             * @param index The index of the layer to retrieve
             * @return The requested layer
             */
            layer operator[](int index) const { return layers[index]; }

            /**
             * @brief Get the batch size of the network.
             * 
             * @return The batch size
             */
            int batch_size() const { return batch; }

            /**
             * @brief Forward pass through the network.
             * 
             * @param input The input matrix for the first layer (features x batch_size)
             * @return The output of the network after the forward pass
             */
            network forward(tensor::matrix<float> input) {

                if (input.size(0) != layers[0].neurons.size(0)) throw "Input features do not match first layer size";
                if (input.size(1) != batch) throw "Input batch size does not match network batch size";

                auto result(*this); // Create a copy of the current network

                // Set input values to the first layer
                for (auto i = 0; i < input.size(0); i++)
                    for (auto j = 0; j < input.size(1); j++) 
                        result.layers[0].neurons(i, j) = input(i, j);

                // Propagate through each layer
                for (auto i = 0; i < size - 1; i++) {

                    // Compute weighted sum: W * input
                    auto sum = result.layers[i].weights * result.layers[i].neurons;

                    // Add biases to each column (broadcast biases across batch)
                    for (auto col = 0; col < sum.size(1); col++)
                        for (auto row = 0; row < sum.size(0); row++) {
                            auto value = sum(row, col) + result.layers[i].biases(row, 0);
                            sum(row, col) = value;
                        }

                    // Apply activation function element-wise and store in next layer
                    for (auto row = 0; row < sum.size(0); row++)
                        for (auto col = 0; col < sum.size(1); col++) {
                            auto activated_value = result.layers[i + 1].function -> f(sum(row, col));
                            result.layers[i + 1].neurons(row, col) = activated_value;
                        }
                }

                return result; // Return new network with updated layer values  
            }

            /**
             * @brief Backpropagation algorithm for training the network.
             * 
             * @param loss_function The loss function to use for training
             * @param learning_rate The learning rate for weight updates
             * @param target The target output for the network (features x batch_size)
             * @return The updated network after backpropagation
             */
            network backpropagate(loss *loss_function, float learning_rate, tensor::matrix<float> target) {

                auto layer = layers[size - 1]; // Last layer of the network

                if (target.size(0) != layer.neurons.size(0)) throw "Target features must match last layer size";
                if (target.size(1) != batch) throw "Target batch size must match layer batch size";

                auto result(*this); // Create a copy of the current network to modify
                tensor::matrix<float> **deltas = new tensor::matrix<float>*[size]; // Arrays to store deltas for each layer
                auto batch = layers[0].neurons.size(1); // Batch size

                // Calculate error for output layer
                deltas[size - 1] = new tensor::matrix<float>(layer.neurons.size(0), batch);

                // Compute delta for output layer: loss_derivative * activation_derivative (element-wise for each sample)
                for (auto sample = 0; sample < batch; sample++)
                    for (auto i = 0; i < layer.neurons.size(0); i++) {

                        auto loss = loss_function -> df(layer.neurons(i, sample), target(i, sample));
                        auto activation = layers[size - 1].function -> df(layer.neurons(i, sample));
                        auto delta = loss * activation;
                        (*(deltas[size - 1]))(i, sample) = delta; // Store delta for output neuron
                    }

                // Backpropagate errors through hidden layers
                for (auto l = size - 2; l >= 1; l--) {

                    auto layer = layers[l];
                    deltas[l] = new tensor::matrix<float>(layer.neurons.size(0), batch);

                    for (auto sample = 0; sample < batch; sample++)
                        for (auto i = 0; i < layer.neurons.size(0); i++) {

                            auto error = 0.0f;
                            for (auto j = 0; j < layers[l + 1].neurons.size(0); j++) 
                                error += layer.weights(j, i) * (*(deltas[l + 1]))(j, sample);
                            auto activation = layer.function -> df(layer.neurons(i, sample));
                            (*(deltas[l]))(i, sample) = error * activation;
                        }
                }

                // Update weights and biases with averaged gradients over the batch
                for (auto l = 0; l < size - 1; l++) {

                    auto layer = layers[l];
                    // Update weights: average gradient over batch
                    for (auto i = 0; i < layer.weights.size(0); i++) {

                        for (auto j = 0; j < layer.weights.size(1); j++) {

                            auto gradient = 0.0f;
                            for (auto sample = 0; sample < batch; sample++) 
                                gradient += (*(deltas[l + 1]))(i, sample) * layer.neurons(j, sample);
                            gradient /= batch;

                            result.layers[l].weights(i, j) = layer.weights(i, j) - learning_rate * gradient;
                        }

                        // Update biases: average bias gradient over batch
                        auto bias_gradient = 0.0f;
                        for (auto sample = 0; sample < batch; sample++) 
                            bias_gradient += (*(deltas[l + 1]))(i, sample);
                        bias_gradient /= batch;

                        result.layers[l].biases(i, 0) = layer.biases(i, 0) - learning_rate * bias_gradient;
                    }
                }

                // Clean up delta matrices
                for (auto i = 1; i < size; i++) delete deltas[i];
                delete[] deltas;

                return result; // Return new network with updated weights and biases
            }
    };

    /**
     * @brief Train the neural network using the specified loss function and training parameters.
     * 
     * @param n The neural network to train
     * @param input The input matrix for training (features x batch_size)
     * @param target The target output matrix for training (features x batch_size)
     * @param loss The loss function to use for training
     * @param epochs The maximum number of training epochs
     * @param max_error The maximum acceptable error for early stopping
     * @param learning_rate The learning rate for weight updates
     * @return network The trained neural network
     */
    network train(network n, tensor::matrix<float> input, tensor::matrix<float> target, loss* loss, int epochs, float max_error, float learning_rate) {

        for (auto epoch = 0; epoch < epochs; epoch++) {

            n = n.forward(input); // Forward pass through the network

            auto error = n[n.depth() - 1].neurons - target; // Calculate error at output layer
            auto early_stop = true; // Flag for early stopping if error is within acceptable range
            
            // Check if all errors are within acceptable range
            for (auto i = 0; i < error.size(0); i++) {
                for (auto j = 0; j < error.size(1); j++) if (abs(error(i, j)) >= max_error) {

                    early_stop = false;
                    break;
                }
                if (!early_stop) break;
            }

            if (early_stop) break; // Stop training if error is within acceptable range
            n = n.backpropagate(loss, learning_rate, target); // Backpropagate error and update weights
        }
        return n; // Return the trained network
    }

    /**
     * @brief Train the neural network using the specified loss function and training parameters.
     * 
     * @param n The neural network to train
     * @param input The input matrix for training (features x batch_size)
     * @param target The target output matrix for training (features x batch_size)
     * @param loss The loss function to use for training
     * @param epochs The maximum number of training epochs
     * @param learning_rate The learning rate for weight updates
     * @return network The trained neural network
     */
    network train(network n, tensor::matrix<float> input, tensor::matrix<float> target, loss* loss, int epochs, float learning_rate) {

        for (auto epoch = 0; epoch < epochs; epoch++) {

            n = n.forward(input); // Forward pass through the network
            n = n.backpropagate(loss, learning_rate, target); // Backpropagate error and update weights
        }
        return n; // Return the trained network
    }
};  

#endif