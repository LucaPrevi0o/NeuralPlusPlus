#ifndef NEURAL_H
#define NEURAL_H

#include <math.h>
#include <stdio.h>
#include "matrix.h"

namespace std {

    /**
     * @brief Namespace for neural network related classes and functions.
     */
    namespace neural {

        /**
         * @brief Abstract class for activation functions.
         * 
         * This class defines the interface for activation and loss functions used in the neural network.
         * It provides methods for computing the function value and its derivative, for both scalar and matrix inputs.
         */
        class Function {

            public:

                /**
                 * @brief Compute the function value for a scalar input.
                 * 
                 * @param x Input value
                 * @return Function value
                 */
                virtual float f(float x) const { return x; } // Default implementation returns the input value;

                /**
                 * @brief Compute the function value for a matrix input.
                 * 
                 * This method applies the function to each element of the input matrix.
                 * 
                 * @param x Input matrix
                 * @return Function value matrix
                 */
                std::data::matrix<float> f(std::data::matrix<float> x) const {

                    std::data::tuple<int> s = x.size();
                    std::data::matrix<float> result(s[0], s[1]);
                    for (int i = 0; i < s[0]; i++) 
                        for (int j = 0; j < s[1]; j++) result(i, j) = f(x(i, j));
                    return result;
                }

                /**
                 * @brief Compute the derivative of the function for a scalar input.
                 * 
                 * @param x Input value
                 * @return Derivative value
                 */
                virtual float df(float x) const { return 1; } // Default implementation returns 1

                /**
                 * @brief Compute the derivative of the function for a matrix input.
                 * 
                 * This method applies the derivative function to each element of the input matrix.
                 * 
                 * @param x Input matrix
                 * @return Derivative value matrix
                 */
                std::data::matrix<float> df(std::data::matrix<float> x) const {

                    std::data::tuple<int> s = x.size();
                    std::data::matrix<float> result(s[0], s[1]);
                    for (int i = 0; i < s[0]; i++) 
                        for (int j = 0; j < s[1]; j++) result(i, j) = df(x(i, j));
                    return result;
                }

                /**
                 * @brief Copy constructor for the Function class.
                 * 
                 * @param other The Function object to copy
                 */
                Function operator=(const Function &other) { return *this; } // Assignment operator

                /**
                 * @brief Destructor for the Function class.
                 * 
                 * This destructor is virtual to allow derived classes to clean up their resources.
                 */
                virtual ~Function() = default;
        };

        /**
         * @brief Abstract class for parametric functions.
         * 
         * This class defines the interface for parametric functions used in the neural network.
         * Parametric functions are functions that depend on an additional parameter.
         * It provides methods for computing the function value and its derivative, for both scalar and matrix inputs.
         */
        class ParametricFunction {

            protected:
            
            public:

                /**
                 * @brief Compute the function value for two scalar inputs.
                 * 
                 * @param x First input value
                 * @param y Second input value
                 * @return Function value
                 */
                virtual float f(float x, float y) const { return (x - y) * (x - y); }

                /**
                 * @brief Compute the function value for two matrix inputs.
                 * 
                 * This method applies the function to each element of the input matrices.
                 * 
                 * @param x First input matrix
                 * @param y Second input matrix
                 * @return Function value matrix
                 */
                std::data::matrix<float> f(std::data::matrix<float> x, std::data::matrix<float> y) const {

                    std::data::tuple<int> s = x.size();
                    std::data::matrix<float> result(s[0], s[1]);
                    for (int i = 0; i < s[0]; i++) 
                        for (int j = 0; j < s[1]; j++) result(i, j) = f(x(i, j), y(i, j));
                    return result;
                }

                /**
                 * @brief Compute the derivative of the function for two scalar inputs.
                 * 
                 * @param x First input value
                 * @param y Second input value
                 * @return Derivative value
                 */
                virtual float df(float x, float y) const { return 2 * (x - y); }

                /**
                 * @brief Compute the derivative of the function for two matrix inputs.
                 * 
                 * This method applies the derivative function to each element of the input matrices.
                 * 
                 * @param x First input matrix
                 * @param y Second input matrix
                 * @return Derivative value matrix
                 */
                std::data::matrix<float> df(std::data::matrix<float> x, std::data::matrix<float> y) const {

                    std::data::tuple<int> s = x.size();
                    std::data::matrix<float> result(s[0], s[1]);
                    for (int i = 0; i < s[0]; i++) 
                        for (int j = 0; j < s[1]; j++) result(i, j) = df(x(i, j), y(i, j));
                    return result;
                }

                /**
                 * @brief Copy constructor for the ParametricFunction class.
                 * 
                 * @param other The ParametricFunction object to copy
                 */
                ParametricFunction operator=(const ParametricFunction &other) { return *this; } // Assignment operator

                /**
                 * @brief Destructor for the ParametricFunction class.
                 * 
                 * This destructor is virtual to allow derived classes to clean up their resources.
                 */
                virtual ~ParametricFunction() = default;
        };

        /**
         * @brief Activation functions for the neural network.
         */
        namespace activation {

            /**
             * @brief Linear activation function.
             * 
             */
            class Linear : public Function {

                public:
                    float f(float x) const override { return x; }
                    float df(float x) const override { return 1; }
            };

            /**
             * @brief Sigmoid activation function.
             * 
             */
            class Sigmoid : public Function {

                public:
                    float f(float x) const override { return 1 / (1 + exp(-x)); }
                    float df(float x) const override { return f(x) * (1 - f(x)); }
            };

            /**
             * @brief SiLU (sigmoid linear unit) activation function.
             * 
             */
            class SiLU : public Function {

                public:
                    float f(float x) const override { return x / (1 + exp(-x)); }
                    float df(float x) const override { return f(x) + (1 - f(x)) * (1 - f(x)); }
            };
            
            /**
             * @brief Hyperbolic tangent activation function.
             * 
             */
            class Tanh : public Function {

                public:
                    float f(float x) const override { return (exp(x) - exp(-x)) / (exp(x) + exp(-x)); }
                    float df(float x) const override { return 1 - f(x) * f(x); }
            };

            /**
             * @brief Softplus activation function.
             * 
             */
            class Softplus : public Function {

                public:
                    float f(float x) const override { return log(1 + exp(x)); }
                    float df(float x) const override { return 1 / (1 + exp(-x)); }
            };
        }

        namespace loss {

            class Mse : public ParametricFunction {

                public:

                    float f(float x, float y) const override { return (x - y) * (x - y); }
                    float df(float x, float y) const override { return 2 * (x - y); }
            };
        }

        /**
         * @brief Neural network class.
         */
        class network {

            private:

                std::data::matrix<float> *weights; // Weights between the layers of the network
                std::data::matrix<float> *layers; // Layers of the network
                std::data::matrix<float> *biases; // Biases of the network
                Function *activations; // Activation functions of the network
                int num_layers; // Size of the network

            public:

                /**
                 * @brief Layer structure.
                 * 
                 * This structure represents a layer in the neural network.
                 */
                typedef struct layer {

                    int neurons; // Number of neurons in the layer
                    Function activation_function; // Activation function for the layer
                
                    /**
                     * @brief Constructor for the layer.
                     * 
                     * @param neurons Number of neurons in the layer
                     * @param activation_function Activation function for the layer
                     */
                    layer(int neurons, Function* activation_function) : neurons(neurons), activation_function(*activation_function) {} // Constructor for the layer
                } layer;

                const int size() { return num_layers; } // Returns the number of layers in the network

                // Constructors and destructor
                template<typename... Args> network(Args... args);
                network(const network &other);
                ~network();

                // Compute function
                template<typename... Args> network compute(Args... input);

                template<typename... Args>
                network backpropagate(float learning_rate, const ParametricFunction *loss_function, Args... expected_output) {

                    float expected_data[] = { expected_output... };
                    int expected_size = sizeof...(expected_output);

                    std::data::matrix<float> expected(expected_size, 1);
                    for (int i = 0; i < expected_size; i++) expected(i, 0) = expected_data[i]; // Set expected output values

                    network n(*this);
                    std::data::matrix<float> grad = loss_function -> df(layers[num_layers - 1], expected);
                    for (int i = n.num_layers - 2; i >= 0; i--) {

                        printf("\nbackpropagation: layer %d\n", i);
                        grad *= activations[i].df(std::data::T(layers[i]) * weights[i] + biases[i]); // Compute gradient
                        printf("debug\n");
                        n.weights[i - 1] -= std::data::T(learning_rate * (grad * std::data::T(layers[i - 1]))); // Update weights
                        printf("debug\n");
                        grad *= std::data::T(weights[i - 1]); // Update gradient
                    }

                    return n; // Return the updated network
                }

                // Index operators
                network operator=(const network &other);
                std::data::matrix<float> operator[](int index) const;
                std::data::matrix<float> operator()(int index) const;
        };
    }
}

/**
 * @brief Neural network constructor.
 * 
 * @param args Sizes of the layers in the network
 */
template<typename... Args>
std::neural::network::network(Args... args) : num_layers(sizeof...(args)) {

    layer sizes[] = { args... };
    weights = new std::data::matrix<float>[num_layers - 1];
    layers = new std::data::matrix<float>[num_layers];
    biases = new std::data::matrix<float>[num_layers - 1];
    activations = new Function[num_layers - 1];

    for (int i = 0; i < num_layers - 1; i++) {

        weights[i] = std::data::matrix<float>(sizes[i].neurons, sizes[i + 1].neurons);
        for (int j = 0; j < sizes[i].neurons * sizes[i + 1].neurons; j++)
            weights[i](j / sizes[i + 1].neurons, j % sizes[i + 1].neurons) = ((float)rand() / RAND_MAX) * 2 - 1;
    }

    for (int i = 0; i < num_layers; i++) layers[i] = std::data::matrix<float>(sizes[i].neurons, 1);

    for (int i = 0; i < num_layers - 1; i++) {
        
        biases[i] = std::data::matrix<float>(1, sizes[i + 1].neurons);
        for (int j = 0; j < sizes[i + 1].neurons; j++) biases[i](0, j) = ((float)rand() / RAND_MAX) * 2 - 1;
    }

    for (int i = 0; i < num_layers - 1; i++) activations[i] = sizes[i].activation_function;
}

/**
 * @brief Neural network copy constructor.
 * 
 * The copy constructor creates a new neural network that is a copy of the given network.
 * 
 * @param other The network to copy
 */
std::neural::network::network(const network &other) : num_layers(other.num_layers) {

    weights = new std::data::matrix<float>[num_layers - 1];
    layers = new std::data::matrix<float>[num_layers];
    biases = new std::data::matrix<float>[num_layers - 1];
    activations = new Function[num_layers - 1];

    for (int i = 0; i < num_layers - 1; i++) weights[i] = other.weights[i];
    for (int i = 0; i < num_layers; i++) layers[i] = other.layers[i];
    for (int i = 0; i < num_layers - 1; i++) biases[i] = other.biases[i];
    for (int i = 0; i < num_layers - 1; i++) activations[i] = other.activations[i];
}

/**
 * @brief Neural network assignment operator.
 * 
 * @param other The network to assign
 * @return A reference to this network
 */
std::neural::network std::neural::network::operator=(const network &other) {

    delete[] weights;
    delete[] layers;
    delete[] biases;
    delete[] activations;

    num_layers = other.num_layers;
    weights = new std::data::matrix<float>[num_layers - 1];
    layers = new std::data::matrix<float>[num_layers];
    biases = new std::data::matrix<float>[num_layers - 1];
    activations = new Function[num_layers - 1];

    for (int i = 0; i < num_layers - 1; i++) weights[i] = other.weights[i];
    for (int i = 0; i < num_layers; i++) layers[i] = other.layers[i];
    for (int i = 0; i < num_layers - 1; i++) biases[i] = other.biases[i];
    for (int i = 0; i < num_layers - 1; i++) activations[i] = other.activations[i];
    return *this;
}

/**
 * @brief Computes the output of the neural network for the given input.
 * 
 * The compute function takes an activation function and input values, computes the output of the network,
 * and returns a new network with the computed values.
 * 
 * @param activation_function The activation function to use.
 * @param input Input values for the network.
 * @return A new network with the computed values.
 */
template<typename... Args>
std::neural::network std::neural::network::compute(Args... input) {

    float input_data[] = { input... };
    int input_size = sizeof...(input);
    
    network n(*this);
    if (input_size != n.layers[0].size()[0]) throw "Input size does not match the size of the first layer";
    for (int i = 0; i < input_size; i++) n.layers[0](i, 0) = input_data[i]; // Set input values
    for (int i = 0; i < n.num_layers - 1; i++) n.layers[i + 1] = std::data::T(activations[i].f(std::data::T(layers[i]) * weights[i] + biases[i])); // Compute the output of the network

    return n; // Return the computed network
}

/**
 * @brief Neural network destructor.
 */
std::neural::network::~network() {

    delete[] weights;
    delete[] layers;
    delete[] biases;
    delete[] activations;
}

/**
 * @brief Neural network index operator.
 * 
 * @param index Index of the weights to return
 * @throws "Index out of bounds" if the index is out of range
 * @return Weights of the network at the given index
 */
std::data::matrix<float> std::neural::network::operator[](int index) const {

    if (index < 0 || index > num_layers - 1) throw "Index out of bounds";
    return weights[index];// | biases[index];
}

/**
 * @brief Neural network index operator.
 * 
 * @param index Index of the layer to return
 * @throws "Index out of bounds" if the index is out of range
 * @return Layer of the network at the given index
 */
std::data::matrix<float> std::neural::network::operator()(int index) const {

    if (index < 0 || index > num_layers) throw "Index out of bounds";
    return layers[index];
}

#endif