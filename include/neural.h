#ifndef NEURAL_H
#define NEURAL_H

#include <math.h>
#include <stdio.h>
#include "tensor.h"
#include "function.h"

namespace std {

    /**
     * @brief Namespace for neural network related classes and functions.
     */
    namespace neural {

        /**
         * @brief Activation functions for the neural network.
         */
        namespace activation {

            /**
             * @brief Linear activation function.
             * 
             */
            class Linear : public Function {

                    float f(float x) const override { return x; }
                    float df(float x) const override { return 1; }
                    Function* clone() const override { return new Linear(*this); }
            };

            /**
             * @brief Sigmoid activation function.
             * 
             */
            class Sigmoid : public Function {

                    float f(float x) const override { return 1 / (1 + exp(-x)); }
                    float df(float x) const override { return f(x) * (1 - f(x)); }
                    Function* clone() const override { return new Sigmoid(*this); }
            };

            /**
             * @brief SiLU (sigmoid linear unit) activation function.
             * 
             */
            class SiLU : public Function {

                    float f(float x) const override { return x / (1 + exp(-x)); }
                    float df(float x) const override { return f(x) + (1 - f(x)) * (1 - f(x)); }
                    Function* clone() const override { return new SiLU(*this); }
            };
            
            /**
             * @brief Hyperbolic tangent activation function.
             * 
             */
            class Tanh : public Function {

                    float f(float x) const override { return (exp(x) - exp(-x)) / (exp(x) + exp(-x)); }
                    float df(float x) const override { return 1 - f(x) * f(x); }
                    Function* clone() const override { return new Tanh(*this); }
            };

            /**
             * @brief Softplus activation function.
             * 
             * */
            class Softplus : public Function {

                    float f(float x) const override { return log(1 + exp(x)); }
                    float df(float x) const override { return 1 / (1 + exp(-x)); }
                    Function* clone() const override { return new Softplus(*this); }
            };
        }

        namespace loss {

            class Mse : public ParametricFunction {

                public:

                    float f(float x, float y) const override { return (x - y) * (x - y); }
                    float df(float x, float y) const override { return 2 * (x - y); }
                    ParametricFunction* clone() const override { return new Mse(*this); }
            };
        }

        /**
         * @brief Neural network class.
         */
        class network {

                std::data::matrix<float> *weights; // Weights between the layers of the network
                std::data::matrix<float> *layers; // Layers of the network
                std::data::matrix<float> *biases; // Biases of the network
                Function **activations; // Activation functions of the network
                int num_layers; // Size of the network

                std::data::matrix<float> compute_layer(int index) { return std::data::T(std::data::T(layers[index]) * weights[index]) + biases[index]; }

            public:

                /**
                 * @brief Layer structure.
                 * 
                 * This structure represents a layer in the neural network.
                 */
                typedef struct layer {

                    int neurons; // Number of neurons in the layer
                    Function* activation_function; // Activation function for the layer
                
                    /**
                     * @brief Constructor for the layer.
                     * 
                     * @param neurons Number of neurons in the layer
                     * @param activation_function Activation function for the layer
                     */
                    layer(int neurons, Function* activation_function) : neurons(neurons), activation_function(activation_function) {} // Constructor for the layer
                } layer;

                int size() { return num_layers; } // Returns the number of layers in the network

                // Constructors and destructor
                template<typename... Args> network(Args... args);
                network(const network &other);
                ~network();

                // Compute function
                template<typename... Args> network compute(Args... input);

                template<typename... Args>
                network backpropagate(float learning_rate, const ParametricFunction *loss_function, Args... expected_output) {

                    float expected_data[] = { static_cast<float>(expected_output)... };
                    int expected_size = sizeof...(expected_output);

                    if (expected_size != layers[num_layers - 1].size()[0]) throw "Expected output size does not match the output layer size";

                    // Compute the output of the network
                    return this; // Return the updated network
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
    activations = new Function*[num_layers - 1];

    for (int i = 0; i < num_layers - 1; i++) {

        weights[i] = std::data::matrix<float>(sizes[i].neurons, sizes[i + 1].neurons);
        for (int j = 0; j < sizes[i].neurons * sizes[i + 1].neurons; j++)
            weights[i](j / sizes[i + 1].neurons, j % sizes[i + 1].neurons) = ((float)rand() / RAND_MAX) * 2 - 1;
    }

    for (int i = 0; i < num_layers; i++) layers[i] = std::data::matrix<float>(sizes[i].neurons, 1);

    for (int i = 0; i < num_layers - 1; i++) {
        
        biases[i] = std::data::matrix<float>(sizes[i + 1].neurons, 1);
        for (int j = 0; j < sizes[i + 1].neurons; j++) biases[i](j, 0) = ((float)rand() / RAND_MAX) * 2 - 1;
    }

    for (int i = 0; i < num_layers - 1; i++) activations[i] = sizes[i].activation_function->clone();
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
    activations = new Function*[num_layers - 1];

    for (int i = 0; i < num_layers - 1; i++) weights[i] = other.weights[i];
    for (int i = 0; i < num_layers; i++) layers[i] = other.layers[i];
    for (int i = 0; i < num_layers - 1; i++) biases[i] = other.biases[i];
    for (int i = 0; i < num_layers - 1; i++) activations[i] = other.activations[i]->clone();
}

/**
 * @brief Neural network assignment operator.
 * 
 * @param other The network to assign
 * @return A reference to this network
 */
std::neural::network std::neural::network::operator=(const network &other) {

    for (int i = 0; i < num_layers - 1; i++) delete activations[i];
    delete[] activations;

    delete[] weights;
    delete[] layers;
    delete[] biases;

    num_layers = other.num_layers;
    weights = new std::data::matrix<float>[num_layers - 1];
    layers = new std::data::matrix<float>[num_layers];
    biases = new std::data::matrix<float>[num_layers - 1];
    activations = new Function*[num_layers - 1];

    for (int i = 0; i < num_layers - 1; i++) weights[i] = other.weights[i];
    for (int i = 0; i < num_layers; i++) layers[i] = other.layers[i];
    for (int i = 0; i < num_layers - 1; i++) biases[i] = other.biases[i];
    for (int i = 0; i < num_layers - 1; i++) activations[i] = other.activations[i]->clone();
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

    for (int i = 1; i < n.num_layers; i++) n.layers[i] = activations[i - 1]->f(compute_layer(i - 1)); // Compute the output of the network
    int input_size = sizeof...(input);
    
    network n(*this);
    if (input_size != layers[0].size()[0]) throw "Input size does not match the size of the first layer";
    for (int i = 0; i < input_size; i++) n.layers[0](i, 0) = input_data[i]; // Set input values
    for (int i = 1; i < n.num_layers; i++) n.layers[i] = activations[i - 1].f(compute_layer(i - 1)); // Compute the output of the network

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