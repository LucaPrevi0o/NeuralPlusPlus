#include "include/tensor.h"
#include "include/cnetwork/neural.h"
#include <stdio.h>
#include <time.h>

using namespace neural; // Neural network namespace
using namespace tensor; // Tensor namespace

int main() {

    srand(static_cast<unsigned int>(time(0)));

    int num_epochs = 350;
    float max_error = 0.01;

    try {

        // Create a neural network with 3 layers
        network net(
            network::shape(3, new SIGMOID()), // First layer with 3 neurons and sigmoid activation
            network::shape(4, new SIGMOID()), // Second layer with 4 neurons and sigmoid activation
            network::shape(2, new SIGMOID())  // Output layer with 2 neurons and sigmoid activation
        );

        // Create an input matrix with 3 features and 1 sample
        matrix<float> input(net[0].neurons.size(0), 1);
        printf("Input: [ ");
        for (auto i = 0; i < input.size(0); i++) {

            input(i, 0) = static_cast<float>(rand()) / RAND_MAX; // Random input between 0 and 1
            printf("%.2f ", input(i, 0));
        }
        printf("]\n");

        // Create a target matrix for training
        matrix<float> target(net[net.depth() - 1].neurons.size(0), 1);
        printf("Target: [ ");
        for (auto i = 0; i < target.size(0); i++) {
            target(i, 0) = static_cast<float>(rand()) / RAND_MAX; // Random target between 0 and 1
            printf("%.2f ", target(i, 0));
        }
        printf("]\n");

        auto output = train(net, input, target, new MAE(), num_epochs, max_error, 0.05f);

        printf("Output: [ ");
        for (auto sample = 0; sample < output[output.depth() - 1].neurons.size(1); sample++)
            for (auto neuron = 0; neuron < output[output.depth() - 1].neurons.size(0); neuron++)
                printf("%.3f ", output[output.depth() - 1].neurons(neuron, sample));
        printf("]\n");

        return 0;

    } catch (const char* e) {
        
        printf("Error: %s\n", e);
        return 1;
    }
}