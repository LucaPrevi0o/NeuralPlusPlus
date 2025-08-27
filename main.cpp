#include "include/tensor.h"
#include "include/cnetwork/neural.h"
#include <stdio.h>

using namespace neural;

int main() {

    try {

        // Create a neural network with 3 layers and a batch size of 3
        network net(
            network::shape(3, new SIGMOID()), // First layer with 3 neurons and sigmoid activation
            network::shape(4, new SIGMOID()), // Second layer with 4 neurons and sigmoid activation
            network::shape(2, new SIGMOID())  // Output layer with 2 neurons and sigmoid activation
        );

        // Create an input matrix with 3 features and 1 sample
        std::matrix<float> input(3, 1);
        input(0, 0) = 0.5f; // Feature 1 sample 1
        input(1, 0) = 0.2f; // Feature 2 sample 1
        input(2, 0) = 0.8f; // Feature 3 sample 1

        // Create a target matrix for training
        std::matrix<float> target(2, 1);
        target(0, 0) = 0.7f; // Target output for sample 1
        target(1, 0) = 0.3f; // Target output for sample 1

        for (int i = 0; i < 200; i++) {

            auto output = net.forward(input);

            printf("Epoch %d: [ ", i + 1);
            for (auto sample = 0; sample < output[output.depth() - 1].neurons.size(1); sample++)
                for (auto neuron = 0; neuron < output[output.depth() - 1].neurons.size(0); neuron++)
                    printf("%f ", output[output.depth() - 1].neurons(neuron, sample));
            printf("]\n");

            net = output.backpropagate(new MSE(), 0.1f, target);
        }

        return 0;

    } catch (const char* e) {
        
        printf("Error: %s\n", e);
        return 1;
    }
}