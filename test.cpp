#include "include/tensor.h"
#include "include/cnetwork/neural.h"
#include <stdio.h>

using namespace std;
using namespace neural;

int main() {

    // Create a 2D tensor (matrix) with dimensions 3x3
    matrix<int> mat(9, 9);

    // Fill the matrix with some values
    for (int i = 0; i < mat.size()[0]; i++)
        for (int j = 0; j < mat.size()[1]; j++) mat(i, j) = i  + j; // Fill with sequential values

    // Print the matrix
    for (int i = 0; i < mat.size()[0]; i++) {

        for (int j = 0; j < mat.size()[1]; j++) printf("%c%d ", (mat(i, j) < 10 ? ' ' : 0), mat(i, j));
        printf("\n");
    }

    network n = network(
        network::layer(3, 1, new SIGMOID()), // Input layer with 3 neurons
        network::layer(5, 1, new SIGMOID()), // Hidden layer with 5 neurons
        network::layer(2, 1, new SIGMOID())  // Output layer with 2 neurons
    );

    // Forward pass through the network with a sample input
    matrix<float> input(3, 1);
    for (int i = 0; i < input.size()[0]; i++) input(i, 0) = i + 1; // Sample input

    auto output = n.forward(input);

    // Print the output of the network
    for (int i = 0; i < output.size(); i++) {
        
        auto layer = n[i];
        printf("Layer %d:\n", i);
        for (int j = 0; j < layer.neurons.size()[0]; j++) {
            for (int k = 0; k < layer.neurons.size()[1]; k++) {
                printf("%c%.2f ", (layer.neurons(j, k) < 10 ? ' ' : 0), layer.neurons(j, k));
            }
            printf("\n");
        }
        printf("\n");
    }

    return 0;
}