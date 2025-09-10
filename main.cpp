#include "include/tensor.h"
#include "include/cnetwork/neural.h"
#include "include/cnetwork/dataset/csv.h"
#include <stdio.h>
#include <time.h>

using namespace neural;
using namespace tensor;

int main() {

    srand(static_cast<unsigned int>(time(0)));

    try {

        printf("=== Example 1: Single Sample Training ===\n");
        
        int num_epochs = 500;

        // Create a neural network with batch size 1
        network net(1,
            network::shape(3, new SIGMOID()), // Input layer with 3 features
            network::shape(4, new SIGMOID()), // Hidden layer with 4 neurons
            network::shape(2, new SIGMOID())  // Output layer with 2 neurons
        );

        // Create input matrix: 3 features x 1 sample
        matrix<float> input(net[0].neurons.size(0), 1);
        printf("Input: [ ");
        for (auto i = 0; i < input.size(0); i++) {

            input(i, 0) = static_cast<float>(rand()) / RAND_MAX;
            printf("%.2f ", input(i, 0));
        }
        printf("]\n");

        // Create target matrix: 2 outputs x 1 sample
        matrix<float> target(net[net.depth() - 1].neurons.size(0), 1);
        printf("Target: [ ");
        for (auto i = 0; i < target.size(0); i++) {

            target(i, 0) = static_cast<float>(rand()) / RAND_MAX;
            printf("%.2f ", target(i, 0));
        }
        printf("]\n");

        // Train the network
        auto trained_net = train(net, input, target, new MAE(), num_epochs, 0.01f);
        auto output = trained_net.forward(input);

        printf("Output: [ ");
        for (auto i = 0; i < output[output.depth() - 1].neurons.size(0); i++) printf("%.3f ", output[output.depth() - 1].neurons(i, 0));
        printf("]\n\n");

        printf("=== Example 2: Batch Training ===\n");
        
        // Create a network for batch processing
        int batch_samples = 4;
        network batch_net(batch_samples,
            network::shape(2, new SIGMOID()), // Input layer with 2 features
            network::shape(3, new SIGMOID()), // Hidden layer with 3 neurons
            network::shape(1, new SIGMOID())  // Output layer with 1 neuron
        );

        // Create batch input: 2 features x 4 samples
        matrix<float> batch_input(batch_net[0].neurons.size(0), batch_samples);
        printf("Batch Input:\n");
        for (auto sample = 0; sample < batch_samples; sample++) {
            
            printf("Sample %d: [ ", sample + 1);
            for (auto feature = 0; feature < batch_input.size(0); feature++) {

                batch_input(feature, sample) = static_cast<float>(rand()) / RAND_MAX;
                printf("%.2f ", batch_input(feature, sample));
            }
            printf("]\n");
        }

        // Create batch target: 1 output x 4 samples
        matrix<float> batch_target(batch_net[batch_net.depth() - 1].neurons.size(0), batch_samples);
        printf("Batch Target: [ ");
        for (auto sample = 0; sample < batch_samples; sample++) {

            // Simple target: sum of inputs > 1.0 ? 1 : 0
            batch_target(0, sample) = (batch_input(0, sample) + batch_input(1, sample) > 1.0f) ? 1.0f : 0.0f;
            printf("%.1f ", batch_target(0, sample));
        }
        printf("]\n");

        // Train the batch network
        auto trained_batch_net = train(batch_net, batch_input, batch_target, new MSE(), 1000, 0.1f);
        auto batch_output = trained_batch_net.forward(batch_input);

        printf("Batch Output: [ ");
        for (auto sample = 0; sample < batch_samples; sample++) {
            printf("%.3f ", batch_output[batch_output.depth() - 1].neurons(0, sample));
        }
        printf("]\n\n");

        printf("=== Example 3: CSV Data Loading ===\n");
        
        // Example of how to use CSV loader (commented out since no CSV file provided)
        
        auto result = csv::load_split("Dataset.csv", 1); // Last column as target
        auto features = result.features;
        auto targets = result.targets;

        if (features.size(0) > 0 && features.size(1) > 0) {

            printf("Loaded %d features and %d samples from CSV\n", features.size(0), features.size(1));
            network csv_net(features.size(1), // batch size = number of samples
                network::shape(features.size(0), new SIGMOID()), // input size = number of features
                network::shape(10,               new SIGMOID()), // hidden layer with 10 neurons
                network::shape(8,                new SIGMOID()), // hidden layer with 8 neurons
                network::shape(targets.size(0),  new SIGMOID())  // output size = number of targets
            );
            
            auto csv_trained = train(csv_net, features, targets, new MSE(), 500, 0.1f);
            auto csv_output = csv_trained.forward(features);
            printf("CSV Output:\n");
            for (int i = 0; i < csv_output[csv_output.depth() - 1].neurons.size(1); i++) {

                if (i > 10) { printf("... (and more)\n"); break; }
                printf("Sample %d: [ ", i + 1);
                for (int j = 0; j < csv_output[csv_output.depth() - 1].neurons.size(0); j++) printf("%.3f ", csv_output[csv_output.depth() - 1].neurons(j, i));
                printf("]\n");
            }
            printf("CSV training completed!\n");
        } else printf("No data loaded from CSV. Please ensure 'Dataset.csv' exists and is properly formatted.\n");

        printf("=== Example 4: XOR Problem (Classic Test) ===\n");
        
        // XOR problem with 4 samples
        network xor_net(4,
            network::shape(2, new SIGMOID()), // 2 inputs
            network::shape(4, new SIGMOID()), // Hidden layer with 4 neurons
            network::shape(1, new SIGMOID())  // 1 output
        );

        // XOR training data: 2 features x 4 samples
        matrix<float> xor_input(xor_net[0].neurons.size(0), 4);
        // Sample 0: [0, 0] -> 0
        xor_input(0, 0) = 0.0f; xor_input(1, 0) = 0.0f;
        // Sample 1: [0, 1] -> 1  
        xor_input(0, 1) = 0.0f; xor_input(1, 1) = 1.0f;
        // Sample 2: [1, 0] -> 1
        xor_input(0, 2) = 1.0f; xor_input(1, 2) = 0.0f;
        // Sample 3: [1, 1] -> 0
        xor_input(0, 3) = 1.0f; xor_input(1, 3) = 1.0f;

        matrix<float> xor_target(xor_net[xor_net.depth() - 1].neurons.size(0), xor_input.size(1));
        xor_target(0, 0) = 0.0f; // 0 XOR 0 = 0
        xor_target(0, 1) = 1.0f; // 0 XOR 1 = 1
        xor_target(0, 2) = 1.0f; // 1 XOR 0 = 1
        xor_target(0, 3) = 0.0f; // 1 XOR 1 = 0

        printf("XOR Training Data:\n");
        for (int i = 0; i < xor_input.size(1); i++) printf("[%.0f, %.0f] -> %.0f\n", xor_input(0, i), xor_input(1, i), xor_target(0, i));

        // Train XOR network
        auto xor_trained = train(xor_net, xor_input, xor_target, new MSE(), 2000, 0.005f);
        auto xor_output = xor_trained.forward(xor_input);

        printf("XOR Results:\n");
        for (int i = 0; i < 4; i++) printf("[%.0f, %.0f] -> %.3f (target: %.0f)\n", 
            xor_input(0, i), xor_input(1, i), 
            xor_output[xor_output.depth() - 1].neurons(0, i), 
            xor_target(0, i));

        return 0;

    } catch (const char* e) {

        printf("Error: %s\n", e);
        return 1;
    }
}