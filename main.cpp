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

        printf("=== Example 3: CSV Data Loading ===\n");
        
        // Example of how to use CSV loader (commented out since no CSV file provided)
        
        auto result = csv::load_split("Dataset.csv", 1); // Last column as target
        auto features = result.features; // 8 x 4000+
        auto targets = result.targets;   // 1 x 4000+

        auto batch_size = 64;
        auto train_samples = 2048;
        auto validation_samples = 256;

        auto input_features = matrix<float>(features.size(0), train_samples); // 8 x 1500
        auto validation_features = matrix<float>(features.size(0), validation_samples);

        for (int i = 0; i < input_features.size(0); i++) 
            for (int j = 0; j < input_features.size(1); j++) {
                input_features(i, j) = features(i, j);
            }

        for (int i = 0; i < validation_features.size(0); i++) 
            for (int j = 0; j < validation_features.size(1); j++) {
                validation_features(i, j) = features(i, j + train_samples);
            }

        if (input_features.size(0) > 0 && input_features.size(1) > 0) {

            activation *sigmoid = new SIGMOID();
            network csv_net(batch_size, // batch size = number of samples
                network::shape(input_features.size(0), sigmoid), // input size = number of features
                network::shape(10,                     sigmoid), // hidden layer with 10 neurons
                network::shape(8,                      sigmoid), // hidden layer with 8 neurons
                network::shape(targets.size(0),        sigmoid)  // output size = number of targets
            );
            
            auto csv_trained = train(csv_net, input_features, targets, new MSE(), 35000, 0.001f, 0.001f);
            auto csv_output = validate(csv_trained, validation_features, targets);
            printf("CSV Output error:\n");
            for (int k = 0; k < csv_output.size(2); k++)
                for (int i = 0; i < csv_output.size(1); i++) {

                    printf("[ ");
                    for (int j = 0; j < csv_output.size(0); j++) printf("%c%.3f ", csv_output(j, i, k) >= 0 ? ' ' : 0, csv_output(j, i, k));
                    printf("]\n");
                }
            printf("CSV training completed!\n");
            delete sigmoid;
        } else printf("No data loaded from CSV. Please ensure 'Dataset.csv' exists and is properly formatted.\n");
        return 0;

    } catch (const char* e) {

        printf("Error: %s\n", e);
        return 1;
    }
}