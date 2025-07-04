#include "include/tensor.h"
#include <stdio.h>

int main() {

    std::data::tensor<int, 2> t1(2, 3);
    std::data::tensor<int, 2> t2(3, 2);

    int data1[] = {1, 2, 3, 4, 5, 6}; // Example data for tensor t1
    int data2[] = {7, 8, 9, 10, 11, 12}; // Example data for tensor t2

    t1 = data1; // Assign data to tensor t1
    t2 = data2; // Assign data to tensor t2

    try {

        auto result = t1 * t2 + 5;
        for (int i = 0; i < result.size()[0]; i++) {

            for (int j = 0; j < result.size()[1]; j++) printf("%d ", result(i, j)); // Print the result of addition
            printf("\n");
        }
        printf("\n");
        result = t2 * t1 + 10; // Perform matrix multiplication and addition
        for (int i = 0; i < result.size()[0]; i++) {

            for (int j = 0; j < result.size()[1]; j++) printf("%d ", result(i, j)); // Print the result of addition
            printf("\n");
        }
        printf("\n");
        auto new_t = std::data::tensor<int, 3>::zero(3, 2, 3); // Create an identity tensor of size 3x3
        for (int i = 0; i < new_t.size()[2]; i++) {

            for (int j = 0; j < new_t.size()[1]; j++) {
                
                for (int k = 0; k < new_t.size()[0]; k++) printf("%d ", new_t(k, j, i)); // Print the identity tensor
                printf("\n");
            }
            printf("\n");
        }
        printf("\n");
    } catch (const char* e) { printf("Error: %s\n", e); }

    return 0;
}