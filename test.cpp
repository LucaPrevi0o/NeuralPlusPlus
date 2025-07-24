#include "include/tensor.h"
//#include "include/matrix.h"
#include <stdio.h>

int main() {

    std::data::tensor<int, 2> t1(2, 3);
    std::data::tensor<int, 2> t2(3, 2);

    std::data::matrix<int> m(2, 3); // Create a matrix of size 2x3

    int data1[] = {1, 2, 3, 4, 5, 6}; // Example data for tensor t1
    int data2[] = {7, 8, 9, 10, 11, 12}; // Example data for tensor t2

    int data3[] = {21, 22, 23, 24, 25, 26}; // Example data for matrix m1

    t1 = data1; // Assign data to tensor t1
    t2 = data2; // Assign data to tensor t2

    m = data3; // Assign data to matrix m

    try {

        auto result = t1 * t2 + 5;
        for (int i = 0; i < result.size()[0]; i++) {

            for (int j = 0; j < result.size()[1]; j++) printf("%d ", result(i, j)); // Print the result of addition
            printf("\n");
        }
        printf("\n");
        std::data::matrix<int> m1(2, 3); // Create a matrix of size 2x3
        m1 = result; // Assign matrix m to m1
        printf(std::data::tr(m1) == 159 + 63 ? "Trace is correct\n" : "Trace is incorrect\n"); // Check if the trace of m1 is correct
        printf("%d\n", std::data::det(m1)); // Check if the determinant of m1 is correct
        result = t2 * t1 + 10; // Perform matrix multiplication and addition
        for (int i = 0; i < result.size()[0]; i++) {

            for (int j = 0; j < result.size()[1]; j++) printf("%d ", result(i, j)); // Print the result of addition
            printf("\n");
        }
        printf("\n");
        result = t1 + m; // Perform matrix multiplication and addition
        for (int i = 0; i < result.size()[0]; i++) {

            for (int j = 0; j < result.size()[1]; j++) printf("%d ", result(i, j)); // Print the result of addition
            printf("\n");
        }

        auto new_m = 2 + m + 5; // Add a scalar to the matrix
        printf("\nNew matrix after adding scalar:\n");
        for (int i = 0; i < new_m.size()[0]; i++) {

            for (int j = 0; j < new_m.size()[1]; j++) printf("%d ", new_m(i, j)); // Print the new matrix after adding scalar
            printf("\n");
        }

        std::data::matrix<int> m2(3, 2); // Create another matrix of size 3x2
        m2 = std::data::matrix<int>::zero(5, 8); // Initialize m2 with zeros
        printf("\nZero matrix:\n");
        for (int i = 0; i < m2.size()[0]; i++) {

            for (int j = 0; j < m2.size()[1]; j++) printf("%d ", m2(i, j)); // Print the zero matrix
            printf("\n");
        }
    } catch (const char* e) { printf("Error: %s\n", e); }

    return 0;
}