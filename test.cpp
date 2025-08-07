#include "include/tensor.h"
#include "include/neural.h"
#include <stdio.h>

using namespace std;

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

    neural::neural n;
    return 0;
}