#include "include/tensor.h"
#include <stdio.h>

using namespace std;

int main() {

    // Create a 2D tensor (matrix) with dimensions 3x3
    matrix<int> mat(9, 7);

    // Fill the matrix with some values
    for (int i = 0; i < mat.size()[0]; i++)
        for (int j = 0; j < mat.size()[1]; j++) mat(i, j) = i * mat.size()[1] + j; // Fill with sequential values

    // Print the matrix
    for (int i = 0; i < mat.size()[0]; i++) {

        for (int j = 0; j < mat.size()[1]; j++) printf("%c%d ", (mat(i, j) < 10 ? ' ' : 0), mat(i, j));
        printf("\n");
    }
    return 0;
}