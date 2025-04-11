#include "include/neural.h"
#include <stdio.h>

using namespace std::neural;

int main(int argc, char **argv) {

    try {

        network test(
            std::neural::network::layer(3, new std::neural::activation::Sigmoid()),
            std::neural::network::layer(4, new std::neural::activation::Sigmoid()),
            std::neural::network::layer(2, new std::neural::activation::Sigmoid()),
            std::neural::network::layer(5, new std::neural::activation::Sigmoid()),
            std::neural::network::layer(3, new std::neural::activation::Sigmoid())
        );
    
        for (int i = 0; i < test.size() - 1; i++) {
    
            printf("%d - %d\n", test[i].size()[0], test[i].size()[1]);
            for (int j = 0; j < test[i].size()[0]; j++) {
    
                for (int k = 0; k < test[i].size()[1]; k++) {
                    
                    //if (k == test[i].size()[0] - 1) printf("| ");
                    printf("%s%f ", (test[i](j, k)<0 ? "" : " "), test[i](j, k));
                }
                printf("\n");
            }
            printf("\n");
        }

        network n2 = test.compute(0.4, 0.3, 0.6);
        for (int i = 0; i < n2.size(); i++) {
    
            printf("%d - %d\n", n2(i).size()[0], n2(i).size()[1]);
            for (int j = 0; j < n2(i).size()[0]; j++) {
    
                for (int k = 0; k < n2(i).size()[1]; k++) printf("%s%f ", (n2(i)(j, k)<0 ? "" : " "), n2(i)(j, k));
                printf("\n");
            }
            printf("\n");
        }

        network n3 = n2.backpropagate(0.005, new std::neural::loss::Mse(), 0.1, 0.1, 0.1);
        for (int i = 0; i < n3.size(); i++) {
    
            printf("%d - %d\n", n3(i).size()[0], n3(i).size()[1]);
            for (int j = 0; j < n3(i).size()[1]; j++) {
    
                for (int k = 0; k < n3(i).size()[0]; k++) printf("%s%f ", (n3(i)(k, j)<0 ? "" : " "), n3(i)(k, j));
                printf("\n");
            }
            printf("\n");
        }
    } catch (const char *e) { printf("Error: %s\n", e); }

    return 0;
}