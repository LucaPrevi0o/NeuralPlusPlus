#ifndef TUPLE_H
#define TUPLE_H

#include "tensor.h"

namespace std {

    namespace data {

        template<typename A>
        class tuple : public std::data::tensor<A, 1> { // Tuple class that inherits from tensor with 1 dimension

            public:

                /**
                 * @brief Construct a new tuple object with a variable number of arguments.
                 * 
                 * @param args Arguments to initialize the tuple with
                 */
                template<typename... Args>
                tuple(Args... args) : std::data::tensor<A, 1>(sizeof...(args)) { // Call parent constructor in initialization list
            
                    A temp[] = {args...}; // Initialize the data array with the arguments
                    for (int i = 0; i < this->capacity[0]; i++) this->data[i] = temp[i]; // Copy the data into the tensor
                }
        };
    }
}

#endif