#ifndef TENSOR_H
#define TENSOR_H

namespace std {

    namespace data {

        /**
         * @brief Tensor class for multi-dimensional arrays.
         * 
         * This class represents a tensor, which is a multi-dimensional array.
         * It provides methods for accessing and manipulating the tensor data.
         * * @tparam T Type of the tensor elements
         * * @tparam N Number of dimensions of the tensor
         */
        template<typename T, int N>
        class tensor {

            // Implementation of the tensor class

            protected:
                T* data;
        };

    } // namespace data
}

#endif