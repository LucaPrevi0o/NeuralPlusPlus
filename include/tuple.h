#ifndef TUPLE_H
#define TUPLE_H

#include "tensor.h"

namespace std {

    namespace data {

        template<typename A>
        class tuple : public std::data::tensor<A, 1> { // Tuple class that inherits from tensor with 1 dimension

            public:

                using std::data::tensor<A, 1>::tensor; // Import constructors from the base class
                using std::data::tensor<A, 1>::operator(); // Import element access operator from the base class
                using std::data::tensor<A, 1>::operator+; // Import sum operator from the base class
                using std::data::tensor<A, 1>::operator+=; // Import self-addition operator from the base class
                using std::data::tensor<A, 1>::operator-; // Import subtraction operator from the base class
                using std::data::tensor<A, 1>::operator-=; // Import self-subtraction operator from the base class
                using std::data::tensor<A, 1>::operator*; // Import multiplication operator from the base class
                using std::data::tensor<A, 1>::operator*=; // Import self-multiplication operator from the base class
                using std::data::tensor<A, 1>::operator==; // Import equality operator from the base class
                using std::data::tensor<A, 1>::operator!=; // Import inequality operator from the base class
                using std::data::tensor<A, 1>::operator=; // Import assignment operator from the base class

                using std::data::tensor<A, 1>::zero(); // Import zero tensor from the base class
                using std::data::tensor<A, 1>::identity(); // Import identity tensor from the base class
        };
    }
}

#endif