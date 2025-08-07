#ifndef FUNCTION_H
#define FUNCTION_H

#include "tensor.h"

namespace neural {

    /**
     * @brief Abstract class for activation functions.
     * 
     * This class defines the interface for activation and loss functions used in the neural network.
     * It provides methods for computing the function value and its derivative, for both scalar and matrix inputs.
     */
    class activation {

        public:

            /**
             * @brief Compute the function value for a scalar input.
             * 
             * @param x Input value
             * @return Function value
             */
            virtual float f(float x) const = 0;

            /**
             * @brief Compute the derivative of the function for a scalar input.
             * 
             * @param x Input value
             * @return Derivative value
             */
            virtual float df(float x) const = 0; // Default implementation returns 1

            virtual activation *clone() const = 0;

            /**
             * @brief Destructor for the Function class.
             * 
             * This destructor is virtual to allow derived classes to clean up their resources.
             */
            virtual ~activation() = default;
    };

    /**
     * @brief Abstract class for loss functions.
     * 
     * This class defines the interface for loss functions used in the neural network.
     * Loss functions are functions that measure the difference between the predicted output and the true output.
     * It provides methods for computing the function value and its derivative, for both scalar and matrix inputs.
     */
    class loss {
        
        public:

            /**
             * @brief Compute the function value for two scalar inputs.
             * 
             * @param x First input value
             * @param y Second input value
             * @return Function value
             */
            virtual float f(float x, float y) const = 0;

            /**
             * @brief Compute the derivative of the function for two scalar inputs.
             * 
             * @param x First input value
             * @param y Second input value
             * @return Derivative value
             */
            virtual float df(float x, float y) const = 0;

            virtual loss *clone() const = 0;

            /**
             * @brief Destructor for the ParametricFunction class.
             * 
             * This destructor is virtual to allow derived classes to clean up their resources.
             */
            virtual ~loss() = default;
    };
}

#endif