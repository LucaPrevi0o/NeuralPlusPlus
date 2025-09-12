#ifndef FUNCTION_H
#define FUNCTION_H

#include "tensor/tensor.h"

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
             * @brief Compute the function value for a matrix input.
             * 
             * @param x Input matrix
             * @return Matrix of function values
             */
            tensor::matrix<float> f(const tensor::matrix<float>& x) const {

                auto result = tensor::matrix<float>(x.size(0), x.size(1));
                for (int i = 0; i < x.size(0); i++)
                    for (int j = 0; j < x.size(1); j++) result(i, j) = f(x(i, j));
                return result;
            }

            /**
             * @brief Compute the derivative of the function for a scalar input.
             * 
             * @param x Input value
             * @return Derivative value
             */
            virtual float df(float x) const = 0;

            /**
             * @brief Compute the derivative of the function for a matrix input.
             * 
             * @param x Input matrix
             * @return Matrix of derivative values
             */
            tensor::matrix<float> df(const tensor::matrix<float>& x) const {

                auto result = tensor::matrix<float>(x.size(0), x.size(1));
                for (int i = 0; i < x.size(0); i++)
                    for (int j = 0; j < x.size(1); j++) result(i, j) = df(x(i, j));
                return result;
            }

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
             * @brief Compute the function value for two matrix inputs.
             * 
             * @param x First input matrix
             * @param y Second input matrix
             * @return Matrix of function values
             */
            tensor::matrix<float> f(const tensor::matrix<float>& x, const tensor::matrix<float>& y) const {

                auto result = tensor::matrix<float>(x.size(0), x.size(1));
                for (int i = 0; i < x.size(0); i++)
                    for (int j = 0; j < x.size(1); j++) result(i, j) = f(x(i, j), y(i, j));
                return result;
            }

            /**
             * @brief Compute the derivative of the function for two scalar inputs.
             * 
             * @param x First input value
             * @param y Second input value
             * @return Derivative value
             */
            virtual float df(float x, float y) const = 0;

            /**
             * @brief Compute the derivative of the function for two matrix inputs.
             * 
             * @param x First input matrix
             * @param y Second input matrix
             * @return Matrix of derivative values
             */
            tensor::matrix<float> df(const tensor::matrix<float>& x, const tensor::matrix<float>& y) const {

                auto result = tensor::matrix<float>(x.size(0), x.size(1));
                for (int i = 0; i < x.size(0); i++)
                    for (int j = 0; j < x.size(1); j++) result(i, j) = df(x(i, j), y(i, j));
                return result;
            }

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