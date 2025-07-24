#ifndef FUNCTION_H
#define FUNCTION_H

#include "tensor.h"

namespace std {

    /**
     * @brief Abstract class for activation functions.
     * 
     * This class defines the interface for activation and loss functions used in the neural network.
     * It provides methods for computing the function value and its derivative, for both scalar and matrix inputs.
     */
    class Function {

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
             * This method applies the function to each element of the input matrix.
             * 
             * @param x Input matrix
             * @return Function value matrix
             */
            std::matrix<float> f(std::matrix<float> x) const {

                std::matrix<float> result(x.size()[0], x.size()[1]);
                for (int i = 0; i < x.size()[0]; i++) 
                    for (int j = 0; j < x.size()[1]; j++) result(i, j) = f(x(i, j));
                return result;
            }

            /**
             * @brief Compute the derivative of the function for a scalar input.
             * 
             * @param x Input value
             * @return Derivative value
             */
            virtual float df(float x) const = 0; // Default implementation returns 1

            /**
             * @brief Compute the derivative of the function for a matrix input.
             * 
             * This method applies the derivative function to each element of the input matrix.
             * 
             * @param x Input matrix
             * @return Derivative value matrix
             */
            std::matrix<float> df(std::matrix<float> x) const {

                std::matrix<float> result(x.size()[0], x.size()[1]);
                for (int i = 0; i < x.size()[0]; i++) 
                    for (int j = 0; j < x.size()[1]; j++) result(i, j) = df(x(i, j));
                return result;
            }

            virtual Function* clone() const = 0; // Clone method to create a copy of the function

            /**
             * @brief Destructor for the Function class.
             * 
             * This destructor is virtual to allow derived classes to clean up their resources.
             */
            virtual ~Function() = default;
    };

    /**
     * @brief Abstract class for parametric functions.
     * 
     * This class defines the interface for parametric functions used in the neural network.
     * Parametric functions are functions that depend on an additional parameter.
     * It provides methods for computing the function value and its derivative, for both scalar and matrix inputs.
     */
    class ParametricFunction {
        
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
             * This method applies the function to each element of the input matrices.
             * 
             * @param x First input matrix
             * @param y Second input matrix
             * @return Function value matrix
             */
            std::matrix<float> f(std::matrix<float> x, std::matrix<float> y) const {

                std::matrix<float> result(x.size()[0], x.size()[1]);
                for (int i = 0; i < x.size()[0]; i++) 
                    for (int j = 0; j < x.size()[1]; j++) result(i, j) = f(x(i, j), y(i, j));
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
             * This method applies the derivative function to each element of the input matrices.
             * 
             * @param x First input matrix
             * @param y Second input matrix
             * @return Derivative value matrix
             */
            std::matrix<float> df(std::matrix<float> x, std::matrix<float> y) const {

                std::matrix<float> result(x.size()[0], x.size()[1]);
                for (int i = 0; i < x.size()[0]; i++) 
                    for (int j = 0; j < x.size()[1]; j++) result(i, j) = df(x(i, j), y(i, j));
                return result;
            }

            virtual ParametricFunction* clone() const = 0; // Clone method to create a copy of the function

            /**
             * @brief Destructor for the ParametricFunction class.
             * 
             * This destructor is virtual to allow derived classes to clean up their resources.
             */
            virtual ~ParametricFunction() = default;
    };
}

#endif