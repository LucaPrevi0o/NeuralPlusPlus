#ifndef FUNCTION_H
#define FUNCTION_H

#include "matrix.h"

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
            virtual float f(float x) const { return x; } // Default implementation returns the input value;

            /**
             * @brief Compute the function value for a matrix input.
             * 
             * This method applies the function to each element of the input matrix.
             * 
             * @param x Input matrix
             * @return Function value matrix
             */
            std::data::matrix<float> f(std::data::matrix<float> x) const {

                std::data::tuple<int> s = x.size();
                std::data::matrix<float> result(s[0], s[1]);
                for (int i = 0; i < s[0]; i++) 
                    for (int j = 0; j < s[1]; j++) result(i, j) = f(x(i, j));
                return result;
            }

            /**
             * @brief Compute the derivative of the function for a scalar input.
             * 
             * @param x Input value
             * @return Derivative value
             */
            virtual float df(float x) const { return 1; } // Default implementation returns 1

            /**
             * @brief Compute the derivative of the function for a matrix input.
             * 
             * This method applies the derivative function to each element of the input matrix.
             * 
             * @param x Input matrix
             * @return Derivative value matrix
             */
            std::data::matrix<float> df(std::data::matrix<float> x) const {

                std::data::tuple<int> s = x.size();
                std::data::matrix<float> result(s[0], s[1]);
                for (int i = 0; i < s[0]; i++) 
                    for (int j = 0; j < s[1]; j++) result(i, j) = df(x(i, j));
                return result;
            }

            /**
             * @brief Copy constructor for the Function class.
             * 
             * @param other The Function object to copy
             */
            Function operator=(const Function &other) { return *this; } // Assignment operator

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
            virtual float f(float x, float y) const { return (x - y) * (x - y); }

            /**
             * @brief Compute the function value for two matrix inputs.
             * 
             * This method applies the function to each element of the input matrices.
             * 
             * @param x First input matrix
             * @param y Second input matrix
             * @return Function value matrix
             */
            std::data::matrix<float> f(std::data::matrix<float> x, std::data::matrix<float> y) const {

                std::data::tuple<int> s = x.size();
                std::data::matrix<float> result(s[0], s[1]);
                for (int i = 0; i < s[0]; i++) 
                    for (int j = 0; j < s[1]; j++) result(i, j) = f(x(i, j), y(i, j));
                return result;
            }

            /**
             * @brief Compute the derivative of the function for two scalar inputs.
             * 
             * @param x First input value
             * @param y Second input value
             * @return Derivative value
             */
            virtual float df(float x, float y) const { return 2 * (x - y); }

            /**
             * @brief Compute the derivative of the function for two matrix inputs.
             * 
             * This method applies the derivative function to each element of the input matrices.
             * 
             * @param x First input matrix
             * @param y Second input matrix
             * @return Derivative value matrix
             */
            std::data::matrix<float> df(std::data::matrix<float> x, std::data::matrix<float> y) const {

                std::data::tuple<int> s = x.size();
                std::data::matrix<float> result(s[0], s[1]);
                for (int i = 0; i < s[0]; i++) 
                    for (int j = 0; j < s[1]; j++) result(i, j) = df(x(i, j), y(i, j));
                return result;
            }

            /**
             * @brief Copy constructor for the ParametricFunction class.
             * 
             * @param other The ParametricFunction object to copy
             */
            ParametricFunction operator=(const ParametricFunction &other) { return *this; } // Assignment operator

            /**
             * @brief Destructor for the ParametricFunction class.
             * 
             * This destructor is virtual to allow derived classes to clean up their resources.
             */
            virtual ~ParametricFunction() = default;
    };
}

#endif