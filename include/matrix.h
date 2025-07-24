#ifndef MATRIX_H
#define MATRIX_H

#include "tensor.h"

namespace std {

    namespace data {

        /**
         * @brief Matrix class for 2-dimensional tensors.
         * 
         * This class represents a matrix, which is a 2-dimensional tensor.
         * It provides methods for accessing and manipulating the matrix data.
         * 
         * @tparam T Type of the matrix elements
         */
        template<typename A>
        class matrix : public std::data::tensor<A, 2> {

            public:

                using std::data::tensor<A, 2>::tensor; // Import constructors from the base class
                using std::data::tensor<A, 2>::operator(); // Import element access operator from the base class
                using std::data::tensor<A, 2>::operator+; // Import sum operator from the base class
                using std::data::tensor<A, 2>::operator+=; // Import self-addition operator from the base class
                using std::data::tensor<A, 2>::operator-; // Import subtraction operator from the base class
                using std::data::tensor<A, 2>::operator-=; // Import self-subtraction operator from the base class
                using std::data::tensor<A, 2>::operator*; // Import multiplication operator from the base class
                using std::data::tensor<A, 2>::operator*=; // Import self-multiplication operator from the base class
                using std::data::tensor<A, 2>::operator==; // Import equality operator from the base class
                using std::data::tensor<A, 2>::operator!=; // Import inequality operator from the base class
                using std::data::tensor<A, 2>::operator=; // Import assignment operator from the base class

                using std::data::tensor<A, 2>::zero; // Import zero tensor from the base class
                using std::data::tensor<A, 2>::identity; // Import identity tensor from the base class
        };

        /**
         * @brief Calculate the transpose of a matrix.
         * 
         * @tparam T Type of the matrix elements
         * @param mat Matrix to calculate the transpose of
         * @return Transpose of the matrix
         */
        template<typename A>
        tensor<A, 2> T(tensor<A, 2> mat) {

            tensor<A, 2> result(mat.size()[1], mat.size()[0]);
            for (int i = 0; i < mat.size()[0]; i++) 
                for (int j = 0; j < mat.size()[1]; j++) result(j, i) = mat(i, j);
            return result;
        }

        /**
         * @brief Calculate the trace of a matrix.
         * 
         * @tparam T Type of the matrix elements
         * @param mat Matrix to calculate the trace of
         * @return Trace of the matrix
         */
        template<typename A>
        A tr(tensor<A, 2> mat) {

            if (mat.size()[0] != mat.size()[1]) throw "Matrix is not square";
            A result = A(0);
            for (int i = 0; i < mat.size()[0]; i++) result += mat(i, i);
            return result;
        }

        /**
         * @brief Calculate the submatrix of a matrix, excluding one row and column.
         * 
         * @tparam T Type of the matrix elements
         * @param mat Matrix to calculate the submatrix of
         * @param excludingRow Row to exclude
         * @param excludingCol Column to exclude
         * @throw "Invalid row or column" if the row or column index is out of bounds
         * @return Submatrix of the matrix
         */
        template<typename A>
        tensor<A, 2> submatrix(tensor<A, 2> mat, tensor<int, 1> del_rows, tensor<int, 1> del_cols) {

            tensor<A, 2> result(mat.size()[0] - del_rows.size()[0], mat.size()[1] - del_cols.size()[0]);
            int r = -1, q = 0;
            for (int i = 0; i < mat.size()[0]; i++) {

                for (int j = 0; j < del_rows.size()[0]; j++) if (i == del_rows(j)) q = 1;
                if (q) { q = 0; continue; }
                r++;
                int c = -1;
                for (int j = 0; j < mat.size()[1]; j++) {

                    for (int k = 0; k < del_cols.size()[0]; k++) if (j == del_cols(k)) q = 1;
                    if (q) { q = 0; continue; }
                    c++;
                    result(r, c) = mat(i, j);
                }
            }
            return result;
        }

        /**
         * @brief Calculate the adjugate of a matrix.
         * 
         * @tparam T Type of the matrix elements
         * @param mat Matrix to calculate the adjugate of
         * @return Adjugate of the matrix
         */
        template<typename A>
        tensor<A, 2> adj(tensor<A, 2> mat) {

            if (mat.size()[0] != mat.size()[1]) throw "Matrix is not square";
            tensor<A, 2> result(mat.size()[0], mat.size()[1]);
            for (int i = 0; i < mat.size()[0]; i++) {
                for (int j = 0; j < mat.size()[1]; j++) {

                    tensor<int, 1> del_row(1); del_row(0) = i;
                    tensor<int, 1> del_col(1); del_col(0) = j;
                    tensor<A, 2> subm = submatrix(mat, del_row, del_col);
                    result(i, j) = det(subm) * ((i + j) % 2 == 0 ? (A)1 : (A)(-1));
                }
            }
            return T(result);
        }

        /**
         * @brief Calculate the determinant of a matrix.
         * 
         * @tparam T Type of the matrix elements
         * @param mat Matrix to calculate the determinant of
         * @return Determinant of the matrix
         */
        template<typename A>
        A det(tensor<A, 2> mat) {

            if (mat.size()[0] != mat.size()[1]) throw "Matrix is not square";
            else if (mat.size()[0] == 1) return mat(0, 0);
            else {

                A res = A(0);
                for (int p = 0; p < mat.size()[0]; p++) {
                    
                    tensor<int, 1> del_row(1); del_row(0) = 0;
                    tensor<int, 1> del_col(1); del_col(0) = p;
                    res += mat(0, p) * det(submatrix(mat, del_row, del_col)) * ((p % 2 == 0) ? A(1) : A(-1));
                }
                return res;
            }
        }
    }
}

#endif