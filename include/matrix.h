#ifndef MATRIX_H
#define MATRIX_H

#include "tuple.h"
#include "tensor.h"
#include <stdio.h>

namespace std { 
    
    namespace data {

        /**
         * @brief Standard matrix class.
         * 
         * @tparam A Type of the matrix elements
         */
        template<typename A>
        class matrix: public std::data::tensor<A, 2> {

            public:

                /**
                 * @brief Construct a new matrix object.
                 */
                matrix() : std::data::tensor<A, 2>(0, 0) {}

                /**
                 * @brief Construct a new matrix object.
                 * 
                 * @param rows Number of rows
                 * @param columns Number of columns
                 */
                matrix(int rows, int columns) : std::data::tensor<A, 2>(rows, columns) {}

                /**
                 * @brief Construct a new matrix object.
                 * 
                 * @param other Matrix to copy
                 */
                matrix(const matrix<A> &other) : std::data::tensor<A, 2>(other->capacity[0], other->capacity[1]) { 
                    
                    for (int i = 0; i < other->capacity[0] * other->capacity[1]; i++) this->data[i] = other.data[i];
                }

                /**
                 * @brief Construct a new matrix object with a single element.
                 * 
                 * @param element Element to initialize the matrix with
                 */
                matrix(const A &element) : std::data::tensor<A, 2>(1, 1) { this->data[0] = element; }

                matrix<A> operator=(matrix<A> other); // Create a new matrix object and assign it the value of another matrix object

                matrix<A> row(int row); // Get a row of the matrix
                matrix<A> column(int column); // Get a column of the matrix

                matrix<A> operator|(matrix<A> other); // Concatenate two matrices

                matrix<A> operator+(matrix<A> other); // Add two matrices
                matrix<A> operator+(A scalar); // Add a scalar to a matrix (the scalar has the same type as the matrix elements)

                /**
                 * @brief Add a matrix to a scalar object. The scalar has the same type as the matrix elements.
                 * 
                 * @param scalar Scalar to add the matrix to
                 * @param mat Matrix to add
                 * @return New matrix object
                 */
                friend matrix<A> operator+(A scalar, matrix<A> mat) {

                    matrix<A> result(mat.rows, mat.columns);
                    for (int i = 0; i < mat.rows * mat.columns; i++) result.data[i] = scalar + mat.data[i];
                    return result;
                }

                matrix<A> operator+=(matrix<A> other); // Add a matrix to this matrix
                matrix<A> operator+=(A scalar); // Add a scalar to this matrix
 
                matrix<A> operator-(matrix<A> other); // Subtract two matrices
                matrix<A> operator-(A scalar); // Subtract a scalar from a matrix (the scalar has the same type as the matrix elements)

                /**
                 * @brief Subtract a matrix from a scalar object. The scalar has the same type as the matrix elements.
                 * 
                 * @param scalar Scalar to subtract the matrix from
                 * @param mat Matrix to subtract
                 * @return New matrix object
                 */
                friend matrix<A> operator-(A scalar, matrix<A> mat) {

                    matrix<A> result(mat.rows, mat.columns);
                    for (int i = 0; i < mat.rows * mat.columns; i++) result.data[i] = scalar - mat.data[i];
                    return result;
                }

                matrix<A> operator-=(matrix<A> other); // Subtract a matrix from this matrix
                matrix<A> operator-=(A scalar); // Subtract a scalar from this matrix

                matrix<A> operator*(matrix<A> other); // Multiply two matrices
                matrix<A> operator*(A scalar); // Multiply a matrix by a scalar (the scalar has the same type as the matrix elements)

                /**
                 * @brief Multiply a matrix by a scalar. The scalar has the same type as the matrix elements.
                 * 
                 * @param scalar Scalar to multiply
                 * @param mat Matrix to multiply by
                 * @return New matrix object
                 */
                friend matrix<A> operator*(A scalar, matrix<A> mat)  {

                    matrix<A> result(mat.rows, mat.columns);
                    for (int i = 0; i < mat.rows * mat.columns; i++) result.data[i] = scalar * mat.data[i];
                    return result;
                }

                matrix<A> operator*=(matrix<A> other); // Multiply this matrix by another matrix
                matrix<A> operator*=(A scalar); // Multiply this matrix by a scalar (the scalar has the same type as the matrix elements)

                matrix<A> operator/(A scalar); // Divide a matrix by a scalar (the scalar has the same type as the matrix elements)
                matrix<A> operator/=(A scalar); // Divide this matrix by a scalar (the scalar has the same type as the matrix elements)

                bool operator==(matrix<A> other); // Check if two matrices are equal
                bool operator!=(matrix<A> other); // Check if two matrices are not equal

                /**
                 * @brief Create a matrix of ones.
                 * 
                 * @param rows Number of rows
                 * @param columns Number of columns
                 * @return New matrix object
                 */
                static const matrix<A> MATRIX_ONES(int rows, int columns) {

                    matrix<A> result(rows, columns);
                    for (int i = 0; i < rows * columns; i++) result.data[i] = A(1);
                    return result;
                }

                /**
                 * @brief Create a matrix of zeros.
                 * 
                 * @param rows Number of rows
                 * @param columns Number of columns
                 * @return New @c <matrix> object
                 */
                static const matrix<A> MATRIX_NULL(int rows, int columns) {

                    matrix<A> result(rows, columns);
                    for (int i = 0; i < rows * columns; i++) result.data[i] = A(0);
                    return result;
                }

                /**
                 * @brief Create an identity matrix.
                 * The identity matrix is a square matrix with ones on the diagonal and zeros elsewhere.
                 * 
                 * @param size Size of the matrix
                 * @return New matrix object
                 */
                static const matrix<A> MATRIX_IDENTITY(int size) {

                    matrix<A> result(size, size);
                    for (int i = 0; i < size; i++)
                        for (int j = 0; j < size; j++) result.data[i * size + j] = (i == j) ? A(1) : A(0);
                    return result;
                }
        };

        // Function prototypes
        template<typename A> matrix<A> submatrix(matrix<A> mat, std::data::tuple<int> excludingRow, std::data::tuple<int> excludingCol); // Calculate the submatrix of a matrix
        template<typename A> matrix<A> adj(matrix<A> mat); // Calculate the adjugate of a matrix
        template<typename A> A tr(matrix<A> mat); // Calculate the trace of a matrix
        template<typename A> A det(matrix<A> mat); // Calculate the determinant of a matrix
        template<typename A> matrix<A> T(matrix<A> mat); // Calculate the transpose of a matrix
    }
}

/**
 * @brief Get a row of the matrix.
 * 
 * @param row Row index
 * @throw "Invalid row" if the row index is out of bounds
 * @return New matrix object
 */
template<typename A>
std::data::matrix<A> std::data::matrix<A>::row(int row) {

    if (row < 0 || row >= this->capacity[0]) throw "Invalid row";
    std::data::matrix<A> result(1, this->capacity[1]);
    for (int i = 0; i < this->capacity[1]; i++) result[i] = this->data[row * this->capacity[1] + i];
    return result;
}

/**
 * @brief Get a column of the matrix.
 * 
 * @param column Column index
 * @throw "Invalid column" if the column index is out of bounds
 * @return New matrix object
 */
template<typename A>
std::data::matrix<A> std::data::matrix<A>::column(int column) {

    if (column < 0 || column >= this->capacity[1]) throw "Invalid column";
    std::data::matrix<A> result(this->capacity[0], 1);
    for (int i = 0; i < this->capacity[0]; i++) result.data[i] = this->data[i * this->capacity[1] + column];
    return result;
}

/**
 * @brief Concatenate two matrices.
 * 
 * @param other Second matrix to concatenate
 * @throw "Invalid matrix sizes" if the number of rows of the first matrix is not equal to the number of rows of the second matrix
 * @return New matrix object
 */
template<typename A>
std::data::matrix<A> std::data::matrix<A>::operator|(std::data::matrix<A> other) {

    if (this->capacity[0] == other->capacity[0]) { // Concatenate horizontally

        std::data::matrix<A> result(this->capacity[0], this->capacity[1] + other->capacity[1]);
        for (int i = 0; i < this->capacity[0]; i++) {

            for (int j = 0; j < this->capacity[1]; j++) result(i, j) = this->data[i * this->capacity[1] + j];
            for (int j = 0; j < other->capacity[1]; j++) result(i, j + this->capacity[1]) = other(i, j);
        }
        return result;
    } else if (this->capacity[1] == other->capacity[1]) { // Concatenate vertically
        
        std::data::matrix<A> result(this->capacity[0] + other->capacity[0], this->capacity[1]);
        
        for (int i = 0; i < this->capacity[1]; i++) {

            for (int j = 0; j < this->capacity[0]; j++) result(j, i) = this->data[j * this->capacity[1] + i];
            for (int j = 0; j < other->capacity[0]; j++) result(j + this->capacity[0], i) = other.data[j * other->capacity[1] + i];
        }
        return result;
    } else throw "Invalid matrix sizes"; // Invalid matrix sizes
}

/**
 * @brief Add two matrices.
 * 
 * @param other Second matrix to add
 * @throw "Matrices have different sizes" if the matrices have different sizes
 * @return New matrix object
 */
template<typename A>
std::data::matrix<A> std::data::matrix<A>::operator+(std::data::matrix<A> other) {

    if (this->capacity[0] != other->capacity[0] || this->capacity[1] != other->capacity[1]) throw "Matrices have different sizes";
    std::data::matrix<A> result(this->capacity[0], this->capacity[1]);
    for (int i = 0; i < this->capacity[0] * this->capacity[1]; i++) result.data[i] = this->data[i] + other.data[i];
    return result;
}
/**
 * @brief Add a scalar to a matrix. The scalar has the same type as the matrix elements.
 * 
 * @param scalar Scalar to add
 * @return New matrix object
 */
template<typename A>
std::data::matrix<A> std::data::matrix<A>::operator+(A scalar) {

    std::data::matrix<A> result(this->capacity[0], this->capacity[1]);
    for (int i = 0; i < this->capacity[0] * this->capacity[1]; i++) result.data[i] = this->data[i] + scalar;
    return result;
}

/**
 * @brief Add a matrix to this matrix.
 * 
 * @param other Matrix to add
 * @throw "Matrices have different sizes" if the matrices have different sizes
 * @return New matrix object
 */
template<typename A>
std::data::matrix<A> std::data::matrix<A>::operator+=(std::data::matrix<A> other) {

    if (this->capacity[0] != other->capacity[0] || this->capacity[1] != other->capacity[1]) throw "Matrices have different sizes";
    for (int i = 0; i < this->capacity[0] * this->capacity[1]; i++) this->data[i] += other.data[i];
    return *this;
}

/**
 * @brief Add a scalar to this matrix.
 * 
 * @param scalar Scalar to add
 * @return New matrix object
 */
template <typename A>
std::data::matrix<A> std::data::matrix<A>::operator+=(A scalar) {

    for (int i = 0; i < this->capacity[0] * this->capacity[1]; i++) this->data[i] += scalar;
    return *this;
}

/**
 * @brief Subtract two matrices.
 * 
 * @param other Matrix to subtract
 * @throw "Matrices have different sizes" if the matrices have different sizes
 * @return New matrix object
 */
template<typename A>
std::data::matrix<A> std::data::matrix<A>::operator-(std::data::matrix<A> other) {

    if (this->capacity[0] != other.rows || this->capacity[1] != other.columns) throw "Matrices have different sizes";
    std::data::matrix<A> result(this->capacity[0], this->capacity[1]);
    for (int i = 0; i < this->capacity[0] * this->capacity[1]; i++) result.data[i] = this->data[i] - other.data[i];
    return result;
}

/**
 * @brief Subtract a scalar from a matrix. The scalar has the same type as the matrix elements.
 * 
 * @param scalar Scalar to subtract
 * @return New matrix object
 */
template<typename A>
std::data::matrix<A> std::data::matrix<A>::operator-(A scalar) {

    std::data::matrix<A> result(this->capacity[0], this->capacity[1]);
    for (int i = 0; i < this->capacity[0] * this->capacity[1]; i++) result.data[i] = this->data[i] - scalar;
    return result;
}

/**
 * @brief Subtract a matrix from this matrix.
 * 
 * @param other Matrix to subtract
 * @throw "Matrices have different sizes" if the matrices have different sizes
 * @return New matrix object
 */
template <typename A>
std::data::matrix<A> std::data::matrix<A>::operator-=(std::data::matrix<A> other) {

    printf("(%d %d) - ", rows, columns);
    printf("(%d %d) -> ", other.rows, other.columns);
    if (rows != other.rows || columns != other.columns) throw "Matrices have different sizes -=";
    printf("res.size = (%d %d)\n", rows, columns);
    for (int i = 0; i < rows * columns; i++) this->data[i] -= other.data[i];
    return *this;
}

/**
 * @brief Subtract a scalar from this matrix.
 * 
 * @param scalar Scalar to subtract
 * @return New matrix object
 */
template <typename A>
std::data::matrix<A> std::data::matrix<A>::operator-=(A scalar) {

    for (int i = 0; i < rows * columns; i++) this->data[i] -= scalar;
    return *this;
}

/**
 * @brief Multiply two matrices.
 * 
 * @param other Matrix to multiply by
 * @throw "Invalid matrix sizes" if the number of columns of the first matrix is not equal to the number of rows of the second matrix
 * @return New matrix object
 */
template<typename A>
std::data::matrix<A> std::data::matrix<A>::operator*(std::data::matrix<A> other) {

    printf("(%d %d) * ", rows, columns);
    printf("(%d %d) -> ", other.rows, other.columns);
    if (columns != other.rows) throw "Invalid matrix sizes";
    printf("res.size = (%d %d)\n", rows, other.columns);
    std::data::matrix<A> result = MATRIX_NULL(rows, other.columns);
    for (int i = 0; i < rows; i++) 
        for (int j = 0; j < other.columns; j++) 
            for (int k = 0; k < columns; k++) result.data[i * result.columns + j] += this->data[i * columns + k] * other(k, j);
    return result;
}

/**
 * @brief Multiply a matrix by a scalar.
 * 
 * @param scalar Scalar to multiply by
 * @return New matrix object
 */
template<typename A>
std::data::matrix<A> std::data::matrix<A>::operator*(A scalar) {

    std::data::matrix<A> result(rows, columns);
    for (int i = 0; i < rows * columns; i++) result.data[i] = this->data[i] * scalar;
    return result;
}

/**
 * @brief Multiply this matrix by another matrix.
 * 
 * @param other Matrix to multiply by
 * @return New matrix object
 */
template <typename A>
std::data::matrix<A> std::data::matrix<A>::operator*=(std::data::matrix<A> other) {

    *this = *this * other;
    return *this;
}

/**
 * @brief Multiply this matrix by a scalar. The scalar has the same type as the matrix elements.
 * 
 * @param scalar Scalar to multiply by
 * @return New matrix object
 */
template <typename A>
std::data::matrix<A> std::data::matrix<A>::operator*=(A scalar) {

    for (int i = 0; i < rows * columns; i++) this->data[i] *= scalar;
    return *this;
}

/**
 * @brief Divide a matrix by a scalar. The scalar has the same type as the matrix elements.
 * 
 * @param scalar Scalar to divide by
 * @throw "Division by zero" if the scalar is zero
 * @return New matrix object
 */
template<typename A>
std::data::matrix<A> std::data::matrix<A>::operator/(A scalar) {
    
    if (scalar == 0) throw "Division by zero";
    std::data::matrix<A> result(rows, columns);
    for (int i = 0; i < rows * columns; i++) result.data[i] = this->data[i] / scalar;
    return result;
}

/**
 * @brief Divide this matrix by a scalar. The scalar has the same type as the matrix elements.
 * 
 * @param scalar Scalar to divide by
 * @throw "Division by zero" if the scalar is zero
 * @return New matrix object
 */
template<typename A>
std::data::matrix<A> std::data::matrix<A>::operator/=(A scalar) {

    if (scalar == 0) throw "Division by zero";
    for (int i = 0; i < rows * columns; i++) this->data[i] /= scalar;
    return *this;
}

/**
 * @brief Check if two matrices are equal.
 * 
 * @param other Matrix to compare
 * @return @c true if the matrices are equal, @c false otherwise
 */
template<typename A>
bool std::data::matrix<A>::operator==(std::data::matrix<A> other) {

    if (rows != other.rows || columns != other.columns) return false;
    for (int i = 0; i < rows * columns; i++) if (this->data[i] != other.data[i]) return false;
    return true;
}

/**
 * @brief Check if two matrices are not equal.
 * 
 * @param other Matrix to compare
 * @return @c true if the matrices are not equal, @c false otherwise
 */
template<typename A>
bool std::data::matrix<A>::operator!=(std::data::matrix<A> other) { return !(*this == other); }

/**
 * @brief Create a new matrix object and assign it the value of another matrix object.
 * 
 * @param other Matrix to copy
 * @return New matrix object
 */
template<typename A>
std::data::matrix<A> std::data::matrix<A>::operator=(std::data::matrix<A> other) {

    delete[] this->data;
    rows = other.rows;
    columns = other.columns;
    this->data = new A[rows * columns];
    for (int i = 0; i < rows * columns; i++) this->data[i] = other.data[i];
    return *this;
}

/**
 * @brief Calculate the transpose of a matrix.
 * 
 * @tparam A Type of the matrix elements
 * @param mat Matrix to calculate the transpose of
 * @return Transpose of the matrix
 */
template<typename A>
std::data::matrix<A> std::data::T(std::data::matrix<A> mat) {

    std::data::matrix<A> result(mat.size()[1], mat.size()[0]);
    for (int i = 0; i < mat.size()[0]; i++) 
        for (int j = 0; j < mat.size()[1]; j++) result(j, i) = mat(i, j);
    return result;
}

/**
 * @brief Calculate the trace of a matrix.
 * 
 * @tparam A Type of the matrix elements
 * @param mat Matrix to calculate the trace of
 * @return Trace of the matrix
 */
template<typename A>
A std::data::tr(std::data::matrix<A> mat) {

    if (mat.rows != mat.columns) throw "Matrix is not square";
    A result = A(0);
    for (int i = 0; i < mat.rows; i++) result += mat(i, i);
    return result;
}

/**
 * @brief Calculate the submatrix of a matrix, excluding one row and column.
 * 
 * @tparam A Type of the matrix elements
 * @param mat Matrix to calculate the submatrix of
 * @param excludingRow Row to exclude
 * @param excludingCol Column to exclude
 * @throw "Invalid row or column" if the row or column index is out of bounds
 * @return Submatrix of the matrix
 */
template<typename A>
std::data::matrix<A> std::data::submatrix(std::data::matrix<A> mat, std::data::tuple<int> del_rows, std::data::tuple<int> del_cols) {

    std::data::matrix<A> result(mat.size()[0] - del_rows.size, mat.size()[1] - del_cols.size);
    int r = -1, q = 0;
    for (int i = 0; i < mat.size()[0]; i++) {

        for (int j = 0; j < del_rows.size; j++) if (i == del_rows[j]) q = 1;
        if (q) { q = 0; continue; }
        r++;
        int c = -1;
        for (int j = 0; j < mat.size()[1]; j++) {

            for (int k = 0; k < del_cols.size; k++) if (j == del_cols[k]) q = 1;
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
 * @tparam A Type of the matrix elements
 * @param mat Matrix to calculate the adjugate of
 * @return Adjugate of the matrix
 */
template<typename A>
std::data::matrix<A> std::data::adj(std::data::matrix<A> mat) {

    if (mat.rows != mat.columns) throw "Matrix is not square";
    std::data::matrix<A> result(mat.rows, mat.columns);
    for (int i = 0; i < mat.rows; i++) {
        for (int j = 0; j < mat.columns; j++) {

            std::data::matrix<A> subm = std::data::submatrix(mat, std::data::tuple<int>(i), std::data::tuple<int>(j));
            result(i, j) = std::data::det(subm) * ((i + j) % 2 == 0 ? (A)1 : (A)(-1));
        }
    }
    return std::data::T(result);
}

/**
 * @brief Calculate the determinant of a matrix.
 * 
 * @tparam A Type of the matrix elements
 * @param mat Matrix to calculate the determinant of
 * @return Determinant of the matrix
 */
template<typename A>
A std::data::det(std::data::matrix<A> mat) { 

    if (mat.rows != mat.columns) throw "Matrix is not square";
    else if (mat.rows == 1) return mat(0, 0);
    else {

        A det = A(0);
        for (int p = 0; p < mat.rows; p++) det += mat(0, p) * std::data::det(submatrix(mat, std::data::tuple<int>(0), std::data::tuple<int>(p))) * ((p % 2 == 0) ? A(1) : A(-1));
        return det;
    }
}

#endif