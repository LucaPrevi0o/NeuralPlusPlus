#ifndef TENSOR_H
#define TENSOR_H

namespace tensor {

    /**
     * @brief Tensor class for multi-dimensional arrays.
     * 
     * This class represents a tensor, which is a multi-dimensional array.
     * It provides methods for accessing and manipulating the tensor data.
     * * @tparam A Type of the tensor elements
     * * @tparam N Number of dimensions of the tensor
     */
    template<typename A, int N>
    class tensor {

        private:

            A* data; // Pointer to the tensor data
            int capacity[N]; // Array to hold the size of each dimension

            tensor(int dims[N]) {

                for (int i = 0; i < N; i++) {
                    
                    capacity[i] = dims[i];
                    if (dims[i] <= 0) throw "Invalid tensor size";
                }
                data = new A[length()];
            }

            int length() const {
                
                int size = 1;
                for (int i = 0; i < N; i++) size *= capacity[i]; // Calculate the total size of the tensor
                return size; // Return the total size
            }

            void check_indices(const tensor& other) const {

                for (int i = 0; i < N; i++)
                    if (other.capacity[i] != capacity[i]) throw "Index does not match tensor dimensions";
            }

        public:

            /**
             * @brief Constructor for the tensor class with variable number of arguments.
             * 
             * Initializes the tensor with specified dimensions using variadic template.
             * 
             * @param args Variable number of size arguments (must match tensor dimensions N)
             */
            template<typename... Args>
            tensor(Args... args) {

                if (sizeof...(args) != N) throw "Number of size arguments must match tensor dimensions";
                int dims[] = {args...};
                for (int i = 0; i < N; ++i) capacity[i] = dims[i];
                data = new A[length()]; // Allocate memory for the tensor data
            }

            /**
             * @brief Copy constructor for the tensor class.
             * 
             * Initializes a new tensor as a copy of another tensor.
             * 
             * @param other The tensor to copy from
             */
            tensor(const tensor& other) {

                for (int i = 0; i < N; ++i) capacity[i] = other.capacity[i];
                data = new A[length()];
                for (int i = 0; i < length(); ++i) data[i] = other.data[i];
            }

            /**
             * @brief Return the size of the tensor in each dimension.
             * 
             * @return Array of sizes for each dimension
             */
            const int* size() const { return capacity; }

            /**
             * @brief Return the size of a specific dimension.
             * @param index Dimension index 
             * @return Size of the specified dimension
             */
            int size(int index) const { return capacity[index]; }

            /**
             * @brief Access the element at the specified index using variadic arguments.
             * 
             * @param args Variable number of indices (must match tensor dimensions N)
             * @return Reference to the element at the specified index
             */
            template<typename... Args>
            A& operator()(Args... args) const {

                if (sizeof...(args) != N) throw "Number of indices must match tensor dimensions";
                int indices[] = {args...}; // Pack arguments into array
                
                int index = 0;
                for (int i = 0; i < N; i++) {
                    
                    if (indices[i] < 0 || indices[i] >= capacity[i]) throw "Index out of bounds";
                    index = index * capacity[i] + indices[i];
                }
                return data[index];
            }

            /**
             * @brief Add two tensors element-wise.
             * 
             * @param other The tensor to add to this tensor
             * @return A new tensor containing the result of the addition
             * @throw "Tensors have different sizes" if the tensors have different sizes
             */
            tensor operator+(const tensor& other) const {

                check_indices(other); // Check if the sizes of the tensors match
                tensor result(*this); // Create a new tensor to hold the result
                for (int i = 0; i < length(); i++) result.data[i] = data[i] + other.data[i]; // Add the elements of the tensors
                return result; // Return the resulting tensor
            }

            /**
             * @brief Add a scalar to each element of the tensor.
             * 
             * @param scalar The scalar to add
             * @return A new tensor containing the result of the addition
             */
            tensor operator+(const A& scalar) const {

                tensor result(*this); // Create a new tensor to hold the result
                for (int i = 0; i < length(); i++) result.data[i] = data[i] + scalar; // Add the scalar to each element of the tensor
                return result; // Return the resulting tensor
            }

            /**
             * @brief Add a scalar to each element of the tensor.
             * 
             * @param scalar The scalar to add
             * @param t The tensor to which the scalar is added
             * @return A new tensor containing the result of the addition
             */
            friend tensor operator+(const A& scalar, const tensor& t) {

                tensor result(t); // Create a new tensor to hold the result
                for (int i = 0; i < t.length(); i++) result.data[i] = scalar + t.data[i]; // Add the scalar to each element of the tensor
                return result; // Return the resulting tensor
            }

            /**
             * @brief Add another tensor to this tensor element-wise.
             * 
             * @param other The tensor to add to this tensor
             * @return Reference to the current tensor after addition
             * @throw "Tensors have different sizes" if the tensors have different sizes
             */
            tensor operator+=(const tensor& other) {

                check_indices(other); // Check if the sizes of the tensors match
                for (int i = 0; i < length(); i++) data[i] += other.data[i]; // Add the elements of the tensors
                return *this; // Return the current tensor
            }

            /**
             * @brief Add a scalar to each element of this tensor.
             * 
             * @param scalar The scalar to add
             * @return Reference to the current tensor after addition
             */
            tensor operator+=(const A& scalar) {

                for (int i = 0; i < length(); i++) data[i] += scalar; // Add the scalar to each element of the tensor
                return *this; // Return the current tensor
            }

            /**
             * @brief Subtract two tensors element-wise.
             * 
             * @param other The tensor to subtract from this tensor
             * @return A new tensor containing the result of the subtraction
             * @throw "Tensors have different sizes" if the tensors have different sizes
             */
            tensor operator-(const tensor& other) const {

                check_indices(other); // Check if the sizes of the tensors match
                tensor result(*this); // Create a new tensor to hold the result
                for (int i = 0; i < length(); i++) result.data[i] = data[i] - other.data[i]; // Subtract the elements of the tensors
                return result; // Return the resulting tensor
            }

            /**
             * @brief Subtract a scalar from each element of the tensor.
             * 
             * @param scalar The scalar to subtract
             * @return A new tensor containing the result of the subtraction
             */
            tensor operator-(const A& scalar) const {

                tensor result(*this); // Create a new tensor to hold the result
                for (int i = 0; i < length(); i++) result.data[i] = data[i] - scalar; // Subtract the scalar from each element of the tensor
                return result; // Return the resulting tensor
            }

            /**
             * @brief Subtract a scalar from each element of the tensor.
             * 
             * @param scalar The scalar to subtract
             * @param t The tensor from which the scalar is subtracted
             * @return A new tensor containing the result of the subtraction
             */
            friend tensor operator-(const A& scalar, const tensor& t) {

                tensor result(t); // Create a new tensor to hold the result
                for (int i = 0; i < t.length(); i++) result.data[i] = scalar - t.data[i]; // Subtract the scalar from each element of the tensor
                return result; // Return the resulting tensor
            }

            /**
             * @brief Subtract another tensor from this tensor element-wise.
             * 
             * @param other The tensor to subtract from this tensor
             * @return Reference to the current tensor after subtraction
             * @throw "Tensors have different sizes" if the tensors have different sizes
             */
            tensor operator-=(const tensor& other) {

                check_indices(other); // Check if the sizes of the tensors match
                for (int i = 0; i < length(); i++) data[i] -= other.data[i]; // Subtract the elements of the tensors
                return *this; // Return the current tensor
            }

            /**
             * @brief Subtract a scalar from each element of this tensor.
             * 
             * @param scalar The scalar to subtract
             * @return Reference to the current tensor after subtraction
             */
            tensor operator-=(const A& scalar) {

                for (int i = 0; i < length(); i++) data[i] -= scalar; // Subtract the scalar from each element of the tensor
                return *this; // Return the current tensor
            }
            
            /**
             * @brief Multiply two tensors using tensor contraction.
             * 
             * This method performs tensor contraction between two tensors.
             * The last dimension of the first tensor must match the first dimension of the second tensor.
             * 
             * @param other The tensor to multiply with
             * @return A new tensor containing the result of the contraction
             * @throw "Tensor contraction: dimensions do not match" if the last dimension of the first tensor does not match the first dimension of the second tensor
             * * @tparam M Number of dimensions of the second tensor
             */ 
            template<int M> tensor<A, N + M - 2> operator*(const tensor<A, M>& other) const {

                // Contrai sull'ultima dimensione di this e la prima di other
                if (capacity[N - 1] != other.capacity[0]) throw "Tensor contraction: dimensions do not match";

                // Calcola le dimensioni del tensore risultato
                int new_dims[N + M - 2], idx = 0;
                for (int i = 0; i < N - 1; ++i) new_dims[idx++] = capacity[i];
                for (int i = 1; i < M; ++i) new_dims[idx++] = other.capacity[i];

                tensor<A, N + M - 2> result(new_dims);

                // Calcola il numero di elementi da contrarre
                int contracted = capacity[N - 1];

                // Calcola il numero di elementi nelle parti non contratte
                int left_size = 1;
                for (int i = 0; i < N - 1; ++i) left_size *= capacity[i];
                int right_size = 1;
                for (int i = 1; i < M; ++i) right_size *= other.capacity[i];

                for (int i = 0; i < left_size; ++i)
                    for (int j = 0; j < right_size; ++j) {

                        A sum = 0;
                        for (int k = 0; k < contracted; ++k) sum += data[i * contracted + k] * other.data[k * right_size + j];
                        result.data[i * right_size + j] = sum;
                    }

                return result;
            }

            /**
             * @brief Multiply a scalar by each element of the tensor.
             * 
             * @param scalar The scalar to multiply by
             * @return A new tensor containing the result of the multiplication
             */
            tensor operator*(const A& scalar) const {

                tensor result(*this); // Create a new tensor to hold the result
                for (int i = 0; i < length(); i++) result.data[i] = data[i] * scalar; // Multiply each element of the tensor by the scalar
                return result; // Return the resulting tensor
            }

            /**
             * @brief Multiply a scalar by each element of the tensor.
             * 
             * @param scalar The scalar to multiply by
             * @param t The tensor to which the scalar is multiplied
             * @return A new tensor containing the result of the multiplication
             */
            friend tensor operator*(const A& scalar, const tensor& t) {

                tensor result(t); // Create a new tensor to hold the result
                for (int i = 0; i < t.length(); i++) result.data[i] = scalar * t.data[i]; // Multiply each element of the tensor by the scalar
                return result; // Return the resulting tensor
            }

            tensor operator*=(const A& scalar) {

                for (int i = 0; i < length(); i++) data[i] *= scalar; // Multiply each element of the tensor by the scalar
                return *this; // Return the current tensor
            }

            /**
             * @brief Check if two tensors are equal.
             * 
             * Compares the sizes of the dimensions and the elements of the tensors.
             * 
             * @param other The tensor to compare with
             * @return true if the tensors are equal, false otherwise
             */
            bool operator==(const tensor& other) const {

                for (int i = 0; i < N; i++)
                    if (capacity[i] != other.capacity[i]) return false; // Check if the sizes of the dimensions match
                for (int i = 0; i < length(); i++)
                    if (data[i] != other.data[i]) return false; // Check if the elements are equal
                return true; // Tensors are equal
            }

            /**
             * @brief Check if two tensors are not equal.
             * 
             * Compares the sizes of the dimensions and the elements of the tensors.
             * 
             * @param other The tensor to compare with
             * @return true if the tensors are not equal, false otherwise
             */
            bool operator!=(const tensor& other) const { return !(*this == other); }

            /**
             * @brief Assign the value of another tensor to this tensor.
             * 
             * Copies the sizes of the dimensions and the elements from another tensor.
             * 
             * @param other The tensor to copy from
             * @return Reference to the current tensor after assignment
             */
            tensor& operator=(const tensor& other) {

                if (this == &other) return *this; // Check for self-assignment

                delete[] data; // Delete the old data
                for (int i = 0; i < N; i++) capacity[i] = other.capacity[i]; // Copy the sizes of the dimensions
                data = new A[length()]; // Allocate new memory for the tensor data
                for (int i = 0; i < length(); i++) data[i] = other.data[i]; // Copy the elements of the tensor
                return *this; // Return the current tensor
            }

            /**
             * @brief Assign the value of an array to this tensor.
             * 
             * Copies the elements from an array to the tensor.
             * 
             * @param arr The array to copy from
             * @return Reference to the current tensor after assignment
             */
            tensor& operator=(const A* arr) {

                for (int i = 0; i < length(); i++) data[i] = arr[i]; // Copy the elements from the array
                return *this; // Return the current tensor
            }

            template<typename... Args>
            static const tensor zero(Args... args) {

                tensor result(args...); // Create a new tensor to hold the result
                for (int i = 0; i < result.length(); i++) result.data[i] = 0; // Initialize all elements to zero
                return result; // Return the resulting tensor
            }

            template<typename... Args>
            static const tensor identity(Args... args) {

                // Check if the tensor is square
                int dims[] = {args...};
                if (sizeof...(args) != N) throw "Number of size arguments must match tensor dimensions";
                for (int i = 0; i < N - 1; i++)
                    if (dims[i] != dims[i + 1]) throw "Identity tensor must be square";

                tensor result(args...); // Create a new tensor to hold the result
                // Inizializza tutti a zero
                for (int i = 0; i < result.length(); i++) result.data[i] = 0;

                // Imposta a 1 solo le posizioni dove tutti gli indici sono uguali
                int size = dims[0];
                // Genera tutte le tuple (i, i, ..., i)
                for (int i = 0; i < size; ++i) {
                    
                    // Calcola l'indice lineare corrispondente a (i, i, ..., i)
                    int idx = 0;
                    for (int d = 0; d < N; ++d) idx = idx * size + i;
                    result.data[idx] = 1;
                }

                return result; // Return the resulting tensor
            }
    };

    /**
     * @brief Alias for a 2D tensor, representing a matrix.
     * 
     * This alias simplifies the usage of a 2D tensor as a matrix.
     * 
     * @tparam A Type of the matrix elements
     */
    template<typename A>
    using matrix = tensor<A, 2>;

    /**
     * @brief Alias for a 1D tensor, representing a tuple.
     * 
     * This alias simplifies the usage of a 1D tensor as a tuple.
     * 
     * @tparam A Type of the tuple elements
     */
    template<typename A>
    using tuple = tensor<A, 1>;

    /**
     * @brief Calculate the transpose of a matrix.
     * 
     * @tparam T Type of the matrix elements
     * @param mat Matrix to calculate the transpose of
     * @return Transpose of the matrix
     */
    template<typename A>
    matrix<A> T(matrix<A> mat) {

        matrix<A> result(mat.size()[1], mat.size()[0]);
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
    A tr(matrix<A> mat) {

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
    matrix<A> submatrix(matrix<A> mat, tuple<int> del_rows, tuple<int> del_cols) {

        matrix<A> result(mat.size()[0] - del_rows.size()[0], mat.size()[1] - del_cols.size()[0]);
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
    matrix<A> adj(matrix<A> mat) {

        if (mat.size()[0] != mat.size()[1]) throw "Matrix is not square";
        matrix<A> result(mat.size()[0], mat.size()[1]);
        for (int i = 0; i < mat.size()[0]; i++)
            for (int j = 0; j < mat.size()[1]; j++) {

                tuple<int> del_row(1); del_row(0) = i;
                tuple<int> del_col(1); del_col(0) = j;
                matrix<A> subm = submatrix(mat, del_row, del_col);
                result(i, j) = det(subm) * ((i + j) % 2 == 0 ? (A)1 : (A)(-1));
            }
        return A(result);
    }

    /**
     * @brief Calculate the determinant of a matrix.
     * 
     * @tparam T Type of the matrix elements
     * @param mat Matrix to calculate the determinant of
     * @return Determinant of the matrix
     */
    template<typename A>
    A det(matrix<A> mat) {

        if (mat.size()[0] != mat.size()[1]) throw "Matrix is not square";
        else if (mat.size()[0] == 1) return mat(0, 0);
        else {

            A res = A(0);
            for (int p = 0; p < mat.size()[0]; p++) {

                tuple<int> del_row(1); del_row(0) = 0;
                tuple<int> del_col(1); del_col(0) = p;
                res += mat(0, p) * det(submatrix(mat, del_row, del_col)) * ((p % 2 == 0) ? A(1) : A(-1));
            }
            return res;
        }
    }
}

#endif