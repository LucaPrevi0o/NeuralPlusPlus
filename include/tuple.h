#ifndef TUPLE_H
#define TUPLE_H

namespace std {

    namespace data {

        template<typename A>
        class tuple {

            private:
                A* data;

            public:

                const int size;

                /**
                 * @brief Construct a new tuple object with a variable number of arguments.
                 * 
                 * @param args Arguments to initialize the tuple with
                 */
                template<typename... Args>
                tuple(Args... args) : size(sizeof...(args)) { // Constructor that takes a variable number of arguments
                    
                    A temp[] = { args... }; // Create an array of the arguments
                    data = new A[size]; // Allocate memory for the tuple
                    for (int i = 0; i < size; i++) data[i] = temp[i]; // Assign the arguments to the tuple
                }

                /**
                 * @brief Access the element at the specified index.
                 * 
                 * @param index Index of the element to access
                 * @throw "Index out of bounds" if the index is out of bounds
                 * @return Element at the specified index
                 */
                A& operator[](int index) const { 
                    
                    if (index < 0 || index >= size) throw "Index out of bounds"; // Check if the index is valid
                    return data[index]; // Return the element at the specified index
                } 
                
                ~tuple() { delete[] data; } // Destructor
        };
    }
}

#endif