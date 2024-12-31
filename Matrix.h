#ifndef MATRIX_H
#define MATRIX_H
#include <vector>
using namespace std;

class Matrix    {
    public:
        int rows;
        int cols;
        vector<vector<double>> matrix;
        
        Matrix(int rows, int cols);

        Matrix transpose();

        // THIS ASSUMES M1 * M2 (eg. dimensions must align, M1: (x,y); M2: (y,z) --> result = (x,z))
        Matrix multiply_matrix(Matrix M);

        Matrix hadamard_product(Matrix M);


        Matrix add(Matrix M);

        // This substracts M from current object
        Matrix subtract(Matrix M);

        // This multiplies M by a scalar "a"
        Matrix multiply_scalar(double a);
        // FOR DEBUGGING
        void print_matrix();
};

#endif