#ifndef MATRIX_H
#define MATRIX_H
#include <vector>
using namespace std;

class Matrix    {
    public:
        int rows;
        int cols;
        vector<vector<double>> matrix;

        // Default constructor to prevent type specifier error in Dense class
        //Matrix();
        
        Matrix(int rows, int cols);

        Matrix transpose();

        // THIS ASSUMES M1 * M2 (eg. dimensions must align, M1: (x,y); M2: (y,z) --> result = (x,z))
        Matrix multiply_matrix(Matrix M);

        // FOR DEBUGGING
        void print_matrix();
};

#endif