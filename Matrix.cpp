#include <iostream>
#include <vector>
#include <random>
#include "Matrix.h"
using namespace std;

Matrix::Matrix(int rows, int cols)  {
    this->rows = rows;
    this->cols = cols;
    vector<double> t(cols);
    for (int i=0; i<rows; i++)  {
        matrix.push_back(t);
    }
}
/*
Matrix::Matrix() {
    this->rows = 0;
    this->cols = 0;
}
*/
Matrix Matrix::transpose()  {
    Matrix out(cols, rows);
    for (int i=0; i<cols; i++)  {
        for (int j=0; j<rows; j++)  {
            out.matrix[i][j] = matrix[j][i];
        }
    }
    return out;
}

// THIS ASSUMES M1 * M2 (eg. dimensions must align, M1: (x,y); M2: (y,z) --> result = (x,z))
Matrix Matrix::multiply_matrix(Matrix M)    {
    Matrix out(rows, M.cols);
    for (int i=0; i<out.rows; i++)  {
        vector<double> row_i(out.cols);
        for (int j=0; j<out.cols; j++)  {
            double row_i_j = 0;
            for (int k=0; k<cols; k++)  {
                row_i_j += matrix[i][k] * M.matrix[k][j];
            }
            row_i[j] = row_i_j;
        }
        out.matrix[i] = row_i;
    }
    return out;
}

// FOR DEBUGGING
void Matrix::print_matrix() {
    for (auto i : matrix) {
        for (auto j : i)    {
            cout << j << " ";
        }
        cout << endl;
    }
}

/*
int main()  {
    Matrix a(2,3);
    a.matrix[0] = {1,2,3};
    a.matrix[1] = {4,5,6};
    Matrix b = a.transpose();
    a.print_matrix();
    b.print_matrix();
    Matrix c = a.multiply_matrix(b);
    c.print_matrix();
}
*/