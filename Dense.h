#ifndef DENSE_H
#define DENSE_H

#include <vector>
#include <string>
#include "Matrix.h"
using namespace std;

class DenseLayer    {
    public:
        Matrix weights;
        Matrix biases;
        string activation;
        DenseLayer(int neurons, int input_size, string activation);
};

#endif
