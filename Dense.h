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
        Matrix logits;
        Matrix activations;
        string activation_function;

        DenseLayer(int neurons, int input_size, string activation_function);

        // Takes previous activations as input, used for forward prop
        Matrix calculate_logits(Matrix a_prev);

        Matrix calculate_activations();
};

#endif
