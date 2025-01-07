#include <iostream>
#include <vector>
#include <string>
#include <random> 
#include "Matrix.h"
#include "Dense.h"
using namespace std;


DenseLayer::DenseLayer(int neurons, int input_size, string activation_function) : weights(neurons, input_size), biases(neurons, 1), logits(neurons, 1), activations(neurons, 1) {
    this->activation_function = activation_function;

    random_device rd;
    default_random_engine generator(rd());

    double denominator = input_size;

    // Checking to see whether to perform He/Xavier Initialization, limited to ReLU/softmax as of now
    if (activation_function != "relu") denominator += neurons;
    double standard_deviation = pow(6.0/denominator,.5);

    uniform_real_distribution<double> distribution(-1*standard_deviation, standard_deviation);

    for (int i=0; i<weights.rows; i++)   {
        for (int j=0; j<weights.cols; j++) {
            weights.matrix[i][j] = distribution(generator);
        }
    }

    for (int i=0; i<biases.rows; i++)   {
        biases.matrix[i][0] = 0;
    }
}

// Takes previous activations as input, used for forward prop
Matrix DenseLayer::calculate_logits(Matrix a_prev)   {
    // Equivalent to WX + b
    logits = weights.multiply_matrix(a_prev).add(biases);
    return logits;
}

Matrix DenseLayer::calculate_activations()  {
    if (activation_function == "relu")  {
        for (int i=0; i<activations.matrix.size(); i++)   {
            activations.matrix[i][0] = max(0.0, logits.matrix[i][0]);
        }
    }

    else if (activation_function == "softmax")  {
        double sum = 0;
        for (auto &i : logits.matrix)   {
            sum += exp(i[0]);
        }

        for (int i=0; i<activations.matrix.size(); i++) {
            activations.matrix[i][0] = exp(logits.matrix[i][0])/sum;
        }
    }
    return activations;
}

Matrix DenseLayer::derivative_ReLU()    {
    Matrix out(logits.matrix.size(), 1);
    for (int i=0; i<logits.matrix.size(); i++)  {
        if (logits.matrix[i][0] >= 0 )  {
            out.matrix[i][0] = 1;
        }
        else    {
            out.matrix[i][0] = 0;
        }
    }
    return out;
}

/*
int main()  {    
    DenseLayer a(10, 5, "relu");
    //a.weights.print_matrix();
    DenseLayer b(10, 5, "relu");
    //b.weights.print_matrix();
    DenseLayer c(10, 5, "relu");
    c.weights.print_matrix();
    cout << endl;
    c.weights = a.weights.hadamard_product(b.weights);
    c.weights.print_matrix();
    
        DO THE FOLLOWING COMMAND IN INTEGRATED TERMINAL
        g++ Dense.cpp Matrix.cpp -o Dense.exe
        ./Dense.exe

        IF CHANGES MADE, relaunch Integrated Terminal by right clicking project folder and clicking "Open in Integrated Terminal"
    
}
*/
