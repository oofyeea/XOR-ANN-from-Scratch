#include <iostream>
#include <vector>
#include <string>
#include <random> 
#include "Matrix.h"
#include "Dense.h"
using namespace std;


DenseLayer::DenseLayer(int neurons, int input_size, string activation) : weights(neurons, input_size), biases(neurons, 1) {
    this->activation = activation;

    random_device rd;
    default_random_engine generator(rd());

    double denominator = input_size;

    // Checking to see whether to perform He/Xavier Initialization, limited to ReLU/softmax as of now
    if (activation != "relu") denominator += neurons;
    double standard_deviation = pow(6.0/denominator,.5);

    uniform_real_distribution<double> distribution(-1*standard_deviation, standard_deviation);

    for (int i=0; i<weights.rows; i++)   {
        for (int j=0; j<weights.cols; j++) {
            weights.matrix[i][j] = distribution(generator);
        }
    }
}


int main()  {
    //cout << "hello";
    
    DenseLayer a(10, 5, "relu");
    a.weights.print_matrix();

    /*
        DO THE FOLLOWING COMMAND IN C:\Users\Angel\C++ Projects\XOR-ANN-from-Scratch>
        g++ Dense.cpp Matrix.cpp -o Dense.exe
        ./Dense.exe

        IF CHANGES MADE, relaunch Integrated Terminal by right clicking project folder and clicking "Open in Integrated Terminal"
    */
}
//