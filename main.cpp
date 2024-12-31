#include <iostream>
#include <vector>
#include <cmath>
#include "Dense.h"
#include "Network.h"
using namespace std;

int main()  {
    /*
        DO THE FOLLOWING COMMAND IN C:\Users\Angel\C++ Projects\XOR-ANN-from-Scratch>
        g++ main.cpp Network.cpp Dense.cpp Matrix.cpp -o main.exe
        ./main.exe

        IF CHANGES MADE, relaunch Integrated Terminal by right clicking project folder and clicking "Open in Integrated Terminal"    
    */
    

    // Represents XOR inputs, where first feature is X and second feature is Y (computes XOR(X,Y))
    vector<vector<double>> x_train = {{1,1},{1,0},{0,1},{0,0}};
    // Represents XOR outputs in one-hot encoded form, where first value represents TRUE (1) and second value represents FALSE (0)
    vector<vector<double>> y_train = {{0,1},{1,0},{1,0},{0,1}};

    Network neural_network;
    DenseLayer l1(2, 2, "relu");
    DenseLayer output(2, 2, "softmax");
    neural_network.add_layer(l1);
    neural_network.add_layer(output);

    double learning_rate = 0.01;
    neural_network.train_model(40000, x_train, y_train, learning_rate);
    cout << endl;

    // Here, the first output represented chance for True (1), second output represents chance for False (0)
    for (auto &x : x_train) {
        neural_network.predict(x);
        cout << endl;
    }
}