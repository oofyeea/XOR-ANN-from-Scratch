#include <iostream>
#include <vector>
#include <cmath>
#include "Dense.h"
#include "Network.h"
using namespace std;

// These are all placeholder initializations
Network::Network() : output(0,0)    {
    loss = 0;
}

void Network::add_layer(DenseLayer layer)    {
    layers.push_back(layer);
    Matrix new_output(layer.biases.matrix.size(), 1);
    output = new_output;
}

void Network::forward_propogation(int training_sample)  {
    loss = 0;
    for (int i=0; i<layers.size(); i++)   {
        if (i==0)   {
            // X is transposed because input is in MxN form, with M training examples and N features in each examples, so transpose allows for dimensions to align with layer 1 weights
            layers[i].calculate_logits(x[training_sample].transpose());
            //layers[i].logits.print_matrix();
            layers[i].calculate_activations();
        }
        else if (i==layers.size()-1)    {
            layers[i].calculate_logits(layers[i-1].activations);
            output = layers[i].calculate_activations();
        }
        else    {
            layers[i].calculate_logits(layers[i-1].activations);
            layers[i].calculate_activations();
        }
    }

    // Calculating loss
    for (int i=0; i<y[0].matrix[0].size(); i++)   {
        loss += -1*log(output.matrix[i][0]) * y[training_sample].matrix[0][i];
    }
}

void Network::back_propogation(int training_sample)    {
    // Sets the cumulative derivative of L w.r.t. z_i tracker equal to the derivative of the loss w.r.t. final layer, which is A - Y in the case of softmax
    Matrix dL_dz_counter(0,0);
    
    if (layers[layers.size()-1].activation_function == "softmax")   {
        // Y is transposed here because the dimensions must align
        dL_dz_counter = layers[layers.size()-1].activations.subtract(y[training_sample].transpose());
    }
    // Actual back-propogation
    for (int i=layers.size()-1; i>-1; i--) {
        Matrix dL_dw_i(0,0), dL_db_i(0,0);
        if (i < layers.size()-1)  {
            if (layers[i].activation_function == "relu")    {
                dL_dz_counter = layers[i+1].weights.transpose().multiply_matrix(dL_dz_counter).hadamard_product(layers[i].derivative_ReLU());
            }
        }

        if (i==0)   {   
            // X is not transposed here because each training sample is in the shape 1x2, so dimensions automatically align
            dL_dw_i = dL_dz_counter.multiply_matrix(x[training_sample]);
        }
        else    {
            dL_dw_i = dL_dz_counter.multiply_matrix(layers[i-1].activations.transpose());
        }

        // This is because dL/dz_i * dz_i/db_i = dL/dz_i since dz_i/db_i = 1
        dL_db_i = dL_dz_counter;

        /*
            This is the layer_i weights adjustment matrix, which uses L2 Regularization
            I was running into an issue of failure to converge in training so I thought it may help
            However, I doubt it did much, still just left it because I can't track the loss landscape/graph it to really know if it helps
        */
        Matrix weights_adjustment = dL_dw_i.add(layers[i].weights.multiply_scalar(2*lambda_));
        layers[i].weights = layers[i].weights.subtract(weights_adjustment.multiply_scalar(learning_rate));

        // This is the layer_i biases update
        layers[i].biases = layers[i].biases.subtract(dL_db_i.multiply_scalar(learning_rate));
    }
}

void Network::train_model(int epochs, vector<vector<double>> train_x, vector<vector<double>> train_y, double learning_rate, double lambda_)   {
    this->learning_rate = learning_rate;
    this->lambda_ = lambda_;

    // Reorganizing Data
    for (vector<double> example : train_x)   {
        Matrix temp_example(1, train_x[0].size());
        temp_example.matrix[0] = example;
        x.push_back(temp_example);
    }

    for (vector<double> example : train_y)   {
        Matrix temp_example(1, train_y[0].size());
        temp_example.matrix[0] = example;
        y.push_back(temp_example);
    }

    // Actual training
    for (int epoch=1; epoch<=epochs; epoch++)  {
        // This iterates through entire data set
        for (int i=0; i<x.size(); i++)  {
            forward_propogation(i);
            back_propogation(i);
        }
        cout << "Loss after epoch " << epoch << ": " << loss << endl;
    }
}

Matrix Network::predict(vector<double> example) {
    Matrix x_test(1, example.size());
    for (int i=0; i<example.size(); i++)    {
        x_test.matrix[0][i] = example[i];
    }

    for (int i=0; i<layers.size(); i++)   {
        if (i==0)   {
            // X is transposed because input is in MxN form, with M training examples and N features in each examples, so transpose allows for dimensions to align with layer 1 weights
            layers[i].calculate_logits(x_test.transpose());
            //layers[i].logits.print_matrix();
            layers[i].calculate_activations();
        }
        else if (i==layers.size()-1)    {
            layers[i].calculate_logits(layers[i-1].activations);
            output = layers[i].calculate_activations();
        }
        else    {
            layers[i].calculate_logits(layers[i-1].activations);
            layers[i].calculate_activations();
        }
    }

    cout << "Prediction for ";
    x_test.print_matrix();
    cout << "------------------" << endl;
    output.transpose().print_matrix();
    return output;
}

/*
int main()  {
    
        DO THE FOLLOWING COMMAND IN C:\Users\Angel\C++ Projects\XOR-ANN-from-Scratch>
        g++ Network.cpp Dense.cpp Matrix.cpp -o Network.exe
        ./Network.exe

        IF CHANGES MADE, relaunch Integrated Terminal by right clicking project folder and clicking "Open in Integrated Terminal"    
    
    

    // Represents XOR inputs, where first feature is X and second feature is Y (computes XOR(X,Y))
    vector<vector<double>> x = {{1,1},{1,0},{0,1},{0,0}};
    // Represents XOR outputs in one-hot encoded form, where first value represents TRUE (1) and second value represents FALSE (0)
    vector<vector<double>> y = {{0,1},{1,0},{1,0},{0,1}};

    Network neural_network;
    DenseLayer l1(2, 2, "relu");
    DenseLayer output(2, 2, "softmax");
    neural_network.add_layer(l1);
    neural_network.add_layer(output);

    double learning_rate = 0.01;
    neural_network.train_model(40000, x, y, learning_rate);

    
    neural_network.predict({1,1});
    cout << endl;

    l1 = neural_network.layers[0];
    output = neural_network.layers[1];
    
    neural_network.x[0].print_matrix();
    cout << endl;
    l1.weights.print_matrix();
    cout << endl;
    l1.biases.print_matrix();
    cout << endl;
    l1.logits.print_matrix();
    cout << endl << "l1 activations: ";
    l1.activations.print_matrix();
    cout << endl;
    output.weights.print_matrix();
    cout << endl;
    output.biases.print_matrix();
    cout << endl;
    output.logits.print_matrix();
    cout << endl << "output activations: ";
    output.activations.print_matrix();
    cout << endl << "network output: ";
    
    for (auto &i : neural_network.output.matrix) {
        cout << i[0] << " ";
    }
}
*/