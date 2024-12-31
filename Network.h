#ifndef NETWORK_H
#define NETWORK_H

#include <vector>
#include "Dense.h"
using namespace std;

class Network    {
    public:
        double learning_rate;
        vector<DenseLayer> layers;
        Matrix output; // Initialized w/ placeholder values in the constructor
        double loss;
        vector<Matrix> x, y; // Initialized w/ placeholder values in the constructor

        Network();

        void add_layer(DenseLayer layer);

        void forward_propogation(int training_sample);

        void back_propogation(int training_sample); 

        void train_model(int epochs, vector<vector<double>> train_x, vector<vector<double>> train_y, double learning_rate);
};

#endif