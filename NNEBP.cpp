#include <iostream>
#include <cmath>

class NeuralNetwork {
private:
    int i;
    int j;
    int k;
    double **input_ni;
    double **output_nk;
    double learning_rate;
    double **weight_ji;
    double **weight_kj;
    int iteration;
public:
    NeuralNetwork(int i, int j, int k);
    setData(double **input_data, double **output_data);
    iterate(int num);
    getOutput(double **input);
    ~NeuralNetwork();
};

int main() {
    return 0;
}