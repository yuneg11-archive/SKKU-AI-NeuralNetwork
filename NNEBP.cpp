#include <iostream>
#include <complex>
#include <random>

class NeuralNetwork {
private:
    int i, j, k;
    double **weight_ji;
    double **weight_kj;
    double activation(double x);
    double getNeuronOutput(int input_num, double input[], double weight[]);
public:
    NeuralNetwork(int _i, int _j, int _k);
    void GetOutput(double input[], double output[]);
    ~NeuralNetwork();
};

NeuralNetwork::NeuralNetwork(int _i, int _j, int _k): i(_i), j(_j), k(_k) {
    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution(-0.01, 0.01);

    weight_ji = new double*[j];
    for(int tj = 0; tj < j; tj++) {
        weight_ji[tj] = new double[i+1];
        for(int ti = 0; ti <= i; ti++)
            weight_ji[tj][ti] = distribution(generator);
    }

    weight_kj = new double*[k];
    for(int tk = 0; tk < k; tk++) {
        weight_kj[tk] = new double[j+1];
        for(int tj = 0; tj <= j; tj++)
            weight_kj[tk][tj] = distribution(generator);
    }
}

double NeuralNetwork::activation(double x) {
    return 1 / (1 + exp(-x)); // Sigmoid
}

double NeuralNetwork::getNeuronOutput(int input_num, double input[], double weight[]) {
    double net = 0;

    for(int ti = 0; ti < input_num; ti++)
        net += input[ti] * weight[ti];
    net += weight[input_num];

    return activation(net);
}

void NeuralNetwork::GetOutput(double input[], double output[]) {
    double *h_j = new double[j];

    for(int tj = 0; tj < j; tj++)
        h_j[tj] = getNeuronOutput(i, input, weight_ji[tj]);

    for(int tk = 0; tk < k; tk++)
        output[tk] = getNeuronOutput(j, h_j, weight_kj[tk]);
}

NeuralNetwork::~NeuralNetwork() {
    for(int tj = 0; tj < j; tj++)
        delete weight_ji[tj];
    delete weight_ji;

    for(int tk = 0; tk < k; tk++)
        delete weight_kj[tk];
    delete weight_kj;
}

int main() {
    return 0;
}