#include <iostream>
#include <cmath>
#include <random>

class NeuralNetwork {
private:
    int i, j, k;
    double **weight_ji;
    double **weight_kj;
public:
    NeuralNetwork(int _i, int _j, int _k);
    ~NeuralNetwork();
};

NeuralNetwork::NeuralNetwork(int _i, int _j, int _k): i(_i), j(_j), k(_k) {
    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution(-0.01, 0.01);

    weight_ji = new double*[j];
    for(int tj = 0; tj < j; tj++) {
        weight_ji[tj] = new double[i];
        for(int ti = 0; ti < i; ti++)
            weight_ji[tj][ti] = distribution(generator);
    }

    weight_kj = new double*[k];
    for(int tk = 0; tk < k; tk++) {
        weight_kj[tk] = new double[j];
        for(int tj = 0; tj < j; tj++)
            weight_kj[tk][tj] = distribution(generator);
    }
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