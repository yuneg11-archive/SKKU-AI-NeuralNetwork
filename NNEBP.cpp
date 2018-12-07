#include <iostream>
#include <iomanip>
#include <complex>
#include <random> // For c++11
//#include <stdlib.h> // Under c++11

using namespace std;

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
    void SetWeight(double **_weight_ji, double **_weight_kj);
    void PrintWeight();
    void TrainNetwork(int data_num, double **input_n, double **output_n,
                      double learning_rate, int iteration);
    ~NeuralNetwork();
};

NeuralNetwork::NeuralNetwork(int _i, int _j, int _k): i(_i), j(_j), k(_k) {
    default_random_engine generator; // For c++11
    uniform_real_distribution<double> distribution(-0.10, 0.10); // For c++11

    weight_ji = new double*[j];
    for(int tj = 0; tj < j; tj++) {
        weight_ji[tj] = new double[i+1];
        for(int ti = 0; ti <= i; ti++)
            weight_ji[tj][ti] = distribution(generator); // For c++11
            //weight_ji[tj][ti] = (rand()%10000-5000)/(double)100000; // Under c++11
    }

    weight_kj = new double*[k];
    for(int tk = 0; tk < k; tk++) {
        weight_kj[tk] = new double[j+1];
        for(int tj = 0; tj <= j; tj++)
            weight_kj[tk][tj] = distribution(generator); // For c++11
            //weight_kj[tk][tj] = (rand()%10000-5000)/(double)100000; // Under c++11
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

    delete [] h_j;
}

void NeuralNetwork::PrintWeight() {
    cout.setf(ios::fixed);

    cout << "Weight_ji" << endl;
    for(int tj = 0; tj < j; tj++)
        for(int ti = 0; ti <= i; ti++) {
            cout << "[" << tj << "," << ti << "] ";
            cout << setprecision(6) << setw(9) << weight_ji[tj][ti] << " " << endl;
        }
    cout << endl;

    cout << "Weight_kj" << endl;
    for(int tk = 0; tk < k; tk++)
        for(int tj = 0; tj <= j; tj++) {
            cout << "[" << tk << "," << tj << "] ";
            cout << setprecision(6) << setw(9) << weight_kj[tk][tj] << " " << endl;
        }
    cout << endl;

    cout.unsetf(ios::fixed);
}

void NeuralNetwork::TrainNetwork(int n, double **input_n, double **output_n,
                                 double learning_rate, int iteration) {
    double **dEdw_ji = new double*[j];
    for(int tj = 0; tj < j; tj++)
        dEdw_ji[tj] = new double[i+1];

    double **dEdw_kj = new double*[k];
    for(int tk = 0; tk < k; tk++)
        dEdw_kj[tk] = new double[j+1];

    double *h_j = new double[j];
    double *o_k = new double[k];
    double *tmoo1mo = new double[k]; // Common part (t_k-o_k)o_k(1-o_k)

    cout << "Iteration | Error" << endl;
    cout << "--------------------" << endl;
    cout.setf(ios::fixed);

    for(int iter = 1; iter <= iteration; iter++) {
        double error = 0;

        for(int tj = 0; tj < j; tj++)
            for(int ti = 0; ti <= i; ti++)
                dEdw_ji[tj][ti] = 0;

        for(int tk = 0; tk < k; tk++)
            for(int tj = 0; tj <= j; tj++)
                dEdw_kj[tk][tj] = 0;

        for(int tn = 0; tn < n; tn++) {
            // Feed Forward
            for(int tj = 0; tj < j; tj++)
                h_j[tj] = getNeuronOutput(i, input_n[tn], weight_ji[tj]);

            for(int tk = 0; tk < k; tk++)
                o_k[tk] = getNeuronOutput(j, h_j, weight_kj[tk]);

            // Error Back Propagation
            for(int tk = 0; tk < k; tk++) {
                tmoo1mo[tk] = (output_n[tn][tk] - o_k[tk]) * o_k[tk] * (1 - o_k[tk]);
                error += abs(output_n[tn][tk] - o_k[tk]);
                for(int tj = 0; tj < j; tj++)
                    dEdw_kj[tk][tj] += -tmoo1mo[tk] * h_j[tj];
                dEdw_kj[tk][j] += -tmoo1mo[tk];
            }

            for(int tj = 0; tj < j; tj++) {
                double sum = 0;
                for(int tk = 0; tk < k; tk++)
                    sum += weight_kj[tk][tj] * tmoo1mo[tk];

                for(int ti = 0; ti < i; ti++)
                    dEdw_ji[tj][ti] += -input_n[tn][ti] * h_j[tj] * (1 - h_j[tj]) * sum;
                dEdw_ji[tj][i] += -1 * h_j[tj] * (1 - h_j[tj]) * sum;
            }
        }

        // Update Weight
        for(int tj = 0; tj < j; tj++)
            for(int ti = 0; ti <= i; ti++)
                weight_ji[tj][ti] -= learning_rate * dEdw_ji[tj][ti];

        for(int tk = 0; tk < k; tk++)
            for(int tj = 0; tj <= j; tj++)
                weight_kj[tk][tj] -= learning_rate * dEdw_kj[tk][tj];

        if(iter % (iteration/100) == 0)
            cout << setw(9) << iter << " | " << setprecision(6) << setw(8) << error << endl;
    }
    cout.unsetf(ios::fixed);
    cout << endl;

    delete [] tmoo1mo;
    delete [] o_k;
    delete [] h_j;

    for(int tk = 0; tk < k; tk++)
        delete [] dEdw_kj[tk];
    delete [] dEdw_kj;

    for(int tj = 0; tj < j; tj++)
        delete [] dEdw_ji[tj];
    delete [] dEdw_ji;
}

void NeuralNetwork::SetWeight(double **_weight_ji, double **_weight_kj) {
    for(int tj = 0; tj < j; tj++)
            for(int ti = 0; ti <= i; ti++)
                weight_ji[tj][ti] = _weight_ji[tj][ti];

    for(int tk = 0; tk < k; tk++)
            for(int tj = 0; tj <= j; tj++)
                weight_kj[tk][tj] = _weight_kj[tk][tj];
}

NeuralNetwork::~NeuralNetwork() {
    for(int tj = 0; tj < j; tj++)
        delete [] weight_ji[tj];
    delete [] weight_ji;

    for(int tk = 0; tk < k; tk++)
        delete [] weight_kj[tk];
    delete [] weight_kj;
}

int main() {
    // Setup training data
    double **input_n = new double*[11];
    double **output_n = new double*[11];
    for(int i = 0; i < 11; i++) {
        input_n[i] = new double[1];
        output_n[i] = new double[1];
    }
    input_n[0][0] = 0.00;   output_n[0][0] = 0.00;
    input_n[1][0] = 0.10;   output_n[1][0] = 0.09;
    input_n[2][0] = 0.20;   output_n[2][0] = 0.16;
    input_n[3][0] = 0.30;   output_n[3][0] = 0.21;
    input_n[4][0] = 0.40;   output_n[4][0] = 0.24;
    input_n[5][0] = 0.50;   output_n[5][0] = 0.25;
    input_n[6][0] = 0.60;   output_n[6][0] = 0.24;
    input_n[7][0] = 0.70;   output_n[7][0] = 0.21;
    input_n[8][0] = 0.80;   output_n[8][0] = 0.16;
    input_n[9][0] = 0.90;   output_n[9][0] = 0.09;
    input_n[10][0] = 1.00;  output_n[10][0] = 0.00;
    cout << "Training data: f(x) = x(1-x)" << endl;
    for(int i = 0; i < 11; i++) {
        cout.setf(ios::fixed);
        cout << "f(" << setw(4) << setprecision(2) << input_n[i][0];
        cout << ") = " << setw(4) << setprecision(2) << output_n[i][0] << endl;
        cout.unsetf(ios::fixed);
    }
    cout << endl << endl;

    // Initialize neural network
    NeuralNetwork nn(1, 4, 1);
        // Input  layer (1 input neuron)
        // Hidden layer (4 neurons)
        // Output layer (1 output neuron)

    // Weights before training
    cout << "Weights before training" << endl << endl;
    nn.PrintWeight();
    cout << endl;

    // Neural network output before training
    cout << "Neural network output before training" << endl;
    double output[1];
    for(int i = 0; i < 11; i++) {
        nn.GetOutput(input_n[i], output);
        cout.setf(ios::fixed);
        cout << "f(" << setw(4) << setprecision(2) << input_n[i][0];
        cout << ") = " << setw(4) << setprecision(2) << output[0] << endl;
        cout.unsetf(ios::fixed);
    }
    cout << endl << endl;

    // Neural network training
    cout << "Training neural network" << endl << endl;
    nn.TrainNetwork(11, input_n, output_n, 0.7, 500000);
    cout << endl;

    // Weights after training
    cout << "Weights after training" << endl << endl;
    nn.PrintWeight();
    cout << endl;

    // Neural network output after training
    cout << "Neural network output after training" << endl;
    for(int i = 0; i < 11; i++) {
        nn.GetOutput(input_n[i], output);
        cout.setf(ios::fixed);
        cout << "f(" << setw(4) << setprecision(2) << input_n[i][0];
        cout << ") = " << setw(4) << setprecision(2) << output[0] << endl;
        cout.unsetf(ios::fixed);
    }
    cout << endl << endl;

    // Neural network output for test data
    cout << "Neural network output for test data" << endl;
    for(int i = 0; i <= 100; i+=5) {
        input_n[0][0] = i / (double)100;
        nn.GetOutput(input_n[0], output);
        cout.setf(ios::fixed);
        cout << "f(" << setw(4) << setprecision(2) << input_n[0][0];
        cout << ") = " << setw(4) << setprecision(2) << output[0] << endl;
        cout.unsetf(ios::fixed);
    }
    cout << endl;

    return 0;
}