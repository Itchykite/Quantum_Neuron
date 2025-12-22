/*
Quantum Neuron & Quantum Layer – Formulas

1. Neuron amplitude (ψ):
   ψ = Σ (i=0..n-1) w_i * x_i + bias
   where:
     w_i    – complex weight of the neuron
     x_i    – complex input
     bias   – complex bias

2. Neuron output:
   y = |ψ|^2
   i.e., the output is the squared magnitude of the complex amplitude

3. Gradient (Wirtinger calculus):
   ∂|ψ|² / ∂w_i* = ψ · conj(x_i)
   ∂|ψ|² / ∂bias* = ψ

4. Gradient of the bias:
   ∂|ψ|^2 / ∂bias ≈ 2 * Re(ψ)

5. Weight and bias update:
   w_i <- w_i − lr * (out - target) * psi * conj(x_i)
   b   <- b   − lr * (out - target) * psi
   where error = target - y, lr = learning rate

6. Quantum layer output:
   y_layer = Σ neurons |ψ_i|^2
   i.e., the sum of outputs of all neurons in the layer

7. Input normalization (optional):
   x_i <- x_i / sqrt(Σ |x_i|^2)
   to keep amplitudes in [0,1] and stabilize training

Notes:
- Each neuron is probabilistic in the quantum sense, but here we use deterministic |ψ|^2 as the output.
- A layer with multiple neurons allows approximating more complex functions than a single neuron.
*/

#include <iostream>
#include <complex>
#include <vector>
#include <random>
#include <cmath>

// ---------------------- Quantum Neuron ----------------------
class QuantumNeuron 
{
public:
    QuantumNeuron(int input_size, std::mt19937& rng) 
    {
        weights.resize(input_size);
        std::uniform_real_distribution<double> dist(-1.0, 1.0);

        for (auto& w : weights)
            w = std::complex<double>(dist(rng), dist(rng));

        bias = std::complex<double>(dist(rng), dist(rng));
    }

    double output(const std::vector<std::complex<double>>& x) 
    {
        std::complex<double> psi = bias;
        for (size_t i = 0; i < x.size(); ++i)
            psi += weights[i] * x[i];

        return std::norm(psi); 
    }

    void train_with_layer_error(const std::vector<std::complex<double>>& x, double layer_error, double lr)
    {
        std::complex<double> psi = bias;
        for (size_t i = 0; i < x.size(); ++i)
            psi += weights[i] * x[i];

        for (size_t i = 0; i < weights.size(); ++i)
            weights[i] -= lr * layer_error * psi * std::conj(x[i]);

        bias -= lr * layer_error * psi;
    }

private:
    std::vector<std::complex<double>> weights;
    std::complex<double> bias;
};

// ---------------------- Zero Quantum Neuron ----------------------
class ZeroQuantumNeuron 
{
public:
    ZeroQuantumNeuron(double target) : target(target) 
    {
        bias = std::sqrt(target);
    }

    double output(const std::vector<std::complex<double>>& x) 
    {
        if (std::abs(x[0]) < 1e-8)
            return bias * bias;
        else
            return 0.0;
    }
    
    void train(const std::vector<std::complex<double>>& x, double lr) 
    {
        if (std::abs(x[0]) < 1e-8) 
        {
            double error = (bias * bias) - target;
            bias -= lr * 2.0 * error;
        }
    }

private:
    double target;
    double bias;
};

// ---------------------- Quantum Layer ----------------------
class QuantumLayer 
{
public:
    QuantumLayer(int num_neurons, int input_size, std::mt19937& rng, double zero_target) : zero_neuron(zero_target) 
    {
        for (int i = 0; i < num_neurons; ++i)
            neurons.emplace_back(input_size, rng);
    }

    double output(const std::vector<std::complex<double>>& x) 
    {
        double sum = 0.0;
        sum += zero_neuron.output(x);
        for (auto& neuron : neurons)
            sum += neuron.output(x);

        return sum;
    }

    void train(const std::vector<std::complex<double>>& x, double target, double lr) 
    {
        zero_neuron.train(x, lr);
        double layer_out = output(x);
        double layer_error = layer_out - target; 
        for (auto& neuron : neurons)
            neuron.train_with_layer_error(x, layer_error, lr);
    }

private:
    std::vector<QuantumNeuron> neurons;
    ZeroQuantumNeuron zero_neuron;
};

// ---------------------- Main ----------------------
int main() 
{
    std::random_device rd;
    std::mt19937 rng(rd());

    int input_size = 1;
    int num_neurons = 10; 
    double zero_target = 5.0;
    QuantumLayer qlayer(num_neurons, input_size, rng, zero_target);

    double lr = 0.001;

    for (int epoch = 0; epoch < 10000; ++epoch) 
    {
        double loss = 0.0;

        for (double x = -1.0; x <= 1.0; x += 0.1) 
        {
            std::vector<std::complex<double>> input = { {x, 0.0} };
            double target = x * x + zero_target;

            double out = qlayer.output(input);
            double err = out - target;
            loss += err * err;

            qlayer.train(input, target, lr);
        }

        if (epoch % 200 == 0)
            std::cout << "Epoch " << epoch << ", Loss: " << loss << std::endl;
    }

    std::cout << "\nTesting after training:\n";
    for (double x = -1.0; x <= 1.0; x += 0.2) 
    {
        std::vector<std::complex<double>> input = { {x,0.0} };
        std::cout << "x=" << x << " | predicted=" << qlayer.output(input)
                  << " | actual=" << x*x+zero_target << std::endl;
    }

    // Test zero input
    std::vector<std::complex<double>> zero_input = { {0.0,0.0} };
    std::cout << "x=0 | predicted=" << qlayer.output(zero_input) 
              << " | target=5" << std::endl;

    return 0;
}

