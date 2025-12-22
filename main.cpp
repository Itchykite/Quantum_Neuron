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

3. Gradient of the weight (real part approximation):
   ∂|ψ|^2 / ∂w_i ≈ 2 * Re(ψ* * x_i)
   where ψ* is the complex conjugate of ψ

4. Gradient of the bias:
   ∂|ψ|^2 / ∂bias ≈ 2 * Re(ψ)

5. Weight and bias update:
   w_i ← w_i + lr * 2 * error * Re(ψ* * x_i)
   bias ← bias + lr * 2 * error * Re(ψ)
   where error = target - y, lr = learning rate

6. Quantum layer output:
   y_layer = Σ neurons |ψ_i|^2
   i.e., the sum of outputs of all neurons in the layer

7. Input normalization (optional):
   x_i ← x_i / sqrt(Σ |x_i|^2)
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

    void train(const std::vector<std::complex<double>>& x, double target, double lr) 
    {
        std::complex<double> psi = bias;
        for (size_t i = 0; i < x.size(); ++i)
            psi += weights[i] * x[i];

        double out = std::norm(psi);
        double error = out - target;

        // w_i <- w_i − lr * (out - target) * psi * conj(x_i)
        // b   <- b   − lr * (out - target) * psi
        for (size_t i = 0; i < weights.size(); ++i)
            weights[i] -= lr * error * psi * std::conj(x[i]);

        bias -= lr * error * psi;
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

// ---------------------- Quantum Layer ----------------------
class QuantumLayer 
{
public:
    QuantumLayer(int num_neurons, int input_size, std::mt19937& rng) 
    {
        for (int i = 0; i < num_neurons; ++i)
            neurons.emplace_back(input_size, rng);
    }

    double output(const std::vector<std::complex<double>>& x) 
    {
        double sum = 0.0;
        for (auto& neuron : neurons)
            sum += neuron.output(x);

        return sum;
    }

    void train(const std::vector<std::complex<double>>& x, double target, double lr) 
    {
        double layer_out = output(x);
        double layer_error = layer_out - target; 
        for (auto& neuron : neurons)
            neuron.train_with_layer_error(x, layer_error, lr);
    }

private:
    std::vector<QuantumNeuron> neurons;
};

// ---------------------- Main ----------------------
int main() 
{
    std::random_device rd;
    std::mt19937 rng(rd());

    int input_size = 1;
    int num_neurons = 5; 
    QuantumLayer qlayer(num_neurons, input_size, rng);

    double lr = 0.001;

    for (int epoch = 0; epoch < 2000; ++epoch) 
    {
        double loss = 0.0;

        for (double x = -1.0; x <= 1.0; x += 0.1) 
        {
            std::vector<std::complex<double>> input = { {x, 0.0} };
            double target = x * x;

            double out = qlayer.output(input);
            double err = out - target;
            loss += err * err;

            qlayer.train(input, target, lr);
        }

        if (epoch % 200 == 0)
            std::cout << "Epoch " << epoch << ", Loss: " << loss << std::endl;
    }

    // Test
    for (double x = -1.0; x <= 1.0; x += 0.2) 
    {
        std::vector<std::complex<double>> input = { {x, 0.0} };
        std::cout << "x=" << x << " | predicted=" << qlayer.output(input)
                  << " | actual=" << x*x << std::endl;
    }

    return 0;
}

