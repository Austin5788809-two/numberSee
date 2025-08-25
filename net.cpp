    // net.cpp
#include "net.hpp"
#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <filesystem>
#include <random>
#include <unordered_map>
#include <thread>
#include <mutex>
#include <sstream>
#include <iterator>

std::mt19937 rng{std::random_device{}()};

float randfloat(float l, float r)
{
    if (l == r)
        return l;
    if (l > r)
        std::swap(l, r);
    constexpr double norm = 1.0 / (std::numeric_limits<float>::max() + 1.0);
    double u = static_cast<float>(rng()) * norm;
    return l + u * (r - l);
}

network::network(int layers, const std::vector<int>& sizes, const std::vector<act_f_t>& acts)
{
    worked = false;
    if (layers < 2 || sizes.size() != layers || acts.size() != layers)
        return;
    layer_count = layers;
    layer_sizes = sizes;
    activations = acts;
    activations_d = acts;
    weights.resize(layers - 1);
    biases.resize(layers - 1);
    outputs.resize(layers);
    deltas.resize(layers - 1);
    for (int i = 0; i < layers - 1; ++i)
    {
        weights[i] = Eigen::MatrixXd(layer_sizes[i + 1], layer_sizes[i]);
        biases[i] = Eigen::VectorXd(layer_sizes[i + 1]);
        deltas[i] = Eigen::VectorXd(layer_sizes[i + 1]);
        // 初始化权重和偏置
        for (int r = 0; r < layer_sizes[i + 1]; ++r)
        {
            for (int c = 0; c < layer_sizes[i]; ++c)
                weights[i](r, c) = randfloat(-1.0, 1.0);
            biases[i](r) = randfloat(-1.0, 1.0);
        }
    }
    worked = true;
}
