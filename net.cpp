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

std::mt19937 rng{std::random_device{}()};

double randfloat(double l, double r)
{
    if (l == r)
        return l;
    if (l > r)
        std::swap(l, r);
    constexpr double norm = 1.0 / (std::numeric_limits<double>::max() + 1.0);
    double u = static_cast<double>(rng()) * norm;
    return l + u * (r - l);
}

network::network(size_t lnum, const std::vector<size_t>& nnum) : layer_number(lnum), node_number(nnum)
{
    is_worked = false;
    // 不是我喜欢的参数，直接return
    if (lnum != nnum.size() || lnum == 0)
        return;
    // 设定大小
    weight.resize(lnum - 1);
    bias.resize(lnum);
    value.resize(lnum);
    act_f.resize(lnum);
    act_f_d.resize(lnum);
    for (int i = 0; i < lnum; i++)
        bias.resize(nnum[i]),
        value.resize(nnum[i]);
    for (auto& i : bias)
        for (auto& j : i)
            // 初始化为随机值
            j = randfloat(-1, 1);
    for (int i = 0; i < lnum - 1; i++)
    {
        weight[i].resize(nnum[i]);
        for (int j = 0; j < nnum[i]; j++)
            weight[i][j].resize(nnum[i + 1]);
        for (int j = 0; j < nnum[i]; j++)
            for (int k = 0; k < nnum[i + 1]; k++)
                // 初始化为随机值
                weight[i][j][k] = randfloat(-1, 1);
        // 隐藏层的激活函数为ReLU
        act_f[i] = [](double x)->double{ return (x > 0) * x; };
        // 隐藏层的激活函数导数为ReLU_d
        act_f_d[i] = [](double x)->double{ return (x > 0); };
    }
    // 输出层没有激活函数及其导数
    act_f.back() = [](double x)->double{ return x; };
    act_f_d.back() = [](double x)->double{ return 1; };
    // 创建成功
    is_worked = true;
}

void softmax(std::vector<double>& input)
{
    double max_val = *std::max_element(input.begin(), input.end());
    double sum = 0.0;
    for (double& val : input)
        sum += std::exp(val - max_val);
    for (double& val : input)
        val = std::exp(val - max_val) / sum;
}
void network::forward(const std::vector<double>& input, std::vector<double>& output, std::vector<double>& probs)
{
    is_worked = false;
    // 不是我喜欢的参数，直接return
    if ( input.size() != value.front().size())
        return;
    std::vector<double> last_value(input); // 表示上一层的输入
    for (int i = 1; i < layer_number; i++)
    { // 对于每一层
        for (int j = 0; j < value[i].size(); j++)
        { // 对于该层的所有节点的值
            // 初始化
            value[i][j] = bias[i][j];
            for (int k = 0; k < last_value.size(); k++) // 对于上一层所有节点的值（输入）
                // 加权求和
                value[i][j] += weight[i - 1][j][k] * last_value[k];
            // 激活函数
            value[i][j] = act_f[i](value[i][j]);
        }
        // 将last_value更新为当前层的value
        std::vector<double> temp(value[i]);
        std::swap(last_value, temp);
    }
    // 将输出层的结果复制到output
    output.resize(value.back().size());
    for (int i = 0; i < value.back().size(); i++)
        output[i] = value.back()[i];
    // 将softmax的结果复制到probs
    softmax(value.back());
    probs.resize(value.back().size());
    for (int i = 0; i < value.back().size(); i++)
        probs[i] = value.back()[i];
    // 执行成功
    is_worked = true;
}

double network::get_loss(const std::vector<double>& input, const std::vector<double>& label, std::vector<double>& output)
{
    static constexpr double EPS = 1e-12; // 防止log(0)
    is_worked = false;
    // 不是我喜欢的参数，直接return
    if (input.size() != value.front().size() || label.size() != value.back().size())
        return 0;
    std::vector<double> probs;
    // 前向传播得到结果
    forward(input, output, probs);
    // forward没执行成功
    if (!is_worked)
        return 0;
    is_worked = false;
    // 计算损失loss
    double loss = 0;
    for (int i = 0; i < probs.size(); i++)
        loss -= label[i] * std::log(std::max(EPS, std::min(1 - EPS, probs[i])));
    // 执行成功
    is_worked = true;
    return loss;
}

void network::backward(const std::vector<double>& input, const std::vector<double>& label)
{
    static constexpr double learning_rate = 0.01; // 学习率
    is_worked = false;
    // 不是我喜欢的参数，直接______
    if (input.size() != value.front().size() || label.size() != value.back().size())
        return;
    std::vector<double> output;
    double loss = get_loss(input, label, output);
    // forward没执行成功
    if (!is_worked)
        return;
    is_worked = false;

    std::vector<std::vector<double>> delta; // 各层的误差delta
    // 计算输出层误差
    delta.back().resize(node_number.back());
    for (size_t i = 0; i < node_number.back(); i++)
        delta.back()[i] = value.back()[i] - label[i];
    // 反向递推隐藏层误差
    for (int l = (int)layer_number - 2; l >= 0; l--)
    {
        delta[l].assign(node_number[l], 0);
        for (size_t j = 0; j < node_number[l]; l++)
        {
            for (size_t k = 0; k < node_number[l + 1]; k++)
                delta[l][j] += weight[l][j][k] * delta[l + 1][k];
            // 激活函数导数
            delta[l][j] *= act_f_d[l](value[l][j]);
        }
    }
    // 更新权重和偏置
    for (size_t l = 1; l < layer_number; l++)
        for (size_t j = 0; j < node_number[l]; j++)
        {
            // 偏置
            bias[l][j] -= learning_rate * delta[l][j];
            // 权重
            for (size_t i = 0; i < node_number[l - 1]; i++)
                weight[l - 1][i][j] -= learning_rate * delta[l][j] * value[l - 1][i];
        }
    // 执行成功
    is_worked = true;
}
