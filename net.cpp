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

network::network(size_t lnum, const std::vector<size_t>& nnum, std::vector<int> af, std::vector<int> afd) : layer_number(lnum), node_number(nnum)
{
    is_worked = false;
    // 不是我喜欢的参数，直接return
    if (lnum != nnum.size() || lnum != af.size() || lnum != afd.size() || lnum == 0)
        return;
    // 设定大小
    weight.resize(lnum - 1);
    bias.resize(lnum);
    value.resize(lnum);
    act_f.resize(lnum);
    act_f_d.resize(lnum);
    act_f = af;
    act_f_d = afd;
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
    }
    // 创建成功
    is_worked = true;
}

void active(std::vector<double>& value, int type) // 工具函数，激活一层
{
    switch (type)
    {
    case act_f_t::NONE:
        break;
    case act_f_t::RELU:
        for (double& i : value)
            i = std::max(i, 0.01 * i);
        break;
    case act_f_t::SIGMOID:
        for (double& i : value)
            i = 1 / (1 + exp(i));
        break;
    case act_f_t::SOFTMAX:
        {
            double sum = 0;
            for (double i : value)
                sum += exp(i);
            for (double& i : value)
                i = exp(i) / sum;
            break;
        }
    default:
        break;
    }
}

void active_d(std::vector<double>& value, const std::vector<double>& g, int type) // 工具函数，激活导数一层
{
    switch (type)
    {
    case act_f_t::NONE:
        break;
    case act_f_t::RELU:
        for (double& i : value)
            i = (i >= 0) ? i : 0.01 * i;
        break;
    case act_f_t::SIGMOID:
        for (double& i : value)
            i = (1 / (1 + exp(i))) * (1 - 1 / (1 + exp(i)));
        break;
    case act_f_t::SOFTMAX:
        {
            double dot_gs = 0.0;
            for (size_t i = 0; i < value.size(); i++)
                dot_gs += g[i] * value[i];
            for (size_t i = 0; i < value.size(); i++)
                value[i] = value[i] * (g[i] - dot_gs);
            break;
        }
    default:
        break;
    }
}

void network::forward(const std::vector<double>& input, std::vector<double>& output)
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
        }
        // 激活函数
        active(value[i], act_f[layer_number]);
        // 将last_value更新为当前层的value
        std::vector<double> temp(value[i]);
        std::swap(last_value, temp);
    }
    // 将输出层的结果复制到output
    output.resize(value.back().size());
    for (int i = 0; i < value.back().size(); i++)
        output[i] = value.back()[i];
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
    forward(input, probs);
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

    std::vector<std::vector<double>> delta(layer_number); // 各层的误差delta
    // 计算输出层误差
    delta.back().resize(node_number.back());
    for (size_t i = 0; i < node_number.back(); i++)
        delta.back()[i] = value.back()[i] - label[i];
    // 反向递推隐藏层误差
    for (int l = (int)layer_number - 2; l >= 0; l--)
    {
        delta[l].assign(node_number[l], 0.0);
        // δ_l[i] = Σ_j δ_{l+1}[j] * w_l[i][j] * σ'_l(z_l[i])
        for (size_t i = 0; i < node_number[l]; i++)
        {
            double sum = 0.0;
            for (size_t j = 0; j < node_number[l + 1]; ++j)
                sum += delta[l + 1][j] * weight[l][i][j];
            delta[l][i] = sum;
        }
        // 激活导数
        active_d(delta[l], std::vector<double>(delta[l].size(), 1.0), act_f_d[l]);
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

template<typename T>
void putline(std::ofstream& put, const std::vector<T>& v)
{
    std::copy(v.begin(), v.end(), std::ostream_iterator<T>(put, " "));
    put << std::endl;
}

void network::save(const char* filename)
{
    is_worked = false;
    std::ofstream put(filename);
    // 不是我喜欢的文件名，直接______
    if (!put.is_open())
        return;
    /**
     *      规则
     *  第一行一个layer_number
     *  第二行共layer_number个数，代表node_number
     *  接下来layer_number - 1组，每组node_number[i]行，每行node_number[i + 1]个数，代表weight
     *  接下来layer_number行，每行node_number[i]个数，代表bias
     *  接下来layer_number行，每行node_number[i]个数，代表value
     *  然后一行layer_number个数，代表激活函数，具体映射：{0: 无, 1: ReLU, 2: sigmoid， 3: softmax}
     *  然后一行layer_number个数，代表激活函数的导数
     */
    put << layer_number << std::endl;
    putline(put, node_number);
    for (int i = 0; i < layer_number - 1; i++)
        for (int j = 0; j < node_number[i]; j++)
            putline(put, weight[i][j]);
    for (int i = 0; i < layer_number; i++)
        putline(put, bias);
    for (int i = 0; i < layer_number; i++)
        putline(put, value);
    putline(put, act_f);
    putline(put, act_f_d);
    is_worked = true;
}

template<typename T>
void getline(std::ifstream& get, std::vector<T>& v)
{
    v.clear();
    std::string line;
    std::getline(get, line);
    std::istringstream ss(line);
    std::copy(std::istream_iterator<T>(ss), std::istream_iterator<T>(), std::back_inserter(v));
}

void network::load(const char* filename)
{
    is_worked = false;
    std::ifstream get(filename);
    // 不是我喜欢的文件名，直接______
    if (!get.is_open())
        return;
    get >> layer_number;
    getline(get, node_number);
    for (int i = 0; i < layer_number - 1; i++)
        for (int j = 0; j < node_number[i]; j++)
            getline(get, weight[i][j]);
    for (int i = 0; i < layer_number; i++)
        getline(get, bias);
    for (int i = 0; i < layer_number; i++)
        getline(get, value);
    getline(get, act_f);
    getline(get, act_f_d);
}
