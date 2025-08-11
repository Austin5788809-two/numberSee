// net.hpp
#pragma once
#include <vector>
#include <stdexcept>
#include <functional>
#include <tuple>

double randfloat(double l, double r);

enum act_f_t
{
    NONE,
    RELU,
    SIGMOID,
    SOFTMAX
};

struct network
{
    std::vector<std::vector<std::vector<double>>> weight; // weight[i][j][k]表示第i层第j个节点连接下一层第k个节点的权重
    std::vector<std::vector<double>> bias;                // bias[i][j]     表示第i层第j个节点的偏置
    std::vector<std::vector<double>> value;               // value[i][j]    表示第i层第j个节点的值
    std::vector<int> act_f;                               // act_f[i]       表示第i层使用的激活函数
    std::vector<int> act_f_d;                             // act_f[i]       表示第i层使用的激活函数的导数
    size_t layer_number;                                  // layer_number   表示层数
    std::vector<size_t> node_number;                      // node_number[i] 表示第i层的节点数
    bool is_worked = false;                               // 记录每一次调用函数是否成功
    network(size_t lnum, const std::vector<size_t>& nnum, std::vector<int> af, std::vector<int> afd);
    network(const char* filename);
    void forward(const std::vector<double>& input, std::vector<double>& prob);                                        // 前向传播, 返回输出层的值，用于测试
    double get_loss(const std::vector<double>& input, const std::vector<double>& label, std::vector<double>& output); // 前向传播，返回损失和输出，用于反向传播
    void backward(const std::vector<double>& input, const std::vector<double>& label);                                // 前向传播+反向传播，用于训练
    void save(const char* filename);
    void load(const char* filename);
    void train(const char* foldername);
};
