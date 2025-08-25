// net.hpp
#pragma once
#include <vector>
#include <Eigen/Dense>

double randfloat(double l, double r);

enum act_f_t
{
    NONE,
    RELU,
    SIGMOID,
    SOFTMAX
};

class network
{
private:
    int layer_count;
    std::vector<int> layer_sizes;
    std::vector<act_f_t> activations;     // 每层的激活函数
    std::vector<act_f_t> activations_d;   // 每层激活函数的导数
    std::vector<Eigen::MatrixXf> weights; // 每层的权重
    std::vector<Eigen::VectorXf> biases;  // 每层的偏置
    std::vector<Eigen::VectorXf> outputs; // 每层的输出
    std::vector<Eigen::VectorXf> deltas;  // 每层的误差
public:
    bool worked; // 在某一函数执行完成后会改变worked的值以表示函数是否成功执行
    network(int layers, const std::vector<int>& sizes, const std::vector<act_f_t>& acts);
    Eigen::VectorXd forward(const Eigen::VectorXd& input);
    void backward(const Eigen::VectorXd& input, const Eigen::VectorXd& target, double learning_rate);
    void save(const char* filename);
    void load(const char* filename);
    void train(const char* foldername);
};
