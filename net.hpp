// net.hpp
#pragma once
#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <filesystem>

struct node
{
    double value;            // 节点的值
    double weight;           // 与上一层连接的权重
    double bias;             // 偏置
    double delta;            // 用于反向传播的误差
    double sum;              // 加权和
    std::vector<node*> next; // 指向下一层节点的指针
};

struct layer
{
    size_t size;
    std::vector<node*> nodes; // 该层的所有神经元
    layer* next_layer;        // 指向下一层
    layer(size_t s) : size(s), nodes(s, nullptr), next_layer(nullptr)
    {
        for (size_t i = 0; i < size; i++)
            nodes[i] = new node{
    /* value */ 0,
   /* weight */ (double)rand() / RAND_MAX * 2 - 1,
     /* bias */ (double)rand() / RAND_MAX * 2 - 1,
    /* delta */ 0,
      /* sum */ 0,
     /* next */ {}
            };
    }
    ~layer()
    {
        for (auto n : nodes)
            delete n;
    }
    void connect(layer* next_layer)
    {
        for (auto n : nodes)
            for (auto next_n : next_layer->nodes)
                n->next.push_back(next_n);
        this->next_layer = next_layer;
    }
    void save(const std::string& filename) const
    {
        std::ofstream ofs(filename);
        if (!ofs)
            return std::cout << "Error opening file for saving layer." << std::endl, void();
        ofs << size << std::endl;
        for (const auto& n : nodes)
            ofs << " " << n->weight << " " << n->bias << " " << n->delta << std::endl;
        ofs.close();
    }
    void load(const std::string& filename)
    {
        std::ifstream ifs(filename);
        if (!ifs)
            return std::cout << "Error opening file for loading layer." << std::endl, void();
        size_t s;
        ifs >> s;
        size = s;
        for (auto& n : nodes)
            ifs >> n->weight >> n->bias >> n->delta;
    }
};

// 激活函数
double ReLU(double x)
{
    return (x > 0) * x;
}
double ReLU_d(double x)
{
    return x > 0;
}
double sigmoid(double x)
{
    return 1.0 / (1.0 + exp(-x));
}
double sigmoid_d(double x)
{
    return x * (1.0 - x);
}

// 向下传播
void forward(layer* l) // 应传入输入层
{
    if (!l)
        return;
    for (auto n : l->nodes)
    { // 对于每个神经元
        double sum = 0;
        for (auto next_n : n->next) // 对于每个连接的神经元
            sum += n->value * next_n->weight; // 计算加权和
        n->value = ReLU(sum + n->bias);
    }
    forward(l->next_layer); // 递归前往下一层
}

//向上传播
void backward(layer* l, double target) // 应传入输出层和目标值
{
    if (!l)
        return;
    // 输出层误差计算
    if (l->next_layer == nullptr)
        for (auto n : l->nodes)
            n->delta = (n->value - target) * ReLU_d(n->value); // MSE 损失的导数
    else
        // 隐藏层误差计算
        for (auto n : l->nodes)
        {
            double error = 0.0;
            for (auto next_n : n->next)
                error += next_n->delta * next_n->weight; // 累加下一层的误差
            n->delta = error * ReLU_d(n->value);
        }
    backward(l->next_layer, target); // 递归前往上一层
}