#pragma once
#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <vector>

struct node
{
    double value;            // 节点的值
    double weight;           // 与上一层连接的权重
    double bias;             // 偏置
    double delta;            // 用于反向传播的误差
    std::vector<node*> next; // 指向下一层节点的指针
};

struct layer
{
    size_t size;
    std::vector<node*> nodes; // 该层
};
