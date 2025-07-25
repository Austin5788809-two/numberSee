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
#include <random>
#include <unordered_map>

double randdouble(double l, double r)
{
    static thread_local std::mt19937_64 rng{std::random_device{}()};
    if (l == r) return l;
    if (l > r) std::swap(l, r);
    constexpr double norm = 1.0 / (std::numeric_limits<double>::max() + 1.0);
    double u = static_cast<double>(rng()) * norm;
    return l + u * (r - l);
}

struct node
{
    double value;
    std::unordered_map<node*, double, decltype([](const node* p){ return std::hash<const node*>{}(p); })> nxt_weight;
    node() : value(0) {}
};

struct layer
{
    size_t size;
    std::vector<node*> nodes;
    layer* next;
    layer(size_t s) : size(s), next(nullptr)
    {
        nodes.resize(size + 1); // 最后一个是bia
        for (size_t i = 0; i <= size; i++)
            nodes[i] = new node;
        nodes.back()->value = 1;
    }
    void connect(layer& nxt)
    {
        next = &nxt;
        for (auto& i : nodes)
            for (auto& j : nxt.nodes)
                i->nxt_weight.insert{j, randdouble(-1, 1)};
    }
};
