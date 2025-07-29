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

double randfloat(double l, double r)
{
    static thread_local std::mt19937_64 rng{std::random_device{}()};
    if (l == r)
        return l;
    if (l > r)
        std::swap(l, r);
    constexpr double norm = 1.0 / (std::numeric_limits<double>::max() + 1.0);
    double u = static_cast<double>(rng()) * norm;
    return l + u * (r - l);
}
layer::layer(size_t s) : next(nullptr), prev(nullptr)
{
    nodes.resize(s);
    for (auto& i : nodes)
        i = {false, 0, {}};
}
size_t layer::size()
{
    return nodes.size();
}
void layer::set_value(const vec& v)
{
    if (v.size() != nodes.size())
        throw error_size_must_be_same();
    for (int i = 0; i < nodes.size(); i++)
        nodes[i].value = v[i];
}
void layer::connect(layer* l)
{
    l->prev = this;
    next = l;
    for (auto& i : nodes)
    {
        i.weight.resize(l->size());
        i.connected = true;
        for (auto& j : i.weight)
            j = randfloat(-1, 1);
    }
    bias.weight.resize(l->size());
    for (auto& i : bias.weight)
        i = randfloat(-1, 1);
}
