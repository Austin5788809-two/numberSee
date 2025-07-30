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
network::network(std::vector<size_t> b)
{
    body.resize(b.size());
    for (int i = 0; i < b.size(); i++)
        body[i] = new layer(b[i]);
    for (int i = 0; i < b.size() - 1; i++)
        body[i]->connect(body[i + 1]);
}
network::~network()
{
    for (auto& i : body)
        delete i;
}
void network::input(const vec& in)
{
    body.front()->set_value(in);
}
void ReLU(double& x)
{
    x = (x >= 0) * x;
}
void SIGMOD(double& x)
{
    x = 1 / (1 + exp(-x));
}
void forward(layer* u)
{
    if (!u)
        return;
    for (int j = 0; j < u->next->size(); j++) // 初始化
        u->next->nodes[j].value = 0;
    for (int i = 0; i < u->size(); i++)
        for (int j = 0; j < u->next->size(); j++) // 加权求和
            u->next->nodes[j].value += u->nodes[i].value * u->nodes[i].weight[j];
    for (int j = 0; j < u->next->size(); j++) // 加偏移量
        u->next->nodes[j].value += u->bias.weight[j];
    for (int j = 0; j < u->next->size(); j++) // 激活函数
        switch ((int)u->next->act_f)
        {
        case activate_func_t::ReLU:
            ReLU(u->next->nodes[j].value);
            break;
        case activate_func_t::SIGMOD:
            SIGMOD(u->next->nodes[j].value);
            break;
        default:
            SIGMOD(u->next->nodes[j].value);
            break;
        }
    forward(u->next);
}
void network::output(vec& out)
{
    out.resize(body.back()->size());
    for (int i = 0; i < out.size(); i++)
        out[i] = body.back()->nodes[i].value;
}
void network::save_chromosome(int& lnum, std::vector<int>& nnum, vec& in)
{
    lnum = body.size();
    nnum.resize(lnum);
    for (int i = 0; i < lnum; i++)
        nnum[i] = body[i]->size();
    in.assign(0, 0);
    for (int i = 0; i < lnum; i++)
        for (int j = 0; j < body[i]->size(); j++)
            for (double k : body[i]->nodes[j].weight)
                in.push_back(k);
}
void network::load_chromosome(int lnum, const std::vector<int>& nnum, const vec& in)
{
    for (auto& i : body)
        delete i;
    body.assign(lnum, nullptr);
    for (int i = 0; i < lnum; i++)
        body[i] = new layer(nnum[i]);
    for (int i = 0; i < lnum - 1; i++)
        body[i]->connect(body[i + 1]);
    size_t lcnt = 0, ncnt = 0, wcnt = 0, i = 0;
    for (; ;)
    {
        if (wcnt >= body[lcnt]->nodes[ncnt].weight.size())
            wcnt = 0,
            ncnt++;
        if (ncnt >= body[lcnt]->size())
            ncnt = 0,
            lcnt++;
        if (lcnt >= body.size())
            break;
        body[lcnt]->nodes[ncnt].weight[wcnt++] = in[i++];
    }
}
network_group::network_group(int unum, int lnum, const std::vector<int>& nnum)
{
    units.resize(unum);
    this->lnum = lnum;
    this->nnum = nnum;
    totalnum = 0;
    for (int i = 0; i < lnum - 1; i++)
        totalnum += nnum[i] * nnum[i + 1];
    for (int i = 0; i < unum; i++)
    {
        units[i].first = 0;
        units[i].second.resize(totalnum);
        for (int j = 0; j < totalnum; j++)
            units[i].second[j] = randfloat(-1, 1);
    }
}
void network_group::test()
{
    
}
