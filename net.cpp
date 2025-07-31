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

void sigmoid(double& x)
{
    x = 1 / (1 + exp(-x));
}

void network::forward(layer* u)
{
    if (!u || !u->next)
        return;
    for (int j = 0; j < u->next->size(); j++)
        u->next->nodes[j].value = 0;
    for (int i = 0; i < u->size(); i++)
        for (int j = 0; j < u->next->size(); j++)
            u->next->nodes[j].value += u->nodes[i].value * u->nodes[i].weight[j];
    for (int j = 0; j < u->next->size(); j++)
        u->next->nodes[j].value += u->bias.weight[j];
    for (int j = 0; j < u->next->size(); j++)
        switch ((int)u->next->act_f)
        {
        case activate_func_t::RELU:
            ReLU(u->next->nodes[j].value);
            break;
        case activate_func_t::SIGMOID:
            sigmoid(u->next->nodes[j].value);
            break;
        default:
            sigmoid(u->next->nodes[j].value);
            break;
        }
    forward(u->next);
}

void network::forward()
{
    forward(body.front());
}

void network::output(vec& out)
{
    out.resize(body.back()->size());
    for (int i = 0; i < out.size(); i++)
        out[i] = body.back()->nodes[i].value;
}

void network::save_chromosome(int& lnum, std::vector<size_t>& nnum, vec& in)
{
    lnum = body.size();
    nnum.resize(lnum);
    for (int i = 0; i < lnum; i++)
        nnum[i] = body[i]->size();
    in.clear();
    for (int i = 0; i < lnum - 1; ++i)
    {
        for (auto& node : body[i]->nodes)
            for (double w : node.weight)
                in.push_back(w);
        for (double w : body[i]->bias.weight)
            in.push_back(w);
    }
}

void network::load_chromosome(int lnum, const std::vector<size_t>& nnum, const vec& in)
{
    for (auto& l : body) delete l;
    body.clear();
    body.resize(lnum);
    for (int i = 0; i < lnum; i++)
        body[i] = new layer(nnum[i]);
    for (int i = 0; i < lnum - 1; i++)
        body[i]->connect(body[i + 1]);

    size_t idx = 0;
    for (int i = 0; i < lnum - 1; i++)
    {
        for (auto& node : body[i]->nodes)
            for (double& w : node.weight)
                w = in[idx++];
        for (double& w : body[i]->bias.weight)
            w = in[idx++];
    }
}

network_group::network_group(double e, double m, int unum, int lnum, const std::vector<size_t>& nnum) : elite(e), mutate_rate(m)
{
    units.resize(unum);
    this->lnum = lnum;
    this->nnum = nnum;
    totalnum = 0;
    for (int i = 0; i < lnum - 1; ++i)
        totalnum += nnum[i] * nnum[i + 1] + nnum[i + 1]; // weights + bias
    for (int i = 0; i < unum; ++i)
    {
        units[i].first = 0.0;
        units[i].second.resize(totalnum);
        for (double& w : units[i].second)
            w = randfloat(-1.0, 1.0);
    }
}

double similarity(const vec& a, const vec& b)
{
    if (a.size() != b.size())
        return 0.0;
    double dot = 0.0, norm_a = 0.0, norm_b = 0.0;
    for (size_t i = 0; i < a.size(); i++)
    {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    double denom = std::sqrt(norm_a * norm_b);
    return denom == 0.0 ? 0.0 : dot / denom;
}

void network_group::test()
{
    network net(nnum);
    std::mutex mtx;
    auto testone = [&](size_t idx)
    {
        network local_net(nnum);
        local_net.load_chromosome(lnum, nnum, units[idx].second);

        double score = 0.0;
        for (int data_cnt = 1; ; data_cnt++)
        {
            std::ifstream read("data/" + std::to_string(data_cnt));
            if (!read.is_open()) break;
            int ans;
            vec data(2500);
            vec label(10, 0.0);
            read >> ans;
            for (int i = 0; i < 2500; i++)
                read >> data[i];
            label[ans] = 1.0;
            local_net.input(data);
            local_net.forward();
            vec output;
            local_net.output(output);
            score += similarity(label, output);
        }
        std::lock_guard<std::mutex> lock(mtx);
        units[idx].first = score;
    };
    unsigned conc = std::thread::hardware_concurrency();
    conc = conc ? conc : 4;
    std::vector<std::thread> pool;
    for (size_t i = 0; i < units.size(); i++)
    {
        if (pool.size() >= conc)
        {
            pool.front().join();
            pool.erase(pool.begin());
        }
        pool.emplace_back(testone, i);
    }
    for (auto& t : pool)
        t.join();
}

void network_group::cross()
{
    std::sort(units.begin(), units.end(), [](const auto& a, const auto& b){ return a.first > b.first; });
    const int pop_size = units.size();
    const int elite_cnt = std::max(1, (int)(elite * pop_size));
    std::vector<std::pair<double, vec>> next_gen;
    for (int i = 0; i < elite_cnt; i++)
        next_gen.push_back(units[i]);
    std::vector<double> fitness(pop_size);
    for (int i = 0; i < pop_size; i++)
        fitness[i] = units[i].first;
    double sum_fit = std::accumulate(fitness.begin(), fitness.end(), 0.0);
    if (sum_fit <= 0.0)
        std::fill(fitness.begin(), fitness.end(), 1.0);
    std::discrete_distribution<int> roulette(fitness.begin(), fitness.end());
    std::uniform_int_distribution<int> coin(0, 1);
    std::uniform_real_distribution<double> real01(0.0, 1.0);
    std::normal_distribution<double> noise(0.0, 0.1);
    auto make_child = [&]()
    {
        const vec& p1 = units[roulette(rng)].second;
        const vec& p2 = units[roulette(rng)].second;
        vec child = p1;
        if (real01(rng) < 0.9)
        {
            int cut = std::uniform_int_distribution<int>(0, totalnum)(rng);
            for (int i = cut; i < totalnum; i++)
                child[i] = p2[i];
        }
        for (double& w : child)
            if (real01(rng) < mutate_rate)
                w += noise(rng);

        return std::make_pair(0.0, child);
    };
    while (next_gen.size() < pop_size)
        next_gen.push_back(make_child());
    units.swap(next_gen);
}
void network_group::output(int& l, std::vector<size_t>& n, std::vector<vec>& u)
{
    l = lnum;
    n = nnum;
    u.clear();
    for (auto[scr, w] : units)
        u.push_back(w);
}
void network_group::save(const char* file)
{
    std::ofstream write(file);
    if (!write.is_open())
        return std::cerr << "cannot open file '" << file << "'.\n", void();
    write << elite << std::endl
          << mutate_rate << std::endl
          << lnum << std::endl
          << totalnum << std::endl
          << units.size() << std::endl;
    for (size_t i : nnum)
        write << i << ' ';
    write << std::endl;
    for (const auto& [score, w] : units)
    {
        for (double val : w)
            write << val << ' ';
        write << std::endl;
    }
    write.close();
}
void network_group::load(const char* file)
{
    std::ifstream read(file);
    if (!read.is_open())
        return std::cerr << "cannot open file'" << file << "'.\n", void();
    int unum;
    read >> elite >> mutate_rate >> lnum >> totalnum >> unum;
    nnum.resize(lnum);
    for (int i = 0; i < lnum; i++)
        read >> nnum[i];
    units.resize(unum);
    for (auto& [score, w] : units)
    {
        for (double& val : w)
            read >> val;
    }
    read.close();
}

