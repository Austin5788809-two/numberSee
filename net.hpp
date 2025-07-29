// net.hpp
#pragma once
#include <vector>
typedef std::vector<double> vec;

double randfloat(double l, double r);
struct error_size_must_be_same{};
struct node
{
    bool connected;
    double value;
    vec weight;
};
enum activate_func_t
{
    ReLU,
    SIGMOD
};
struct layer
{
    std::vector<node> nodes;
    node bias;
    layer* next;
    layer* prev;
    activate_func_t act_f;
    layer(size_t s);
    size_t size(void);
    void set_value(const vec& v);
    void connect(layer* l);
};
class network
{
private:
    std::vector<layer*> body;
public:
    network(std::vector<size_t> b);
    ~network();
    void input(const vec& in);
    void output(vec& out);
    void save_chromosome(vec& out);
    void load_chromosome(int lnum, const std::vector<int>& nnum, const vec& in); // layer num, node num, input
};
class network_group
{
private:
    std::vector<std::pair<double, vec>> units; // score, weights
public:
    network_group();
    void test();
    void cross();
};