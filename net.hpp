// net.hpp
#pragma once
#include <vector>
#include <stdexcept>
typedef std::vector<double> vec;

double randfloat(double l, double r);
struct error_size_must_be_same : public std::exception{};
struct node
{
    bool connected;
    double value;
    vec weight;
};
enum activate_func_t
{
    RELU,
    SIGMOID
};
struct layer
{
    std::vector<node> nodes;
    node bias;
    layer* next;
    layer* prev;
    activate_func_t act_f = activate_func_t::SIGMOID;
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
    void forward(layer* u);
    void forward();
    void output(vec& out);
    void save_chromosome(int& lnum, std::vector<size_t>& nnum, vec& out);
    void load_chromosome(int lnum, const std::vector<size_t>& nnum, const vec& in); // layer num, node num, input
};
class network_group
{
private:
    double elite; // 分数，代表选取的比例
    double mutate_rate; // 变异率
    int lnum;
    std::vector<size_t> nnum;
    int totalnum;
    std::vector<std::pair<double, vec>> units; // score, weights
public:
    network_group(double e, double m, int unum, int lnum, const std::vector<size_t>& nnum);
    void test();
    void cross();
    void output(int& l, std::vector<size_t>& n, std::vector<vec>& u);
    void save(const char* file);
    void load(const char* file);
};