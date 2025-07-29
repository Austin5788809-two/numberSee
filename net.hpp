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
class layer
{
private:
    std::vector<node> nodes;
    node bia;
    layer* next;
    layer* prev;
public:
    layer(size_t s);
    size_t size(void);
    void set_value(const vec& v);
    void connect(layer& l);
    void output(vec& v);
    void load(const char* file);
    void save(const char* file);

    friend void forward(layer& input);
};

