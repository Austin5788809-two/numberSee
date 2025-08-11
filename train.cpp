// train.cpp
#include "net.hpp"
#include <iostream>
#include <fstream>
#include <iomanip>

int main()
{
    network net(3, {900, 128, 10}, {RELU, RELU, SOFTMAX}, {{RELU, RELU, SOFTMAX}});
    net.train("data");
    net.save("network");
    return 0;
}