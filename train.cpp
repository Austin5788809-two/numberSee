// train.cpp
#include "net.hpp"

int main()
{
    network_group agents(0.15, 0.02, 30, 3, {2500, 128, 10});
    for (int i = 1; i <= 100; i++)
    {
        agents.test();
        agents.cross();
    }
    agents.save("agents.txt");
    return 0;
}
