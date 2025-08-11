// test.cpp
#include "net.hpp"

int main()
{
    network net("network");
    std::cout << "starting input progam...\n";
    system("python user_input.py");
    std::cout << "computing...\n";
    std::ifstream get("user_input");
    std::vector<double> data(900);
    for (int i = 0; i < 900; i++)
        get >> data[i];
    std::vector<double> res;
    net.forward(data, res);
    for (int i = 0; i < 10; i++)
        std::cout << std::fixed << std::setprecision(5) << i << " : " << res[i] << '\n';
    return 0;
}
