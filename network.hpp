#pragma once
#include <vector>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <string>

std::pair<int, std::vector<int>> get_data(int n)
{
    std::vector<int> data;
    int ans;
    std::ifstream file("data/" + std::to_string(n));
    if (!file.is_open())
        return {-1, std::vector<int>()};
    file >> ans;
    std::string line;
    while (std::getline(file, line))
        for (char c : line)
            data.push_back(c - '0');
    file.close();
    return {ans, data};
}


