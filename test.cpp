// test.cpp
#include "net.hpp"

int main()
{
    layer input_layer(2500);
    layer hidden_layer(128);
    layer output_layer(10);
    input_layer.connect(&hidden_layer);
    hidden_layer.connect(&output_layer);
    std::cout << "Network initialized" << std::endl;
    std::cout << "Start drawing program..." << std::endl;
    system("python user_input.py");
    std::cout << "Drawing program finished." << std::endl;
    std::cout << "Reading user input..." << std::endl;
    std::ifstream ifs("input");
    if (!ifs)
        return std::cout << "Error opening input file." << std::endl, 1;
    std::vector<double> data(2500);
    for (int i = 0; i < 2500; i++)
        ifs >> data[i],
        std::cout << data[i] << " \n"[i % 50 == 49];
    ifs.close();
    for (int j = 0; j < 2500; j++)
        input_layer.nodes[j]->value = data[j]; // 设置输入层的值
    std::cout << "Calculating output..." << std::endl;
    forward(&input_layer); // 前向传播
    std::cout << "Output values:" << std::endl;
    for (int j = 0; j < 10; j++)
        std::cout << "Class " << j << ": " << output_layer.nodes[j]->value << std::endl;
    system("del input"); // 删除输入文件
    std::cout << "Prediction completed." << std::endl;
    return 0;
}
