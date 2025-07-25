// train.cpp
#include "net.hpp"

int main()
{
    srand(time(0));
    layer input_layer(2500); // 50 * 50 matrix
    layer hidden_layer(128);
    layer output_layer(10); // 10 classes for digits 0-9
    input_layer.connect(&hidden_layer);
    hidden_layer.connect(&output_layer);
    std::cout << "Network initialized" << std::endl;
    std::cout << "finding data files..." << std::endl;
    int maxn = 0;
    // 查找数据文件
    for (const auto& entry : std::filesystem::directory_iterator("data"))
        if (entry.is_regular_file() && entry.path().extension() == "")
            maxn = std::max(maxn, std::stoi(entry.path().stem().string()));
    if (!maxn)
        return std::cout << "No data files found." << std::endl, 0;
    std::cout << "Found " << maxn << " data files." << std::endl;
    std::cout << "Training..." << std::endl;
    std::ifstream ifs;
    for (int k = 1; k <= 10; k++) // 训练十次
        for (int i = 1; i <= maxn; i++)
        {
            ifs.open("data/" + std::to_string(i));
            int ans;
            std::vector<double> data(2500);
            ifs >> ans; // 获取标签
            for (int i = 0; i < 2500; i++) // 获取数据
                ifs >> data[i];
            ifs.close();
            for (int j = 0; j < 2500; j++)
                input_layer.nodes[j]->value = data[j]; // 设置输入层的值
            forward(&input_layer); // 前向传播
            for (int j = 0; j < 10; j++) // 计算输出层的误差
                output_layer.nodes[j]->delta = (j == ans ? 1.0 : 0.0) - output_layer.nodes[j]->value;
            backward(&output_layer, ans); // 反向传播
            for (auto n : output_layer.nodes) // 更新权重和偏置
                n->weight += 0.01 * n->delta * n->value, // 简单的梯度下降
                n->bias += 0.01 * n->delta;
        }
    std::cout << "Training completed." << std::endl;
    // 保存网络
    input_layer.save("model/input_layer");
    hidden_layer.save("model/hidden_layer");
    output_layer.save("model/output_layer");
    std::cout << "Network saved." << std::endl;
    std::cout << "Training finished." << std::endl;
    return 0;
}
