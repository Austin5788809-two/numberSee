@echo off
clang++ -std=c++20 net.cpp test.cpp -o test.exe -pthread
clang++ -std=c++20 net.cpp train.cpp -o train.exe -pthread