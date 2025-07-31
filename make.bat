@echo off
g++ -std=c++20 net.cpp test.cpp -o test.exe -pthread
g++ -std=c++20 net.cpp train.cpp -o train.exe -pthread