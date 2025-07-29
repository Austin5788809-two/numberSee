@echo off
clang++ -std=c++20 net.cpp test.cpp -o test.exe -target x86_64-w64-mingw32 -fuse-ld=ld
clang++ -std=c++20 net.cpp train.cpp -o train.exe -target x86_64-w64-mingw32 -fuse-ld=ld