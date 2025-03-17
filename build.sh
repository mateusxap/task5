#!/bin/bash

cd src
g++ -c conv.cpp -o conv.o
g++ -c conv_test.cpp -o conv_test.o -I/usr/include -pthread
g++ conv.o conv_test.o -o conv_test -lgtest -lgtest_main -lpthread
cd ..

./src/conv_test
