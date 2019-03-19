

all:
	g++ src/test.cpp -std=c++1y -o test -I/home/thierry/workspace/LibOPFcpp/include -O2 -Wall


openmp:
	g++ src/test.cpp -std=c++1y -o test -I/home/thierry/workspace/LibOPFcpp/include -O2 -fopenmp -Wall


