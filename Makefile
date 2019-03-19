

test:
	g++ samples/test.cpp -std=c++1y -o test -Iinclude -O2


openmp:
	g++ samples/test.cpp -std=c++1y -o test -Iinclude -O2 -fopenmp -Wall


example:
	g++ samples/example.cpp -std=c++1y -o example -Iinclude -O2


