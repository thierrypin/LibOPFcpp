

test:
	clang++ samples/test.cpp -std=c++1y -o test -Iinclude -O2


openmp:
	clang++ samples/test.cpp -std=c++1y -o test -Iinclude -O2 -fopenmp -Wall


example:
	clang++ samples/example.cpp -std=c++1y -o example -Iinclude -O2 -fopenmp


