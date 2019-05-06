

test:
	clang++ samples/test.cpp -std=c++1y -o test -Iinclude -O2 -Wall


openmp:
	clang++ samples/test.cpp -std=c++1y -o test_parallel -Iinclude -O2 -fopenmp -Wall


persistence:
	clang++ samples/persistence_test.cpp -std=c++1y -o persistence -Iinclude -O2 -fopenmp -Wall


