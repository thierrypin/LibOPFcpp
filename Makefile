
all: test openmp persistence utest
	

test:
	clang++ samples/test.cpp -std=c++1y -o test -Iinclude -O3 -Wall

openmp:
	clang++ samples/test.cpp -std=c++1y -o test_parallel -Iinclude -O3 -fopenmp -Wall

persistence:
	clang++ samples/persistence_test.cpp -std=c++1y -o persistence -Iinclude -O3 -fopenmp -Wall

utest:
	clang++ samples/test_unsup.cpp -std=c++1y -o test_unsup -Iinclude -O3 -Wall

clean:
	rm -f test test_parallel persistence test_unsup

