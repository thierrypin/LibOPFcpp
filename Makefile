

test:
	clang++ samples/test.cpp -std=c++1y -o test -Iinclude -O2 -Wall

openmp:
	clang++ samples/test.cpp -std=c++1y -o test_parallel -Iinclude -O2 -fopenmp -Wall

persistence:
	clang++ samples/persistence_test.cpp -std=c++1y -o persistence -Iinclude -O2 -fopenmp -Wall

utest:
	clang++ samples/test_unsup.cpp -std=c++1y -o test_unsup -Iinclude -g -Wall

clean:
	rm -f test test_parallel persistence test_unsup

