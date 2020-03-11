	
all: test openmp persistence unsup anomaly datcsv
	

test:
	clang++ samples/test.cpp -std=c++1y -o test -Iinclude -O3 -Wall

openmp:
	clang++ samples/test.cpp -std=c++1y -o test_parallel -Iinclude -O3 -fopenmp -Wall

persistence:
	clang++ samples/persistence_test.cpp -std=c++1y -o persistence -Iinclude -O3 -fopenmp -Wall

unsup:
	clang++ samples/test_unsup.cpp -std=c++1y -o test_unsup -Iinclude -O3 -Wall

anomaly:
	clang++ samples/test_anomaly.cpp -std=c++1y -o test_anomaly -Iinclude -O3 -Wall

datcsv:
	clang++ tools/convert_dat_csv.cpp -std=c++1y -o tools/datcsv -Iinclude -O3 -Wall
	
clean:
	rm -f test test_parallel persistence test_unsup test_anomaly tools/datcsv

