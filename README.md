# LibOPFcpp
C++17 implementation of OPF classifier in one header. For a Python binding, check [PyOPF](https://github.com/marcoscleison/PyOPF).

## How to cite

The algorithm is described in:

J. P. Papa, A. X. Falcão, and Celso T. N. Suzuki. Supervised pattern classification based on optimum-path forest.  International Journal of Imaging Systems and Technology, 19(2):120-131, 2009.

Bibtex:
```latex
@article{papa2009,
 author = {Papa, J. P. and Falc\~{a}o, A. X. and Suzuki, C. T. N.},
 title = {Supervised Pattern Classification Based on Optimum-path Forest},
 journal = {International Journal of Imaging Systems and Technology},
 issue_date = {June 2009},
 volume = {19},
 number = {2},
 month = jun,
 year = {2009},
 issn = {0899-9457},
 pages = {120--131},
 numpages = {12},
 doi = {10.1002/ima.v19:2},
 publisher = {John Wiley \& Sons, Inc.},
 address = {New York, NY, USA},
 keywords = {graph-search algorithms, image analysis, image foresting transform, pattern recognition, supervised learning},
}
```


## Usage

The library was designed to look like scikit-learn. The main class is called OPFClassifier, and it contains one constructor and two functions:

### Constructor
```cpp
SupervisedOPF(bool precomputed=false, distance_function<T> distance=euclidean_distance<T>);
```

Inputs:
precomputed: true if it will receive precomputed distance matrices. False otherwise.
distance: distance function, case precomputed == false.


### Functions

```cpp
void fit(const Mat<T> &train_data, const std::vector<int> &labels);
```

Fits the classifier.

Inputs:
- train_data: if `precomputed == false`, a matrix where the rows are the samples and the columns are the features. If `precomputed == true`, a distance matrix of `size n_samples x n_samples`.

- labels: an int vector with the ground truths.


`std::vector<int> predict(const Mat<T> &test_data);`

Predicts label values for given data.

Inputs:
- test_data: if `precomputed == false`, a matrix where the rows are the samples and the columns are the features. If `precomputed == true`, a distance matrix of size `n_test_samples x n_train_samples`.

Returns:
- A vector of `n_test_samples` integers corresponding to predicted classes.


### The Mat class

The class `Mat<T>` was heavily inspired by OpenCV's `Mat`, although it is not nearly as powerful. It implements RAII memory management when built with all constructors but `Mat(T* data, size_t rows, size_t cols);`.

Its constructors include:

```cpp
Mat();
Mat(Mat<T>& other);
Mat(const Mat<T>& other);
Mat(size_t rows, size_t cols);
Mat(size_t rows, size_t cols, T val);
Mat(std::shared_ptr<T>& data, size_t rows, size_t cols);
// Does not use RAII.
Mat(T* data, size_t rows, size_t cols);
```

To access data and general assignment, the following functions:

```cpp
// access a row's content, similar to OpenCV's m.ptr<T>(i)
T* row(size_t i);

// m.at(i, j), like OpenCV Mat's m.at<T>(i, j)
T& at(size_t i, size_t j);

// m[i][j] element access
T* operator[](size_t i);

// similar to above, but const
const T* operator[](size_t i) const;

// assign operator does not copy matrix contents
Mat<T>& operator=(const Mat<T>& other);

// copies matrix contents
Mat<T> copy();

// releases content (subtracts the reference counter)
void release();
```

### Example usage

The library works in a scikit-learn manner:

```cpp
opf::SupervisedOPF<float> opf;
opf.fit(train_data, train_labels);
preds = opf.predict(test_data);
```


The following example uses a dataset contained in **data/**. Note that most of the code is preamble, declaration, data reading and pre-processing.

```cpp
#include <iostream>

#include "libopfcpp/OPF.hpp"
#include "libopfcpp/util.hpp" // General utilities

using namespace std;

int main()
{
    opf::Mat<float> data, train_data, test_data;
    vector<int> labels, train_labels, test_labels;

    // Read the data however you prefer
    // There are 
    opf::read_mat_labels("data/digits.dat", data, labels); // in util.hpp

    opf::StratifiedShuffleSplit sss(0.5); // in util.hpp
    pair<vector<int>, vector<int>> splits = sss.split(labels);

    opf::index_by_list<float>(data, splits.first, train_data); // in util.hpp
    opf::index_by_list<float>(data, splits.second, test_data);

    opf::index_by_list<int>(labels, splits.first, train_labels);
    opf::index_by_list<int>(labels, splits.second, test_labels);

    // Fit and predict
    opf::SupervisedOPF<float> opf;
    opf.fit(train_data, train_labels);
    vector<int> preds = opf.predict(test_data);

    // And print accuracy
    float acc = opf::accuracy(test_labels, preds); // in util.hpp
    cout << "Accuracy: " << acc*100 << "%" << endl;
}

```

Compile with:

```bash
g++ samples/example.cpp -std=c++1y -o example -Iinclude
```

Or run:

```
make example
```


Output varies, since it is a random shuffle:

```
Accuracy: 98.5491%
```

## OPF in the literature

`
João Paulo Papa, Silas Evandro Nachif Fernandes, and Alexandre Xavier Falcão. 2017. Optimum-Path Forest based on k-connectivity. Pattern Recognition Letters, 87 (February 2017), 117-126.
`

`
Leonardo Marques Rocha, Fábio A. M. Cappabianco, and Alexandre Xavier Falcão. 2009. Data clustering as an optimum-path forest problem with applications in image analysis. Int J. International Journal of Imaging Systems and Technology. 19, 2 (June 2009), 50-68.
`

`
João P. Papa, Alexandre X. Falcão, Victor Hugo C. de Albuquerque, João Manuel R.S. Tavares. Efficient supervised optimum-path forest classification for large datasets. Pattern Recognition, Volume 45, Issue 1, (2012), 512-520.
`

