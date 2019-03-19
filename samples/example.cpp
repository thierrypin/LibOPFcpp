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

