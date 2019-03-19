/******************************************************
 * A simple example usage for SupervisedOPF           *
 *                                                    *
 * Author: Thierry Moreira                            *
 *                                                    *
 ******************************************************/

// Copyright 2019 Thierry Moreira
// 
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// 
//     http://www.apache.org/licenses/LICENSE-2.0
// 
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.


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

