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
#include <fstream>

#include "libopfcpp/OPF.hpp"
#include "libopfcpp/util.hpp" // General utilities

using namespace std;

/**
 * This example trains a model, computes its accuracy and saves the model into a file.
 * Then it loads the file and computes its accuracy again.
 */
int main()
{
    opf::Mat<float> data, train_data, test_data;
    vector<int> labels, train_labels, test_labels;

    // Read the data however you prefer
    // There are a few helper functions in libopfcpp/util.hpp
    opf::read_mat_labels("data/digits.dat", data, labels); // in util.hpp
    cout << "Data shape: " << data.rows << ", " << data.cols << endl;

    opf::StratifiedShuffleSplit sss(0.5); // in util.hpp
    pair<vector<int>, vector<int>> splits = sss.split(labels);

    opf::index_by_list<float>(data, splits.first, train_data); // in util.hpp
    opf::index_by_list<float>(data, splits.second, test_data);

    opf::index_by_list<int>(labels, splits.first, train_labels);
    opf::index_by_list<int>(labels, splits.second, test_labels);


    cout << "########################################" << endl;
    cout << "Supervised OPF" << endl;
    /////////////////////////////////////////////
    // Fit and predict
    opf::SupervisedOPF<float> opf;
    opf.fit(train_data, train_labels);
    vector<int> preds = opf.predict(test_data);
    /////////////////////////////////////////////

    // And print accuracy
    float acc1 = opf::accuracy(test_labels, preds); // in util.hpp
    cout << "Accuracy: " << acc1*100 << "%" << endl;



    /*** Now test persistence ***/
    // Write
    cout << "Write..." << endl;

    {   // Sub scope to destroy variable "contents"
        std::string contents = opf.serialize(opf::SFlags::Sup_SavePrototypes);
        std::ofstream ofs ("teste.dat", std::ios::out | std::ios::binary);
        if (!ofs)
        {
            std::cout << "Can't open file" << std::endl;
            return -1;
        }
        
        opf::write_bin<char>(ofs, contents.data(), contents.size());
        ofs.close();
    }



    // Read
    cout << "Read..." << endl;
    // Read file contents
    std::ifstream ifs ("teste.dat", std::ios::in | std::ios::binary);
    if (!ifs)
    {
        std::cout << "Can't open file" << std::endl;
        return -1;
    }
    std::string contents( (std::istreambuf_iterator<char>(ifs)), (std::istreambuf_iterator<char>()) );
    ifs.close();

    // Unserialize contents into an OPF object
    opf::SupervisedOPF<float> opf2 = opf::SupervisedOPF<float>::unserialize(contents);

    // Predict again an check if accuracies are equal
    preds = opf2.predict(test_data);
    float acc2 = opf::accuracy(test_labels, preds); // in util.hpp
    cout << "Accuracy: " << acc2*100 << "%" << endl;


    if (acc1 == acc2)
        cout << "Model persistence successful." << endl;
    else
        cout << "Model persistence failed." << endl;

    cout << "########################################" << endl;
    cout << "Unsupervised OPF" << endl;

    return 0;
}


