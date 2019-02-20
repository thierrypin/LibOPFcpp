
#include <vector>
#include <algorithm>
#include <iostream>
#include <string>
#include <fstream>
#include <random>
#include <set>

#include <sys/time.h>
#include <ctime>
#include <cstdio>
#include <cassert>


#include "libopfcpp/OPF.hpp"
#include "libopfcpp/util.hpp"

using namespace std;
using namespace opf;


int main(int argc, char *argv[])
{
    vector<string> datasets = {"data/iris.dat", "data/digits.dat"};

    for (string dataset : datasets)
    {
        cout << dataset << endl;

        Mat<float> data;
        vector<int> labels;

        // Read data
        read_mat_labels(dataset, data, labels);

        // Split
        StratifiedShuffleSplit sss(0.5);
        pair<vector<int>, vector<int>> splits = sss.split(labels);

        Mat<float> train_data, test_data;
        vector<int> train_labels, ground_truth;

        from_indices<vector<float>>(data, splits.first, train_data);
        from_indices<vector<float>>(data, splits.second, test_data);

        from_indices<int>(labels, splits.first, train_labels);
        from_indices<int>(labels, splits.second, ground_truth);

        // Train clasifier
        SupervisedOPF<float> opf;
        opf.fit(train_data, train_labels);
        // And predict test data
        vector<int> preds = opf.predict(test_data);

        // Measure accuracy
        float acc = accuracy(ground_truth, preds);
        cout << "\tAccuracy: " << acc * 100 << "%" << endl;
        
        float papa_acc = papa_accuracy(ground_truth, preds);
        cout << "\tPapa's accuracy: " << papa_acc * 100 << "%" << endl;
    }
    return 0;
}

