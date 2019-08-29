/******************************************************
 * Example usage and time testing for SupervisedOPF.  *
 * Useful for comparing time using openmp.            *
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

#include <vector>
#include <iostream>
#include <string>
#include <set>

#include <sys/time.h>
#include <ctime>
#include <cstdio>
#include <cassert>

#include <libopfcpp/OPF.hpp>
#include <libopfcpp/util.hpp>

using namespace std;
using namespace opf;


typedef timeval timer;
#define TIMING_START() timer TM_start, TM_now, TM_now1;\
    gettimeofday(&TM_start,NULL);\
    TM_now = TM_start;
#define SECTION_START(M, ftime) gettimeofday(&TM_now,NULL);\
    fprintf(ftime,"================================================\nStarting to measure %s\n",M);
#define TIMING_SECTION(M, ftime, measurement) gettimeofday(&TM_now1,NULL);\
    *measurement=(TM_now1.tv_sec-TM_now.tv_sec)*1000.0 + (TM_now1.tv_usec-TM_now.tv_usec)*0.001;\
    fprintf(ftime,"%.3fms:\tSECTION %s\n",*measurement,M);\
    TM_now=TM_now1;
#define TIMING_END(ftime) gettimeofday(&TM_now1,NULL);\
    fprintf(ftime,"\nTotal time: %.3fs\n================================================\n",\
      	 (TM_now1.tv_sec-TM_start.tv_sec) + (TM_now1.tv_usec-TM_start.tv_usec)*0.000001);


#define outchannel stdout


/**
 * This example trains and tests the model in five datasets.
 * For each dataset, we compute testing accuracy and execution time for the regular usage
 * and using precomputed distance matrices.
 */
int main(int argc, char *argv[])
{
    float measurement;
    TIMING_START();

    if (argc < 5)
    {
        cerr << "Usage: " << argv[0] << " [input.dat] [kmin] [kmax] [step] [frac]" << endl;
        exit(1);
    }

    Mat<float> data, train_data;
    vector<int> labels, train_labels;
    read_mat_labels<float>(argv[1], data, labels);
    
    int kmin = stoi(argv[2]);
    int kmax = stoi(argv[3]);
    int step = stoi(argv[4]);
    float frac = 0.1;
    if (argc >= 6)
        frac = stof(argv[5]);

    StratifiedShuffleSplit sss(frac);
    pair<vector<int>, vector<int>> split = sss.split(labels);

    index_by_list(data, split.first, train_data);
    index_by_list(labels, split.first, train_labels);
    
    // train_data = data;
    // train_labels = labels;

    cout << "Train size: " << train_data.rows << "x" << train_data.cols << endl;

    set<int> unique_labels;
    for (size_t i = 0; i < labels.size(); i++)
        unique_labels.insert(labels[i]);
    
    cout << unique_labels.size() << " unique labels." << endl;

    // for (auto it = unique_labels.begin(); it != unique_labels.end(); ++it)
    //     cout << *it << " ";
    // cout << endl;

    TIMING_SECTION("Read data", outchannel, &measurement);



    UnsupervisedOPF<float> opf;
    opf.find_best_k(train_data, kmin, kmax, step);
    // for (int i = 0; i < 74; i++)
    //     opf.fit(train_data);

    cout << "k: " << opf.get_k() << endl;

    TIMING_SECTION("Fit", outchannel, &measurement);

    vector<int> assigned_labels = opf.predict(train_data);

    TIMING_SECTION("Predict", outchannel, &measurement);

    cout << opf.get_n_clusters() << " clusters." << endl;

    Mat<int> correspondence(unique_labels.size(), opf.get_n_clusters(), 0);
    cout << "Build mat " << correspondence.rows << "x" << correspondence.cols << endl;

    for (size_t i = 0; i < train_labels.size(); i++)
        correspondence[train_labels[i]][assigned_labels[i]]++;

    
    cout << "Populate mat" << endl;

    for (int i = 0; i < correspondence.rows; i++)
    {
        for (int j = 0; j < correspondence.cols; j++)
        {
            printf("% 4d ", correspondence[i][j]);
            // cout << correspondence[i][j] << " ";
        }
        cout << endl;
    }

    cout << opf.get_n_clusters() << " clusters." << endl;

    TIMING_SECTION("Confusion", outchannel, &measurement);

    //

    TIMING_END(outchannel);

    return 0;
}

