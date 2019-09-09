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

#include <sys/time.h>
#include <ctime>
#include <cstdio>
#include <cassert>


#include "libopfcpp/OPF.hpp"
#include "libopfcpp/util.hpp"

using namespace std;
using namespace opf;


typedef timeval timer;
#define outchannel stdout
#define TIMING_START() timer TM_start, TM_now, TM_now1;\
    gettimeofday(&TM_start,NULL);\
    TM_now = TM_start;
#define SECTION_START(M) gettimeofday(&TM_now,NULL);\
    fprintf(outchannel,"================================================\nStarting to measure %s\n",M);
#define TIMING_SECTION(M, measurement) gettimeofday(&TM_now1,NULL);\
    *measurement=(TM_now1.tv_sec-TM_now.tv_sec)*1000.0 + (TM_now1.tv_usec-TM_now.tv_usec)*0.001;\
    fprintf(outchannel,"%.3fms:\tSECTION %s\n",*measurement,M);\
    TM_now=TM_now1;
#define TIMING_END() gettimeofday(&TM_now1,NULL);\
    fprintf(outchannel,"\nTotal time: %.3fs\n================================================\n",\
      	 (TM_now1.tv_sec-TM_start.tv_sec) + (TM_now1.tv_usec-TM_start.tv_usec)*0.000001);


/**
 * This example trains and tests the model in five datasets.
 * For each dataset, we compute testing accuracy and execution time for the regular usage
 * and using precomputed distance matrices.
 */
int main(int argc, char *argv[])
{
    vector<string> datasets = {"data/iris.dat", "data/digits.dat", "data/olivetti_faces.dat", "data/wine.dat", "data/mnist_test.dat"};
    TIMING_START();

    vector<vector<float>> times(5, vector<float>(datasets.size()));
    float measurement;

    for (unsigned int i = 0; i < datasets.size(); i++)
    {
        string dataset = datasets[i];
        Mat<float> data;
        vector<int> labels;

        // Read data
        read_mat_labels(dataset, data, labels);

        // Split
        SECTION_START(dataset.c_str());
        printf("Data size %d x %d\n\n", data.rows, data.cols);

        printf("Preparing data\n");
        StratifiedShuffleSplit sss(0.5);
        pair<vector<int>, vector<int>> splits = sss.split(labels);

        TIMING_SECTION("data split", &measurement);

        Mat<float> train_data, test_data;
        vector<int> train_labels, ground_truth;

        index_by_list<float>(data, splits.first, train_data);
        index_by_list<float>(data, splits.second, test_data);

        index_by_list<int>(labels, splits.first, train_labels);
        index_by_list<int>(labels, splits.second, ground_truth);

        TIMING_SECTION("indexing", &measurement);


        // *********** Training time ***********
        printf("\nRunning OPF...\n");

        // Train clasifier
        SupervisedOPF<float> opf;
        opf.fit(train_data, train_labels);

        TIMING_SECTION("OPF training", &measurement);
        times[0][i] = measurement;
        
        // And predict test data
        vector<int> preds = opf.predict(test_data);

        TIMING_SECTION("OPF testing", &measurement);
        times[1][i] = measurement;
        
        // Measure accuracy
        float acc = accuracy(ground_truth, preds);
        printf("Accuracy: %.3f%%\n", acc*100);
        
        float papa_acc = papa_accuracy(ground_truth, preds);
        printf("Papa's accuracy: %.3f%%\n", papa_acc*100);

        // *********** Precomputed training time ***********
        printf("\n");

        printf("\nRunning OPF with precomputed values...\n");

        Mat<float> precomp_train_data = compute_train_distances<float>(train_data);
        Mat<float> precomp_test_data = compute_test_distances<float>(test_data, train_data);
        TIMING_SECTION("Precompute train and test data", &measurement);
        times[2][i] = measurement;

        // Train clasifier
        SupervisedOPF<float> opf_precomp(true);
        opf_precomp.fit(precomp_train_data, train_labels);

        TIMING_SECTION("OPF precomputed training", &measurement);
        times[3][i] = measurement;
        
        // And predict test data
        preds = opf_precomp.predict(precomp_test_data);

        TIMING_SECTION("OPF precomputed testing", &measurement);
        times[4][i] = measurement;
        
        // Measure accuracy
        acc = accuracy(ground_truth, preds);
        printf("Accuracy: %.3f%%\n", acc*100);
        
        papa_acc = papa_accuracy(ground_truth, preds);
        printf("Papa's accuracy: %.3f%%\n", papa_acc*100);


        cout << "================================================\n" << endl;
    }

    FILE *f;
    f = fopen("timing.txt", "a");
    for (size_t i = 0; i < datasets.size(); i++)
        fprintf(f, "%s;%.3f;%.3f;%.3f;%.3f;%.3f\n", datasets[i].c_str(), times[0][i], times[1][i], times[2][i], times[3][i], times[4][i]);
    fclose(f);

    f = fopen("training.txt", "a");
    for (size_t i = 0; i < datasets.size(); i++)
        fprintf(f, "%s;%.3f\n", datasets[i].c_str(), times[0][i]);
    fclose(f);

    TIMING_END();

    return 0;
}

