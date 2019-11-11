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

    /****************************
     * Parsing and initialization
     ****************************/
    Mat<float> data, train_data;
    vector<int> labels, train_labels;

    int kmin = stoi(argv[2]);
    int kmax = stoi(argv[3]);
    int step = stoi(argv[4]);
    float frac = 0.5;
    if (argc >= 6)
        frac = stof(argv[5]);

    // Get data
    read_mat_labels<float>(argv[1], data, labels);
    
    // Train/test split
    StratifiedShuffleSplit sss(frac);
    pair<vector<int>, vector<int>> split = sss.split(labels);

    // Matrix indexing
    index_by_list(data, split.first, train_data);
    index_by_list(labels, split.first, train_labels);

    cout << "Train size: " << train_data.rows << "x" << train_data.cols << endl;

    set<int> unique_labels;
    for (size_t i = 0; i < labels.size(); i++)
        unique_labels.insert(labels[i]);
    
    cout << unique_labels.size() << " unique labels." << endl;
    TIMING_SECTION("Read data", outchannel, &measurement);



    cout << "\n\nTraining and prediction" << endl;
    cout << "-----------------------" << endl;

    /****************************
     * Classification
     ****************************/

    UnsupervisedOPF<float> opf;
    // Find best k
    opf.find_best_k(train_data, kmin, kmax, step);
    cout << "k: " << opf.get_k() << endl;

    TIMING_SECTION("Fit", outchannel, &measurement);

    // Predict
    vector<int> assigned_labels = opf.predict(train_data);
    TIMING_SECTION("Predict", outchannel, &measurement);


    /******************************
     * Build correspondence matrix
     ******************************/
    cout << opf.get_n_clusters() << " clusters." << endl;

    
    // Create and assign Mat
    Mat<int> correspondence(unique_labels.size(), opf.get_n_clusters(), 0);
    for (size_t i = 0; i < train_labels.size(); i++)
        correspondence[train_labels[i]][assigned_labels[i]]++;

    // Print results
    for (int i = 0; i < correspondence.rows; i++)
    {
        for (int j = 0; j < correspondence.cols; j++)
            printf("% 4d ", correspondence[i][j]);
        cout << endl;
    }

    TIMING_SECTION("Confusion", outchannel, &measurement);


    /******************************
     * Find anomaly points
     ******************************/
    cout << "\n\nFind anomaly points" << endl;
    cout << "-----------------------" << endl;

    UnsupervisedOPF<float> anomaly(opf.get_k(), false, true, .001);
    vector<int> anomaly_preds = anomaly.fit_predict(train_data);
    
    correspondence = Mat<int>(unique_labels.size(), anomaly.get_n_clusters(), 0);
    for (size_t i = 0; i < train_labels.size(); i++)
        correspondence[train_labels[i]][anomaly_preds[i]]++;

    // Print results
    for (int i = 0; i < correspondence.rows; i++)
    {
        for (int j = 0; j < correspondence.cols; j++)
            printf("% 4d ", correspondence[i][j]);
        cout << endl;
    }

    TIMING_SECTION("Anomaly", outchannel, &measurement);

    /******************************
     * Persistence
     ******************************/

    ////////////////////////////////////////////////
    cout << "\n\nSerialization and persistence" << endl;
    cout << "------------------------------" << endl;

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
    TIMING_SECTION("Serialize and persist", outchannel, &measurement);

    /////////////////////
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
    opf::UnsupervisedOPF<float> opf2 = opf::UnsupervisedOPF<float>::unserialize(contents);
    cout << "Loaded model: k=" << opf2.get_k() << ", " << opf.get_n_clusters() << " clusters" << endl;

    TIMING_SECTION("Loading saved model", outchannel, &measurement);



    ////////////////////////////////////////////////////////
    vector<int> persist_preds = opf2.predict(train_data);

    // Build and populate mat
    Mat<int> correspondence2(unique_labels.size(), opf2.get_n_clusters(), 0);
    for (size_t i = 0; i < persist_preds.size(); i++)
        correspondence2[train_labels[i]][persist_preds[i]]++;

    // Print results
    for (int i = 0; i < correspondence2.rows; i++)
    {
        for (int j = 0; j < correspondence2.cols; j++)
            printf("% 4d ", correspondence2[i][j]);
        cout << endl;
    }


    TIMING_END(outchannel);

    return 0;
}

