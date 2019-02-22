
#include <vector>
#include <iostream>
#include <string>

#include <sys/time.h>
#include <ctime>
#include <cstdio>
#include <cassert>


#include "libopfcpp/OPF.hpp"

using namespace std;
using namespace opf;


typedef timeval timer;
#define TIMING_START() timer TM_start, TM_now, TM_now1;\
  gettimeofday(&TM_start,NULL);\
  TM_now = TM_start;
#define SECTION_START(M, ftime) gettimeofday(&TM_now,NULL);\
  fprintf(ftime,"================================================\nStarting to measure %s\n",M);
#define TIMING_SECTION(M, ftime) gettimeofday(&TM_now1,NULL);\
  fprintf(ftime,"%.3fms:\tSECTION %s\n",(TM_now1.tv_sec-TM_now.tv_sec)*1000.0 + (TM_now1.tv_usec-TM_now.tv_usec)*0.001,M);\
  TM_now=TM_now1;
#define TIMING_END(ftime) gettimeofday(&TM_now1,NULL);\
  fprintf(ftime,"\nTotal time: %.3fs\n================================================\n",\
      	 (TM_now1.tv_sec-TM_start.tv_sec) + (TM_now1.tv_usec-TM_start.tv_usec)*0.000001);


#define outchannel stdout

int main(int argc, char *argv[])
{

    vector<string> datasets = {"data/iris.dat", "data/digits.dat", "data/olivetti_faces.dat", "data/wine.dat"};
    TIMING_START();

    for (string dataset : datasets)
    {
        Mat<float> data;
        vector<int> labels;

        // Read data
        read_mat_labels(dataset, data, labels);

        // Split
        SECTION_START(dataset.c_str(), outchannel);
        fprintf(outchannel, "Data size %d x %d\n\n", data.rows, data.cols);

        fprintf(outchannel, "Preparing data\n");
        StratifiedShuffleSplit sss(0.5);
        pair<vector<int>, vector<int>> splits = sss.split(labels);

        TIMING_SECTION("data split", outchannel);

        Mat<float> train_data, test_data;
        vector<int> train_labels, ground_truth;

        index_by_list<float>(data, splits.first, train_data);
        index_by_list<float>(data, splits.second, test_data);

        index_by_list<int>(labels, splits.first, train_labels);
        index_by_list<int>(labels, splits.second, ground_truth);

        TIMING_SECTION("indexing", outchannel);


        // *********** Training time ***********
        fprintf(outchannel, "\nRunning OPF...\n");

        // Train clasifier
        SupervisedOPF<float> opf;
        opf.fit(train_data, train_labels);

        TIMING_SECTION("OPF training", outchannel);
        
        // And predict test data
        vector<int> preds = opf.predict(test_data);

        TIMING_SECTION("OPF testing", outchannel);
        
        // Measure accuracy
        float acc = accuracy(ground_truth, preds);
        fprintf(outchannel, "Accuracy: %.3f%%\n", acc*100);
        
        float papa_acc = papa_accuracy(ground_truth, preds);
        fprintf(outchannel, "Papa's accuracy: %.3f%%\n", papa_acc*100);

        // *********** Precomputed training time ***********
        fprintf(outchannel, "\n");

        fprintf(outchannel, "\nRunning OPF with precomputed values...\n");

        Mat<float> precomp_train_data = compute_train_distances<float>(train_data);
        Mat<float> precomp_test_data = compute_test_distances<float>(train_data, test_data);
        TIMING_SECTION("Precompute train and test data", outchannel);

        // Train clasifier
        SupervisedOPF<float> opf_precomp(true);
        opf_precomp.fit(precomp_train_data, train_labels);

        TIMING_SECTION("OPF precomputed training", outchannel);
        
        // And predict test data
        preds = opf_precomp.predict(precomp_test_data);

        TIMING_SECTION("OPF precomputed testing", outchannel);
        
        // Measure accuracy
        acc = accuracy(ground_truth, preds);
        fprintf(outchannel, "Accuracy: %.3f%%\n", acc*100);
        
        papa_acc = papa_accuracy(ground_truth, preds);
        fprintf(outchannel, "Papa's accuracy: %.3f%%\n", papa_acc*100);


        cout << "================================================\n" << endl;
    }

    TIMING_END(outchannel);

    return 0;
}

