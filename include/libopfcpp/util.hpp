
#ifndef UTIL_HPP
#define UTIL_HPP

#include "libopfcpp/matrix.hpp"

#include <algorithm>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>

#include <cmath>

namespace opf
{

template <class T>
bool read_mat(const std::string &filename, Mat<T> &data)
{
    std::ifstream file (filename, std::ios::in | std::ios::binary);

    if (!file.is_open())
    {
        std::cerr << "Could not open file: " << filename << std::endl;
        return false;
    }

    int rows, cols;
    file.read((char*)&rows, sizeof(int));
    file.read((char*)&cols, sizeof(int));

    data = make_mat<T>(rows, cols);

    T val;
    for (int i = 0; i < rows; i++)
    {   
        for (int j = 0; j < cols; j++)
        {
            file.read((char*)&val, sizeof(T));
            data[i][j] = val;
        }
    }
    file.close();
    
    return true;
}

template <class T>
bool read_mat_labels(const std::string &filename, Mat<T> &data, std::vector<int> &labels)
{
    std::ifstream file (filename, std::ios::in | std::ios::binary);

    if (!file.is_open())
    {
        std::cerr << "Could not open file: " << filename << std::endl;
        return false;
    }

    int rows, cols;
    file.read((char*)&rows, sizeof(int));
    file.read((char*)&cols, sizeof(int));

    data = make_mat<T>(rows, cols);
    labels = std::vector<int>(rows);

    int label;
    T val;
    for (int i = 0; i < rows; i++)
    {
        // label
        file.read((char*)&label, sizeof(int));
        labels[i] = label;
        
        for (int j = 0; j < cols; j++)
        {
            file.read((char*)&val, sizeof(T));
            data[i][j] = val;
        }
    }
    file.close();
    
    return true;
}


template <class T>
bool write_mat(const std::string &filename, const Mat<T> &data)
{
    int rows, cols = 0;
    rows = data.size();
    if (rows > 0)
        cols = data[0].size();
    if (rows == 0 || cols == 0)
    {
        std::cerr << "Invalid data size:" << rows << ", " << cols << std::endl;
        return false;
    }

    std::ofstream file (filename, std::ios::out | std::ios::binary);

    if (!file.is_open())
    {
        std::cerr << "Could not open file: " << filename << std::endl;
        return false;
    }

    file.write((char*)&rows, sizeof(int));
    file.write((char*)&cols, sizeof(int));

    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            file.write((char*)&data[i][j], sizeof(T));
    
    file.close();
    
    return true;
}

template <class T>
bool write_mat_labels(const std::string &filename, const Mat<T> &data, const std::vector<int> &labels)
{
    int rows, cols = 0;
    rows = data.size();
    if (rows > 0)
        cols = data[0].size();
    if (rows == 0 || cols == 0)
    {
        std::cerr << "Invalid data size:" << rows << ", " << cols << std::endl;
        return false;
    }

    std::ofstream file (filename, std::ios::out | std::ios::binary);
    if (!file.is_open())
    {
        std::cerr << "Could not open file: " << filename << std::endl;
        return false;
    }

    file.write((char*)&rows, sizeof(int));
    file.write((char*)&cols, sizeof(int));


    for (int i = 0; i < rows; i++)
    {
        // label
        file.write((char*)&labels[i], sizeof(int));
        
        for (int j = 0; j < cols; j++)
        {
            file.write((char*)&data[i][j], sizeof(T));
            
        }
    }
    file.close();
    
    return true;
}


// train indices, test indices
std::pair<std::vector<int>, std::vector<int>> stratified_shuffle_split(float train_ratio, const std::vector<int> &labels)
{
    std::map<int, int> totals, target, current;
    std::map<int, int>::iterator it;
    std::pair<std::vector<int>, std::vector<int>> splits;

    int test_sz, train_sz = 0;

    for (int l : labels)
        totals[l]++;

    // Find the number of samples for each class
    for (it = totals.begin(); it != totals.end(); ++it)
    {
        target[it->first] = (int) round((float)it->second * train_ratio);
        train_sz += target[it->first];
    }
    test_sz = labels.size() - train_sz;

    // Initialize output
    splits.first.resize(train_sz);
    splits.second.resize(test_sz);

    // Shuffle indices
    std::vector<int> idx(labels.size());
    for (int i = 0; i < labels.size(); i++)
        idx[i] = i;
    
    std::random_shuffle(idx.begin(), idx.end());
    // Assign folds
    int j, l;
    int train_idx = 0, test_idx = 0;
    for (int i = 0; i < labels.size(); i++)
    {
        j = idx[i];
        l = labels[j];

        if (current[l] < target[labels[j]])
        {
            splits.first[train_idx++] = j;
            current[l]++;
        }
        else
        {
            splits.second[test_idx++] = j;
        }
    }

    return splits;
}


template <class T>
void from_indices(const std::vector<T> &data, const std::vector<int> &indices, std::vector<T> &output)
{
    int size = indices.size();
    output = std::vector<T>(size);

    for (int i = 0; i < size; i++)
    {
        output[i] = data[indices[i]];
    }
}


// Computes accuracy according to: TODO (Insert reference)
float opf_accuracy(const vector<int> preds, const vector<int> ground_truth)
{
    int rows = ground_truth.size();
    set<int> s(ground_truth.begin(), ground_truth.end());
    int nlabels = s.size();

    vector<int> class_occ(nlabels+1, 0);
    for (int i = 0; i < rows; i++)
        class_occ[ground_truth[i]]++;

    Mat<float> errors = std::vector<vector<float>>(nlabels+1, std::vector<float>(2, 0));

    for (int i = 0; i < rows; i++)
    {
        if (ground_truth[i] != preds[i])
        {
            errors[preds[i]][0]++;
            errors[ground_truth[i]][1]++;
        }
    }

    int label_count = 0;

    for (int i = 1; i <= nlabels; i++)
    {
        if (class_occ[i] != 0)
        {
            errors[i][0] /= (float) (rows - class_occ[i]);
            errors[i][1] /= (float) class_occ[i];
            label_count++;
        }
    }

    float error = 0.;

    for (int i = 1; i <= nlabels; i++)
    {
        if (class_occ[i] != 0)
        {
            error += errors[i][0] + errors[i][1];
        }
    }

    return 1. - (error / (2.0 * nlabels));;
}




}

#endif
