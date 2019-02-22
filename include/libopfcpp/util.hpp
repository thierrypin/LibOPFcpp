
#ifndef UTIL_HPP
#define UTIL_HPP

#include "libopfcpp/matrix.hpp"

#include <algorithm>
#include <iostream>
#include <fstream>
#include <random>
#include <chrono>
#include <string>
#include <vector>
#include <map>
#include <set>

#include <cmath>

namespace opf
{


template <class T>
void print_vector(T* v, int size)
{
    std::cout << "[";
    for (int i = 0; i < size; i++)
        std::cout << v[i] << ' ';
    std::cout << "]";
}

template <class T>
void print_matrix(Mat<T> m)
{
    for (int i = 0; i < m.size(); i++)
    {
        T* row = m.row(i);
        print_vector(row, m.cols);
        std::cout << '\n';
    }
    std::cout << std::endl;
}

template <class T>
using distance_function = std::function<T (const T*, const T*, int)>;


template <class T>
T euclidean_distance(const T* a, const T* b, int size)
{
    T sum = 0;
    for (size_t i = 0; i < size; i++)
    {
        sum += pow(a[i]-b[i], 2);
    }
    return (T)sqrt(sum);
}

template <class T>
T magnitude(const T* v, int size)
{
    T sum = 0;
    for (int i = 0; i < size; i++)
    {
        sum += pow(v[i], 2);
    }
    return (T)sqrt(sum);
}

template <class T>
T cosine_distance(const T* a, const T* b, int size)
{
    T dividend = 0;
    for (int i = 0; i < size; i++)
    {
        dividend += a[i] * b[i];
    }

    T divisor = magnitude<T>(a, size) * magnitude<T>(b, size);

    // 1 - cosine similarity
    return 1 - (dividend / divisor);
}

template <class T>
Mat<T> compute_train_distances(const Mat<T> &features, distance_function<T> distance=euclidean_distance<T>)
{
    Mat<float> distances(features.rows, features.rows);
    for (int i = 0; i < features.rows; i++)
        distances[i][i] = 0;

    for (int i = 0; i < features.rows - 1; i++)
    {
        for (int j = i + 1; j < features.rows; j++)
        {
            distances[i][j] = distances[j][i] = distance(features[i], features[j], features.cols);
        }
    }

    return distances;
}

template <class T>
Mat<float> compute_test_distances(const Mat<T> &train_data, const Mat<T> &test_data, distance_function<T> distance=euclidean_distance<T>)
{
    Mat<float> distances(test_data.rows, train_data.rows);
    int vec_size = train_data.cols;

    for (int i = 0; i < distances.rows; i++)
    {
        for (int j = 0; j < distances.cols; j++)
        {
            distances[i][j] = distance(test_data[i], train_data[j], vec_size);
        }
    }

    return distances;
}



template <class T>
bool read_mat(const std::string &filename, Mat<T> &data)
{
    std::ifstream file (filename, std::ios::in | std::ios::binary);

    if (!file.is_open())
    {
        std::cerr << "[util/read_mat] Could not open file: " << filename << std::endl;
        return false;
    }

    int rows, cols;
    file.read((char*)&rows, sizeof(int));
    file.read((char*)&cols, sizeof(int));

    data = Mat<T>(rows, cols);

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
        std::cerr << "[util/read_mat_labels] Could not open file: " << filename << std::endl;
        return false;
    }

    int rows, cols;
    file.read((char*)&rows, sizeof(int));
    file.read((char*)&cols, sizeof(int));

    data = Mat<T>(rows, cols);
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
    int rows = data.rows;
    int cols = data.cols;
    if (rows == 0 || cols == 0)
    {
        std::cerr << "[util/write_mat] Invalid data size:" << rows << ", " << cols << std::endl;
        return false;
    }

    std::ofstream file (filename, std::ios::out | std::ios::binary);

    if (!file.is_open())
    {
        std::cerr << "[util/write_mat] Could not open file: " << filename << std::endl;
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
    int rows = data.rows;
    int cols = data.cols;
    if (rows == 0 || cols == 0)
    {
        std::cerr << "[util/write_mat_labels] Invalid data size:" << rows << ", " << cols << std::endl;
        return false;
    }

    std::ofstream file (filename, std::ios::out | std::ios::binary);
    if (!file.is_open())
    {
        std::cerr << "[util/write_mat_labels] Could not open file: " << filename << std::endl;
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

class StratifiedShuffleSplit
{
private:
    float train_ratio;
    std::default_random_engine random_engine;
    
public:
    StratifiedShuffleSplit(float train_ratio = 0.5) : train_ratio(train_ratio)
    {
        unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
        this->random_engine = std::default_random_engine(seed);
    }
    std::pair<std::vector<int>, std::vector<int>> split(const std::vector<int> &labels);
};

// train indices, test indices
std::pair<std::vector<int>, std::vector<int>> StratifiedShuffleSplit::split(const std::vector<int> &labels)
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
        target[it->first] = (int) round((float)it->second * this->train_ratio);
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
    
    std::shuffle(idx.begin(), idx.end(), this->random_engine);

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
void index_by_list(const std::vector<T> &data, const std::vector<int> &indices, std::vector<T> &output)
{
    int size = indices.size();
    output = std::vector<T>(size);

    for (int i = 0; i < size; i++)
    {
        output[i] = data[indices[i]];
    }
}

template <class T>
void index_by_list(const Mat<T> &data, const std::vector<int> &indices, Mat<T> &output)
{
    int size = indices.size();
    output = Mat<T>(size, data.cols);

    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < data.cols; j++)
            output[i][j] = data[indices[i]][j];
    }
}


// Compute Papa's accuracy 
// Papa, João & Falcão, Alexandre & Suzuki, C.T.N.. (2009). Supervised Pattern Classification Based on Optimum-Path Forest. International Journal of Imaging Systems and Technology. 19. 120 - 131. 10.1002/ima.20188.
float papa_accuracy(const std::vector<int> preds, const std::vector<int> ground_truth)
{
    if (ground_truth.size() != preds.size())
    {
        std::cerr << "[util/papa_accuracy] Error: ground truth and prediction sizes do not match. " << ground_truth.size() << " x " << preds.size() << std::endl;
    }

    int rows = ground_truth.size();
    std::set<int> s(ground_truth.begin(), ground_truth.end());
    int nlabels = s.size();

    std::vector<int> class_occ(nlabels+1, 0);
    for (int i = 0; i < rows; i++)
        class_occ[ground_truth[i]]++;

    Mat<float> errors(nlabels+1, 2, 0);
 
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

float accuracy(const std::vector<int> ground_truth, const std::vector<int> preds)
{
    if (ground_truth.size() != preds.size())
    {
        std::cerr << "[util/accuracy] Error: ground truth and prediction sizes do not match. " << ground_truth.size() << " x " << preds.size() << std::endl;
    }

    int n = ground_truth.size();
    float acc = 0;
    for (int i = 0; i < n; i++)
        if (ground_truth[i] == preds[i])
            acc++;

    return acc / n;

}


}

#endif
