
#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <functional>
#include <iostream>
namespace opf
{


/*****************************************/
/************** Matrix type **************/
/*****************************************/

using distance_function = std::function<float (std::vector<float>, std::vector<float>)>;

template <class T>
using Mat = std::vector<std::vector<T>>;

template <class T>
Mat<T> make_mat(int rows, int cols)
{
    return std::vector<std::vector<T>>(rows, std::vector<T>(cols));
}


template <class T>
void print_vector(std::vector<T> v)
{
    std::cout << "[";
    for (int i = 0; i < v.size(); i++)
        std::cout << v[i] << ' ';
    std::cout << "]";
}

template <class T>
void print_matrix(Mat<T> m)
{
    for (int i = 0; i < m.size(); i++)
    {
        print_vector(m[i]);
        std::cout << '\n';
    }
    std::cout << std::endl;
}

template <class T, class U=float>
U euclidean_distance(const std::vector<T> &a, const std::vector<T> &b)
{
    U sum = 0;
    for (size_t i = 0; i < a.size(); i++)
    {
        sum += pow(a[i]-b[i], 2);
    }
    return (U)sqrt(sum);
}

template <class T, class U=float>
U magnitude(const std::vector<T> &v)
{
    U sum = 0;
    for (int i = 0; i < v.size(); i++)
    {
        sum += pow(v[i], 2);
    }
    return (U)sqrt(sum);
}

template <class T, class U=float>
U cosine_distance(const std::vector<T> &a, const std::vector<T> &b)
{
    U dividend = 0;
    for (int i = 0; i < a.size(); i++)
    {
        dividend += a[i] * b[i];
    }

    U divisor = magnitude<T>(a) * magnitude<T>(b);

    // 1 - cosine similarity
    return 1 - (dividend / divisor);
}

template <class T>
Mat<float> compute_train_distances(const Mat<T> &features, distance_function distance=euclidean_distance<float>)
{
    Mat<float> distances = make_mat<float>(features.size(), features.size());
    for (int i = 0; i < features.size(); i++)
        distances[i][i] = 0;

    for (int i = 0; i < features.size() - 1; i++)
    {
        for (int j = i + 1; j < features.size(); j++)
        {
            distances[i][j] = distances[j][i] = distance(features[i], features[j]);
        }
    }

    return distances;
}

template <class T>
Mat<float> compute_test_distances(const Mat<T> &train_data, const Mat<T> &test_data, distance_function distance=euclidean_distance<float>)
{
    Mat<float> distances = make_mat<float>(test_data.size(), train_data.size());

    for (int i = 0; i < test_data.size(); i++)
    {
        for (int j = 0; j < train_data.size(); j++)
        {
            distances[i][j] = distance(test_data[i], train_data[j]);
        }
    }

    return distances;
}

/*****************************************/


}

#endif
