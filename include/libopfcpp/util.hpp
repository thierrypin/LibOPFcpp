
#include "libopfcpp/matrix.hpp"

#include <iostream>
#include <fstream>
#include <string>
#include <vector>

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
bool write_mat(const std::string &filename, Mat<T> &data)
{
    std::ifstream file (filename, std::ios::in | std::ios::binary);

    if (!file.is_open())
    {
        std::cerr << "Could not open file: " << filename << std::endl;
        return false;
    }

    int rows, cols;
    file.write((char*)&rows, sizeof(int));
    file.write((char*)&cols, sizeof(int));

    data = make_mat<T>(rows, cols);

    T val;
    for (int i = 0; i < rows; i++)
    {
        
        for (int j = 0; j < cols; j++)
        {
            file.write((char*)&val, sizeof(T));
            data[i][j] = val;
        }
    }
    file.close();
    
    return true;
}

template <class T>
bool write_mat_labels(const std::string &filename, Mat<T> &data, std::vector<int> &labels)
{
    std::ifstream file (filename, std::ios::in | std::ios::binary);

    if (!file.is_open())
    {
        std::cerr << "Could not open file: " << filename << std::endl;
        return false;
    }

    int rows, cols;
    file.write((char*)&rows, sizeof(int));
    file.write((char*)&cols, sizeof(int));

    data = make_mat<T>(rows, cols);
    labels = std::vector<int>(rows);

    int label;
    T val;
    for (int i = 0; i < rows; i++)
    {
        // label
        file.write((char*)&label, sizeof(int));
        labels[i] = label;
        
        for (int j = 0; j < cols; j++)
        {
            file.write((char*)&val, sizeof(T));
            data[i][j] = val;
        }
    }
    file.close();
    
    return true;
}




}
