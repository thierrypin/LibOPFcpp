/**
 * Matrix class with reference-counted data vector.
 * 
 * It was weakly inspired by OpenCV's Mat class, although
 * only by behaviour -- I did not look into the code.
 * Also, OpenCV's class is much more complete.
 * 
 * Author: Thierry Moreira, 2019
 * 
 */

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

#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <functional>
#include <iostream>
#include <memory>

namespace opf
{


/*****************************************/
/************** Matrix type **************/
/*****************************************/


// template <class T=float>
// class ArrayDeleter
// {
//   void operator ()( T const * p)
//   { 
//     delete[] p; 
//   }
// };

template <class T=float>
class Mat
{
private:
    std::shared_ptr<T> data;
public:
    int rows, cols;
    int size;
    Mat();
    Mat(Mat<T>& other);
    Mat(const Mat<T>& other);
    Mat(int rows, int cols);
    Mat(int rows, int cols, T val);
    Mat(std::shared_ptr<T>& data, int rows, int cols);
    Mat(T* data, int rows, int cols);

    T* row(int i);
    T& at(int i, int j);
    T* operator[](int i);
    const T* operator[](int i) const;
    Mat<T>& operator=(const Mat<T>& other);
    Mat<T> copy();
};

template <class T>
Mat<T>::Mat()
{
    this->rows = this->cols = this-> size = 0;
}

template <class T>
Mat<T>::Mat(Mat<T>& other)
{
    this->rows = other.rows;
    this->cols = other.cols;
    this->size = other.size;
    this->data = other.data;
}

template <class T>
Mat<T>::Mat(const Mat<T>& other)
{
    this->rows = other.rows;
    this->cols = other.cols;
    this->size = other.size;
    this->data = other.data;
}

template <class T>
Mat<T>::Mat(int rows, int cols)
{
    this->rows = rows;
    this->cols = cols;
    this->size = rows * cols;
    this->data = std::shared_ptr<T>(new T[this->size], std::default_delete<T[]>());
}

template <class T>
Mat<T>::Mat(int rows, int cols, T val)
{
    this->rows = rows;
    this->cols = cols;
    this->size = rows * cols;
    this->data = std::shared_ptr<T>(new T[this->size], std::default_delete<T[]>());

    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            this->at(i, j) = val;
}

template <class T>
Mat<T>::Mat(std::shared_ptr<T>& data, int rows, int cols)
{
    this->rows = rows;
    this->cols = cols;
    this->size = rows * cols;
    this->data = data;
}

template <class T>
Mat<T>::Mat(T* data, int rows, int cols)
{
    this->rows = rows;
    this->cols = cols;
    this->size = rows * cols;
    this->data = std::shared_ptr<T>(data, [](T *p) {});
}

template <class T>
T* Mat<T>::row(int i)
{
    int idx = i * this->cols;
    return &this->data.get()[idx];
}

template <class T>
T& Mat<T>::at(int i, int j)
{
    int idx = i * this->cols + j;
    return this->data.get()[idx];
}

template <class T>
T* Mat<T>::operator[](int i)
{
    int idx = i * this->cols;
    return &this->data.get()[idx];
}

template <class T>
const T* Mat<T>::operator[](int i) const
{
    int idx = i * this->cols;
    return &this->data.get()[idx];
}

template <class T>
Mat<T>& Mat<T>::operator=(const Mat<T>& other)
{
    this->rows = other.rows;
    this->cols = other.cols;
    this->size = other.size;
    this->data = other.data;

    return *this;
}

template <class T>
Mat<T> Mat<T>::copy()
{
    Mat<T> out(this->rows, this->cols);
    for (int i = 0; i < this->rows; i++)
        for (int j = 0; j < this->cols; j++)
            out[i][j] = this->data[i][j];
}

/*****************************************/


}

#endif
