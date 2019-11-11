/******************************************************
 * A C++ program for the OPF classification machine,  *
 * all contained in a single header file.             *
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

#ifndef OPF_HPP
#define OPF_HPP

#include <functional>
#include <stdexcept>
#include <algorithm>
#include <typeinfo>
#include <sstream>
#include <utility>
#include <cstring>
#include <string>
#include <limits>
#include <memory>
#include <vector>
#include <cmath>
#include <map>
#include <set>

#include <omp.h>

namespace opf
{

using uchar = unsigned char;
#define INF std::numeric_limits<float>::infinity()
#define NIL -1

// Generic distance function
template <class T>
using distance_function = std::function<T (const T*, const T*, size_t)>;



/*****************************************/
/*************** Binary IO ***************/
/*****************************************/

////////////
// OPF types
enum Type : unsigned char
{
    Classifier = 1,
    Clustering = 2,
};

//////////////////////
// Serialization Flags
enum SFlags : unsigned char
{
    Sup_SavePrototypes = 1,
    Unsup_Anomaly = 2,
};

///////////////
// IO functions
template <class T>
void write_bin(std::ostream& output, const T& val)
{
    output.write((char*) &val, sizeof(T));
}

template <class T>
void write_bin(std::ostream& output, const T* val, int n=1)
{
    output.write((char*) val, sizeof(T) * n);
}

template <class T>
T read_bin(std::istream& input)
{
    T val;
    input.read((char*) &val, sizeof(T));
    return val;
}

template <class T>
void read_bin(std::istream& input, T* val, int n=1)
{
    input.read((char*) val, sizeof(T) * n);
}


/*****************************************/
/************** Matrix type **************/
/*****************************************/
template <class T=float>
class Mat
{
protected:
    std::shared_ptr<T> data;
public:
    int rows, cols;
    int size;
    Mat();
    Mat(const Mat<T>& other);
    Mat(size_t rows, size_t cols);
    Mat(size_t rows, size_t cols, T val);
    Mat(std::shared_ptr<T>& data, size_t rows, size_t cols);
    Mat(T* data, size_t rows, size_t cols);

    virtual T& at(size_t i, size_t j);
    const virtual T at(size_t i, size_t j) const;
    virtual T* row(size_t i);
    const virtual T* row(size_t i) const;
    virtual T* operator[](size_t i);
    const virtual T* operator[](size_t i) const;
    Mat<T>& operator=(const Mat<T>& other);
    virtual Mat<T> copy();

    void release();
};

template <class T>
Mat<T>::Mat()
{
    this->rows = this->cols = this-> size = 0;
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
Mat<T>::Mat(size_t rows, size_t cols)
{
    this->rows = rows;
    this->cols = cols;
    this->size = rows * cols;
    this->data = std::shared_ptr<T>(new T[this->size], std::default_delete<T[]>());
}

template <class T>
Mat<T>::Mat(size_t rows, size_t cols, T val)
{
    this->rows = rows;
    this->cols = cols;
    this->size = rows * cols;
    this->data = std::shared_ptr<T>(new T[this->size], std::default_delete<T[]>());

    for (size_t i = 0; i < rows; i++)
    {
        T* row = this->row(i);
        for (size_t j = 0; j < cols; j++)
            row[j] = val;
    }
}

template <class T>
Mat<T>::Mat(std::shared_ptr<T>& data, size_t rows, size_t cols)
{
    this->rows = rows;
    this->cols = cols;
    this->size = rows * cols;
    this->data = data;
}

// Receives a pointer to some data, which may not be deleted.
template <class T>
Mat<T>::Mat(T* data, size_t rows, size_t cols)
{
    this->rows = rows;
    this->cols = cols;
    this->size = rows * cols;
    this->data = std::shared_ptr<T>(data, [](T *p) {});
}

template <class T>
T& Mat<T>::at(size_t i, size_t j)
{
    size_t idx = i * this->cols + j;
    return this->data.get()[idx];
}

template <class T>
const T Mat<T>::at(size_t i, size_t j) const
{
    size_t idx = i * this->cols + j;
    return this->data.get()[idx];
}

template <class T>
T* Mat<T>::row(size_t i)
{
    size_t idx = i * this->cols;
    return &this->data.get()[idx];
}

template <class T>
const T* Mat<T>::row(size_t i) const
{
    size_t idx = i * this->cols;
    return &this->data.get()[idx];
}

template <class T>
T* Mat<T>::operator[](size_t i)
{
    size_t idx = i * this->cols;
    return &this->data.get()[idx];
}

template <class T>
const T* Mat<T>::operator[](size_t i) const
{
    size_t idx = i * this->cols;
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
    for (size_t i = 0; i < this->rows; i++)
        for (size_t j = 0; j < this->cols; j++)
            out[i][j] = this->at(i, j);
    
    return out;
}

template <class T>
void Mat<T>::release()
{
    this->data.reset();
}

/*****************************************/


// Default distance function
template <class T>
T euclidean_distance(const T* a, const T* b, size_t size)
{
    T sum = 0;
    for (size_t i = 0; i < size; i++)
    {
        sum += (a[i]-b[i]) * (a[i]-b[i]);
    }
    return (T)sqrt(sum);
}

template <class T>
T magnitude(const T* v, size_t size)
{
    T sum = 0;
    for (size_t i = 0; i < size; i++)
    {
        sum += v[i] * v[i];
    }
    return (T)sqrt(sum);
}

// One alternate distance function
template <class T>
T cosine_distance(const T* a, const T* b, size_t size)
{
    T dividend = 0;
    for (size_t i = 0; i < size; i++)
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

    #pragma omp parallel for shared(features, distances)
    for (int i = 0; i < features.rows - 1; i++)
    {
        distances[i][i] = 0;
        for (int j = i + 1; j < features.rows; j++)
        {
            distances[i][j] = distances[j][i] = distance(features[i], features[j], features.cols);
        }
    }

    return distances;
}


/*****************************************/
/********* Distance matrix type **********/
/*****************************************/
// Instead of storing n x n elements, we only store the upper triangle,
// which has (n * (n-1))/2 elements (less than half).
template <class T>
class DistMat: public Mat<T>
{
private:
    T diag_vals = static_cast<T>(0);
    int get_index(int i, int j) const;
public:
    DistMat(){this->rows = this->cols = this->size = 0;};
    DistMat(const DistMat& other);
    DistMat(const Mat<T>& features, distance_function<T> distance=euclidean_distance<T>);
    virtual T& at(size_t i, size_t j);
    const virtual T at(size_t i, size_t j) const;
};

// The first row has n-1 cols, the second has n-2, and so on until row n has 0 cols.
// This way,
#define SWAP(a, b) (((a) ^= (b)), ((b) ^= (a)), ((a) ^= (b)))
template <class T>
inline int DistMat<T>::get_index(int i, int j) const
{
    if (i > j)
        SWAP(i, j);
    return ((((this->rows<<1) - i - 1) * i) >> 1) + (j - i - 1);
}

template <class T>
DistMat<T>::DistMat(const DistMat& other)
{
    this->rows = other.rows;
    this->cols = other.cols;
    this->size = other.size;
    this->data = other.data;
}

template <class T>
DistMat<T>::DistMat(const Mat<T>& features, distance_function<T> distance)
{
    this->rows = features.rows;
    this->cols = features.rows;
    this->size = (this->rows * (this->rows - 1)) / 2;
    this->data = std::shared_ptr<T>(new float[this->size], std::default_delete<float[]>());
    for (int i = 0; i < this->rows; i++)
    {
        for (int j = i+1; j < this->rows; j++)
            this->data.get()[get_index(i, j)] = distance(features[i], features[j], features.cols);
    }
}

template <class T>
T& DistMat<T>::at(size_t i, size_t j)
{
    if (i == j)
        return this->diag_vals = static_cast<T>(0);
    return this->data.get()[this->get_index(i, j)];
}

template <class T>
const T DistMat<T>::at(size_t i, size_t j) const
{
    if (i == j)
        return 0;
    return this->data.get()[this->get_index(i, j)];
}


/*****************************************/
/************ Data structures ************/
/*****************************************/

/**
 * Color codes for Prim's algorithm
 */
enum Color{
    WHITE, // New node
    GRAY,  // On the heap
    BLACK  // Already seen
};

/**
 * Plain class to store node information
 */
class Node
{
public:
    Node()
    {
        this->color = WHITE;
        this->pred = -1;
        this->cost = INF;
        this->is_prototype = false;
    }
    
    size_t index;      // Index on the list -- makes searches easier *
    Color color;       // Color on the heap. white: never visiter, gray: on the heap, black: removed from the heap *
    float cost;        // Cost to reach the node
    int true_label;    // Ground truth *
    int label;         // Assigned label
    int pred;          // Predecessor node *
    bool is_prototype; // Whether the node is a prototype *
};

/**
 * Heap data structure to use as a priority queue
 * 
 */
class Heap
{
private:
    std::vector<Node> *nodes; // A reference for the original container vector
    std::vector<Node*> vec;   // A vector of pointers to build the heap upon
    
    static bool compare_element(const Node* lhs, const Node* rhs)
    {
        return lhs->cost >= rhs->cost;
    }

public:
    // Size-constructor
    Heap(std::vector<Node> *nodes, const std::vector<int> &labels)
    {
        this->nodes = nodes;
        size_t n = nodes->size();
        this->vec.reserve(n);
        for (size_t i = 0; i < n; i++)
        {
            (*this->nodes)[i].index = i;
            (*this->nodes)[i].true_label = (*this->nodes)[i].label = labels[i];
        }
    }
    // Insert new element into heap
    void push(int item, float cost)
    {
        // Update node's cost value
        (*this->nodes)[item].cost = cost;

        // Already on the heap
        if ((*this->nodes)[item].color == GRAY)
            make_heap(this->vec.begin(), this->vec.end(), compare_element); // Remake the heap
        
        // New to the heap
        else if ((*this->nodes)[item].color == WHITE)
        {
            (*this->nodes)[item].color = GRAY;
            this->vec.push_back(&(*this->nodes)[item]);
            push_heap(this->vec.begin(), this->vec.end(), compare_element); // Push new item to the heap
        }
        // Note that black items can not be inserted into the heap
    }

    // Update item's cost without updating the heap
    void update_cost(int item, float cost)
    {
        // Update node's cost value
        (*this->nodes)[item].cost = cost;
        if ((*this->nodes)[item].color == WHITE)
        {
            (*this->nodes)[item].color = GRAY;
            this->vec.push_back(&(*this->nodes)[item]);
        }
    }

    // Update the heap.
    // This is used after multiple calls to update_cost in order to reduce the number of calls to make_heap.
    void heapify()
    {
        make_heap(this->vec.begin(), this->vec.end(), compare_element); // Remake the heap
    }

    // Remove and return the first element of the heap
    int pop()
    {
        // Obtain and mark the first element
        Node *front = this->vec.front();
        front->color = BLACK;
        // Remove it from the heap
        pop_heap(this->vec.begin(), this->vec.end(), compare_element);
        this->vec.pop_back();
        // And return it
        return front->index;
    }

    bool empty()
    {
        return this->vec.size() == 0;
    }

    size_t size()
    {
        return this->vec.size();
    }
};

/*****************************************/



/*****************************************/
/****************** OPF ******************/
/*****************************************/

/******** Supervised ********/
template <class T=float>
class SupervisedOPF
{
private:
    // Model
    Mat<T> train_data; // Training data (original vectors or distance matrix)
    std::vector<Node> nodes; // Learned model
    std::vector<int> ordered_nodes; // List of nodes ordered by cost. Useful for speeding up classification

    // Options
    bool precomputed;
    distance_function<T> distance;

    void prim_prototype(const std::vector<int> &labels);


public:
    SupervisedOPF(bool precomputed=false, distance_function<T> distance=euclidean_distance<T>);
    
    void fit(const Mat<T> &train_data, const std::vector<int> &labels);
    std::vector<int> predict(const Mat<T> &test_data);

    // Serialization functions
    std::string serialize(uchar flags=0);
    static SupervisedOPF<T> unserialize(std::string& contents);

    // Training information
    std::vector<std::vector<float>> get_prototypes();
};

template <class T>
SupervisedOPF<T>::SupervisedOPF(bool precomputed, distance_function<T> distance)
{
    this->precomputed = precomputed;
    this->distance = distance;
}

/**
 * - The first step in OPF's training procedure. Finds the prototype nodes using Prim's
 * Minimum Spanning Tree algorithm.
 * - Any node with an adjacent node of a different class is taken as a prototype.
 */
template <class T>
void SupervisedOPF<T>::prim_prototype(const std::vector<int> &labels)
{
    this->nodes = std::vector<Node>(this->train_data.rows);
    Heap h(&this->nodes, labels); // Heap as a priority queue

    // Arbitrary first node
    h.push(0, 0);

    while(!h.empty())
    {
        // Gets the head of the heap and marks it black
        size_t s = h.pop();

        // Prototype definition
        int pred = this->nodes[s].pred;
        if (pred != NIL)
        {
            // Find points in the border between two classes...
            if (this->nodes[s].true_label != this->nodes[pred].true_label)
            {
                // And set them as prototypes
                this->nodes[s].is_prototype = true;
                this->nodes[pred].is_prototype = true;
            }
        }
        

        // Edge selection
        #pragma omp parallel for default(shared)
        for (size_t t = 0; t < this->nodes.size(); t++)
        {
            // If nodes are different and t has not been poped out of the heap (marked black)
            if (s != t && this->nodes[t].color != BLACK) // TODO if s == t, t is black
            {
                // Compute weight
                float weight;
                if (this->precomputed)
                    weight = this->train_data[s][t];
                else
                    weight = this->distance(this->train_data[s], this->train_data[t], this->train_data.cols);
                
                // Assign if smaller than current value
                if (weight < this->nodes[t].cost)
                {
                    this->nodes[t].pred = static_cast<int>(s);
                    // h.push(t, weight);
                    #pragma omp critical(updateHeap)
                    h.update_cost(t, weight);
                }
            }
        }
        h.heapify();
    }
}

/**
 * Trains the model with the given data and labels.
 * 
 * Inputs:
 *  - train_data:
 *    - original feature vectors [n_samples, n_features] -- if precomputed == false
 *    - distance matrix          [n_samples, n_samples]  -- if precomputed == true
 *  - labels:
 *    - true label values        [n_samples]
 */
template <class T>
void SupervisedOPF<T>::fit(const Mat<T> &train_data, const std::vector<int> &labels)
{
    if ((size_t)train_data.rows != labels.size())
        throw std::invalid_argument("[OPF/fit] Error: data size does not match labels size: " + std::to_string(train_data.rows) + " x " + std::to_string(labels.size()));

    // Store data reference for testing
    this->train_data = train_data;

    // Initialize model
    this->prim_prototype(labels); // Find prototypes
    Heap h(&this->nodes, labels); // Heap as a priority queue

    // Initialization
    for (Node& node: this->nodes)
    {
        node.color = WHITE;
        // Prototypes cost 0, have no predecessor and populate the heap
        if (node.is_prototype)
        {
            node.pred = NIL;
            node.cost = 0;
        }
        else // Other nodes start with cost = INF
        {
            node.cost = INF;
        }
        // Since all nodes are connected to all the others
        h.push(node.index, node.cost);
    }

    // List of nodes ordered by cost
    // Useful for speeding up classification
    this->ordered_nodes.reserve(this->nodes.size());

    // Consume the queue
    while(!h.empty())
    {
        int s = h.pop();
        this->ordered_nodes.push_back(s);

        // Iterate over all neighbors
        #pragma omp parallel for default(shared)
        for (int t = 0; t < (int) this->nodes.size(); t++)
        {
            if (s != t && this->nodes[s].cost < this->nodes[t].cost) // && this->nodes[t].color != BLACK ??
            {
                // Compute weight
                float weight;
                if (precomputed)
                    weight = this->train_data[s][t];
                else
                    weight = distance(this->train_data[s], this->train_data[t], this->train_data.cols);

                float cost = std::max(weight, this->nodes[s].cost);
                if (cost < this->nodes[t].cost)
                {
                    this->nodes[t].pred = s;
                    this->nodes[t].label = this->nodes[s].true_label;
                    // h.push(t, cost);
                    #pragma omp critical(updateHeap)
                    h.update_cost(t, cost);
                }
            }
        }
        h.heapify();
    }
}

/**
 * Classify a set of samples using a model trained by SupervisedOPF::fit.
 * 
 * Inputs:
 *  - test_data:
 *    - original feature vectors [n_test_samples, n_features]      -- if precomputed == false
 *    - distance matrix          [n_test_samples, n_train_samples] -- if precomputed == true
 * 
 * Returns:
 *  - predictions:
 *    - a vector<int> of size [n_test_samples] with classification outputs.
 */
template <class T>
std::vector<int> SupervisedOPF<T>::predict(const Mat<T> &test_data)
{
    int n_test_samples = (int) test_data.rows;
    int n_train_samples = (int) this->nodes.size();

    // Output predictions
    std::vector<int> predictions(n_test_samples);

    #pragma omp parallel for default(shared)
    for (int i = 0; i < n_test_samples; i++)
    {
        int idx = this->ordered_nodes[0];
        int min_idx = 0;
        T min_cost = INF;
        T weight = 0;

        // 'ordered_nodes' contains sample indices ordered by cost, so if the current
        // best connection costs less than the next node, it is useless to keep looking.
        for (int j = 0; j < n_train_samples && min_cost > this->nodes[idx].cost; j++)
        {
            // Get the next node in the ordered list
            idx = this->ordered_nodes[j];

            // Compute its distance to the query point
            if (precomputed)
                weight = test_data[i][idx];
            else
                weight = distance(test_data[i], this->train_data[idx], this->train_data.cols);

            // The cost corresponds to the max between the distance and the reference cost
            float cost = std::max(weight, this->nodes[idx].cost);

            if (cost < min_cost)
            {
                min_cost = cost;
                min_idx = idx;
            }
        }

        predictions[i] = this->nodes[min_idx].label;
    }
    
    return predictions;
}

/*****************************************/
/*              Persistence              */
/*****************************************/

template <class T>
std::string SupervisedOPF<T>::serialize(uchar flags)
{
    if (this->precomputed)
        throw std::invalid_argument("Serialization for precomputed OPF not implemented yet");
    // Open file
    std::ostringstream output(std::ios::out | std::ios::binary);

    int n_samples = this->train_data.rows;
    int n_features = this->train_data.cols;
    
    // Header
    write_bin<char>(output, "OPF", 3);
    write_bin<uchar>(output, Type::Classifier);
    write_bin<uchar>(output, flags);
    write_bin<uchar>(output, static_cast<uchar>(0)); // Reserved flags byte
    write_bin<int>(output, n_samples);
    write_bin<int>(output, n_features);

    // Data
    int size = this->train_data.size;
    T* data = this->train_data.row(0);
    write_bin<T>(output, data, size);

    // Nodes
    for (int i = 0; i < n_samples; i++)
    {
        write_bin<float>(output, this->nodes[i].cost);
        write_bin<int>(output, this->nodes[i].label);
    }

    // Ordered_nodes
    write_bin<int>(output, this->ordered_nodes.data(), n_samples);

    // Prototypes
    if (flags & SFlags::Sup_SavePrototypes)
    {
        // Find which are prototypes first, because we need the correct amount
        std::set<int> prots;
        for (int i = 0; i < n_samples; i++)
        {
            if (this->nodes[i].is_prototype)
                prots.insert(i);
        }

        write_bin<int>(output, prots.size());
        for (auto it = prots.begin(); it != prots.end(); ++it)
            write_bin<int>(output, *it);
    }

    return output.str();
}

template <class T>
SupervisedOPF<T> SupervisedOPF<T>::unserialize(std::string& contents)
{
    // Header
    int n_samples;
    int n_features;

    char header[4];

    SupervisedOPF<float> opf;

    // Open stream
    std::istringstream ifs(contents); // , std::ios::in | std::ios::binary

    // Check if stream is an OPF serialization
    read_bin<char>(ifs, header, 3);
    header[3] = '\0';
    if (strcmp(header, "OPF"))
        throw std::invalid_argument("Input is not an OPF serialization");
    
    // Get type and flags
    uchar type = read_bin<uchar>(ifs);
    uchar flags = read_bin<uchar>(ifs);
    read_bin<uchar>(ifs); // Reserved byte

    if (type != Type::Classifier)
        throw std::invalid_argument("Input is not a Supervised OPF serialization");

    n_samples = read_bin<int>(ifs);
    n_features = read_bin<int>(ifs);

    // Data
    int size = n_samples * n_features;
    opf.train_data = Mat<T>(n_samples, n_features);
    T* data = opf.train_data.row(0);
    read_bin<T>(ifs, data, size);

    // Nodes
    opf.nodes = std::vector<Node>(n_samples);
    for (int i = 0; i < n_samples; i++)
    {
        opf.nodes[i].cost = read_bin<float>(ifs);
        opf.nodes[i].label = read_bin<int>(ifs);
    }

    // Ordered_nodes
    opf.ordered_nodes = std::vector<int>(n_samples);
    read_bin<int>(ifs, opf.ordered_nodes.data(), n_samples);

    if (flags & SFlags::Sup_SavePrototypes)
    {
        int prots = read_bin<int>(ifs);
        for (int i = 0; i < prots; i++)
        {
            int idx = read_bin<int>(ifs);
            opf.nodes[idx].is_prototype = true;
        }
    }

    return opf;
}

/*****************************************/
/*              Data Access              */
/*****************************************/

template <class T>
std::vector<std::vector<float>> SupervisedOPF<T>::get_prototypes()
{
    std::set<int> prots;
    for (int i = 0; i < this->train_data.rows; i++)
    {
        if (this->nodes[i].is_prototype)
            prots.insert(i);
    }

    std::vector<std::vector<float>> out(prots.size(), std::vector<float>(this->train_data.cols));
    int i = 0;
    for (auto it = prots.begin(); it != prots.end(); ++it, ++i)
    {
        for (int j = 0; j < this->train_data.cols; j++)
        {
            out[i][j] = this->train_data[*it][j];
        }
    }

    return out;
}

/*****************************************/

/******** Unsupervised ********/

// Index + distance to another node
using Pdist = std::pair<int, float>;

static bool compare_neighbor(const Pdist& lhs, const Pdist& rhs)
{
    return lhs.second < rhs.second;
}

// Aux class to find the k nearest neighbors from a given node
// In the future, this should be replaced by a kdtree
class BestK
{
private:
    int k;
    std::vector<Pdist> heap; // idx, dist

public:
    // Empty initializer
    BestK(int k) : k(k) {this->heap.reserve(k);}
    // Tries to insert another element to the heap
    void insert(int idx, float dist)
    {
        if (heap.size() < static_cast<unsigned int>(this->k))
        {
            heap.push_back(Pdist(idx, dist));
            push_heap(this->heap.begin(), this->heap.end(), compare_neighbor);
        }
        else
        {
            // If the new point is closer than the farthest neighbor
            Pdist farthest = this->heap.front();
            if (dist < farthest.second)
            {
                // Remove one from the heap and add the other
                pop_heap(this->heap.begin(), this->heap.end(), compare_neighbor);
                this->heap[this->k-1] = Pdist(idx, dist);
                push_heap(this->heap.begin(), this->heap.end(), compare_neighbor);
            }
        }
    }

    std::vector<Pdist>& get_knn() { return heap; }
};


/**
 * Plain class to store node information
 */
class NodeKNN
{
public:
    NodeKNN()
    {
        this->pred = -1;
    }
    
    std::set<Pdist> adj; // Node adjacency
    size_t index;        // Index on the list -- makes searches easier
    int label;           // Assigned label
    int pred;            // Predecessor node
    float value;         // Path value
    float rho;           // probability density function
};

// Unsupervised OPF classifier
template <class T=float>
class UnsupervisedOPF
{
private:
    // Model
    std::shared_ptr<const Mat<T>> train_data;   // Training data (original vectors or distance matrix)
    distance_function<T> distance; // Distance function
    std::vector<NodeKNN> nodes;    // Learned model
    std::vector<int> queue;        // Priority queue implemented as a linear search in a vector
    int k;                         // The number of neighbors to build the graph
    int n_clusters;                // Number of clusters in the model -- computed during fit

    // Training attributes
    float sigma_sq;             // Sigma squared, used to compute probability distribution function
    float delta;                // Adjustment term
    float denominator;          // sqrt(2 * math.pi * sigma_sq) -- compute it only once

    // Options
    float thresh;
    bool anomaly;
    bool precomputed;

    // Queue capabilities
    int get_max();

    // Training subroutines
    void build_graph();
    void build_initialize();
    void cluster();

public:
    UnsupervisedOPF(int k=5, bool precomputed=false, bool anomaly=false, float thresh=1., distance_function<T> distance=euclidean_distance<T>);
    
    void fit(const Mat<T> &train_data);
    std::vector<int> fit_predict(const Mat<T> &train_data);
    std::vector<int> predict(const Mat<T> &test_data);

    void find_best_k(Mat<float>& train_data, int kmin, int kmax, int step=1, bool precompute=true);

    // Clustering info
    float quality_metric();

    // Getters
    int get_n_clusters() {return this->n_clusters;}
    int get_k() {return this->k;}

    // Serialization functions
    std::string serialize(uchar flags=0);
    static UnsupervisedOPF<T> unserialize(std::string& contents);
};

template <class T>
UnsupervisedOPF<T>::UnsupervisedOPF(int k, bool precomputed, bool anomaly, float thresh, distance_function<T> distance)
{
    this->k = k;
    this->precomputed = precomputed;
    this->anomaly = anomaly;
    if (this->anomaly)
        this->n_clusters = 2;
    this->thresh = thresh;
    this->distance = distance;
}

// Builds the KNN graph
template <class T>
void UnsupervisedOPF<T>::build_graph()
{
    // Proportional to the length of the biggest edge
    for (size_t i = 0; i < this->nodes.size(); i++)
    {
        // Find the k nearest neighbors
        BestK bk(this->k);
        for (size_t j = 0; j < this->nodes.size(); j++)
        {
            if (i != j)
            {
                float dist;
                if (this->precomputed)
                    dist = this->train_data->at(i, j);
                else
                    dist = this->distance(this->train_data->row(i), this->train_data->row(j), this->train_data->cols);
                
                bk.insert(j, dist);
            }
        }

        this->sigma_sq = 0.;
        std::vector<Pdist> knn = bk.get_knn();
        for (auto it = knn.cbegin(); it != knn.cend(); ++it)
        {
            // Since the graph is undirected, make connections from both nodes
            this->nodes[i].adj.insert(*it);
            this->nodes[it->first].adj.insert(Pdist(i, it->second));

            // Finding sigma
            if (it->second > this->sigma_sq)
                this->sigma_sq = it->second;
        }
    }
    
    this->sigma_sq /= 3;
    this->sigma_sq = this->sigma_sq * this->sigma_sq;
    this->denominator = sqrt(2 * M_PI * this->sigma_sq);
}

// Build and initialize the graph
template <class T>
void UnsupervisedOPF<T>::build_initialize()
{
    // Precompute during training to speed up the process?
    this->build_graph();

    // Compute rho
    std::set<Pdist>::iterator it;
    for (size_t i = 0; i < this->nodes.size(); i++)
    {
        int n_neighbors = this->nodes[i].adj.size(); // A node may have more than k neighbors
        float div = this->denominator * n_neighbors;
        float sum = 0;

        for (it = this->nodes[i].adj.cbegin(); it != this->nodes[i].adj.cend(); ++it)
        {
            float dist = it->second; // this->distances[i][*it]
            sum += expf((-dist * dist) / (2 * this->sigma_sq));
        }

        this->nodes[i].rho = sum / div;
    }

    // Compute delta
    this->delta = INF;
    for (size_t i = 0; i < this->nodes.size(); i++)
    {
        for (it = this->nodes[i].adj.begin(); it != this->nodes[i].adj.end(); ++it)
        {
            float diff = abs(this->nodes[i].rho - this->nodes[it->first].rho);
            if (this->delta > diff)
                this->delta = diff;
        }
    }

    // And, finally, initialize each node
    this->queue.resize(this->nodes.size());
    for (size_t i = 0; i < this->nodes.size(); i++)
    {
        this->nodes[i].value = this->nodes[i].rho - this->delta;
        this->queue[i] = static_cast<int>(i);
    }
}

// Get the node with the biggest path value
template <class T>
int UnsupervisedOPF<T>::get_max()
{
    float maxval = -INF;
    int maxidx = -1;
    int size = this->queue.size();
    for (int i = 0; i < size; i++)
    {
        int idx = this->queue[i];
        if (this->nodes[idx].value > maxval)
        {
            maxidx = i;
            maxval = this->nodes[idx].value;
        }
    }

    int best = this->queue[maxidx];
    int tmp = this->queue[size-1];
    this->queue[size-1] = this->queue[maxidx];
    this->queue[maxidx] = tmp;
    this->queue.pop_back();

    return best;
}

// OPF clustering
template <class T>
void UnsupervisedOPF<T>::cluster()
{
    // Cluster labels
    int l = 0;
    // Priority queue
    while (!this->queue.empty())
    {
        int s = this->get_max(); // Pop the highest value

        // If it has no predecessor, make it a prototype
        if (this->nodes[s].pred == -1)
        {
            this->nodes[s].label = l++;
            this->nodes[s].value = this->nodes[s].rho;
        }

        // Iterate and conquer over its neighbors
        for (auto it = this->nodes[s].adj.begin(); it != this->nodes[s].adj.end(); ++it)
        {
            int t = it->first;
            if (this->nodes[t].value < this->nodes[s].value)
            {
                float tmp = std::min(this->nodes[s].value, this->nodes[t].rho);
                if (tmp > this->nodes[t].value)
                {
                    this->nodes[t].label = this->nodes[s].label;
                    this->nodes[t].pred = s;
                    this->nodes[t].value = tmp;
                }
            }
        }
    }

    this->n_clusters = l;
}


// Fit the model
template <class T>
void UnsupervisedOPF<T>::fit(const Mat<T> &train_data)
{
    this->train_data = std::shared_ptr<const Mat<T>>(&train_data, [](const Mat<T> *p) {});
    this->nodes = std::vector<NodeKNN>(this->train_data->rows);
    this->build_initialize();
    if (!this->anomaly)
        this->cluster();
}

// Fit and predict for all nodes
template <class T>
std::vector<int> UnsupervisedOPF<T>::fit_predict(const Mat<T> &train_data)
{
    this->fit(train_data);

    std::vector<int> labels(this->nodes.size());

    if (this->anomaly)
        for (size_t i = 0; i < this->nodes.size(); i++)
            labels[i] = (this->nodes[i].rho < this->thresh) ? 1 : 0;
    else
        for (size_t i = 0; i < this->nodes.size(); i++)
            labels[i] = this->nodes[i].label;
    
    return labels;
}

// Predict cluster pertinence
template <class T>
std::vector<int> UnsupervisedOPF<T>::predict(const Mat<T> &test_data)
{
    std::vector<int> preds(test_data.rows);
    // For each test sample
    for (int i = 0; i < test_data.rows; i++)
    {
        // Find the k nearest neighbors
        BestK bk(this->k);
        for (int j = 0; j < static_cast<int>(this->nodes.size()); j++)
        {
            if (i != j)
            {
                float dist;
                if (this->precomputed)
                    dist = test_data.at(i, j);
                else
                    dist = this->distance(test_data[i], this->train_data->row(j), this->train_data->cols);
                
                bk.insert(j, dist);
            }
        }

        // Compute the testing rho
        std::vector<Pdist> neighbors = bk.get_knn();
        int n_neighbors = neighbors.size();

        float div = this->denominator * n_neighbors;
        float sum = 0;

        for (int j = 0; j < n_neighbors; j++)
        {
            float dist = neighbors[j].second; // this->distances[i][*it]
            sum += expf((-dist * dist) / (2 * this->sigma_sq));
        }

        float rho = sum / div;

        if (this->anomaly)
        {
            // And returns anomaly detection based on graph density
            preds[i] = (rho < this->thresh) ? 1 : 0;
        }
        else
        {
            // And find which node conquers this test sample
            float maxval = 0;
            int maxidx = -1;
            for (int j = 0; j < n_neighbors; j++)
            {
                int s = neighbors[j].first;
                float val = std::min(this->nodes[s].value, rho);
                if (val > maxval)
                {
                    maxval = val;
                    maxidx = s;
                }
            }

            preds[i] = this->nodes[maxidx].label;
        }
    }

    return preds;
}

// Quality metric
// From: A Robust Extension of the Mean Shift Algorithm using Optimum Path Forest
// Leonardo Rocha, Alexandre Falcao, and Luis Meloni
template <class T>
float UnsupervisedOPF<T>::quality_metric()
{
    if (this->anomaly)
        throw std::invalid_argument("Quality metric not implemented for anomaly detection yet");
    std::vector<float> w(this->n_clusters, 0);
    std::vector<float> w_(this->n_clusters, 0);
    for (int i = 0; i < this->train_data->rows; i++)
    {
        int l = this->nodes[i].label;

        for (auto it = this->nodes[i].adj.begin(); it != this->nodes[i].adj.end(); ++it)
        {
            int l_ = this->nodes[it->first].label;
            float tmp = 0;

            if (it->second != 0)
                tmp = 1. / it->second;

            if (l == l_)
                w[l] += tmp;
            else
                w_[l] += tmp;
        }
    }

    float C = 0;
    for (int i = 0; i < this->n_clusters; i++)
        C += w_[i] / (w_[i] + w[i]);

    return C;
}

// Brute force method to find the best value of k
template <class T>
void UnsupervisedOPF<T>::find_best_k(Mat<float>& train_data, int kmin, int kmax, int step, bool precompute)
{
    float best_quality = INF;
    UnsupervisedOPF<float> best_opf;
    DistMat<float> distances;
    if (precompute)
        distances = DistMat<float>(train_data, this->distance);

    for (int k = kmin; k <= kmax; k += step)
    {
        // Instanciate and train the model
        UnsupervisedOPF<float> opf(k, precompute, false, 0, this->distance);
        if (precompute)
            opf.fit(distances);
        else
            opf.fit(train_data);
       
        // Compare its clustering grade
        float quality = opf.quality_metric(); // Normalized cut
        if (quality < best_quality)
        {
            best_quality = quality;
            best_opf = opf;
        }

    }

    if (this->precomputed)
        this->train_data = std::shared_ptr<Mat<T>>(&distances, std::default_delete<Mat<T>>());
    else
        this->train_data = std::shared_ptr<Mat<T>>(&train_data, [](Mat<T> *p) {});

    this->k = best_opf.k;
    this->n_clusters = best_opf.n_clusters;
    this->nodes = best_opf.nodes;
    this->denominator = best_opf.denominator;
    this->sigma_sq = best_opf.sigma_sq;
    this->delta = best_opf.delta;
}

/*****************************************/
/*              Persistence              */
/*****************************************/

template <class T>
std::string UnsupervisedOPF<T>::serialize(uchar flags)
{
    if (this->precomputed)
        throw std::invalid_argument("Serialization for precomputed OPF not implemented yet");

    // Open file
    std::ostringstream output   ;
    int n_samples = this->train_data->rows;
    int n_features = this->train_data->cols;

    // Output flags
    flags = 0; // For now, there are no user-defined flags
    if (this->anomaly)
        flags += SFlags::Unsup_Anomaly;

    // Header
    write_bin<char>(output, "OPF", 3);
    write_bin<uchar>(output, Type::Clustering);
    write_bin<uchar>(output, flags);
    write_bin<uchar>(output, static_cast<uchar>(0)); // Reserved byte
    write_bin<int>(output, n_samples);
    write_bin<int>(output, n_features);

    // Scalar data
    write_bin<int>(output, this->k);
    if (!this->anomaly)
        write_bin<int>(output, this->n_clusters);
    write_bin<float>(output, this->denominator);
    write_bin<float>(output, this->sigma_sq);

    // Data
    int size = this->train_data->size;
    const T* data = this->train_data->row(0);
    write_bin<T>(output, data, size);

    // Nodes
    for (int i = 0; i < n_samples; i++)
    {
        write_bin<float>(output, this->nodes[i].value);
        if (!this->anomaly)
            write_bin<int>(output, this->nodes[i].label);
    }

    return output.str();
}

template <class T>
UnsupervisedOPF<T> UnsupervisedOPF<T>::unserialize(std::string& contents)
{
    UnsupervisedOPF<float> opf;

    // Open stream
    std::istringstream ifs(contents); // , std::ios::in | std::ios::binary

    /// Header
    int n_samples;
    int n_features;
    char header[4];

    // Check if stream is an OPF serialization
    read_bin<char>(ifs, header, 3);
    header[3] = '\0';
    if (strcmp(header, "OPF"))
        throw std::invalid_argument("Input is not an OPF serialization");    

    // Get type and flags
    uchar type = read_bin<uchar>(ifs);
    uchar flags = read_bin<uchar>(ifs); // Flags byte
    read_bin<uchar>(ifs); // reserved byte

    if (flags & SFlags::Unsup_Anomaly)
        opf.anomaly = true;

    if (type != Type::Clustering)
        throw std::invalid_argument("Input is not an Unsupervised OPF serialization");

    // Data size
    n_samples = read_bin<int>(ifs);
    n_features = read_bin<int>(ifs);

    // Scalar data
    opf.k = read_bin<int>(ifs);
    if (!opf.anomaly)
        opf.n_clusters = read_bin<int>(ifs);
    opf.denominator = read_bin<float>(ifs);
    opf.sigma_sq = read_bin<float>(ifs);

    /// Data
    // Temporary var to read data, since opf's train_data is const
    auto train_data = std::shared_ptr<Mat<T>>(new Mat<T>(n_samples, n_features), std::default_delete<Mat<T>>());
    // Read data
    int size = n_samples * n_features;
    T* data = train_data->row(0);
    read_bin<T>(ifs, data, size);
    // Assign to opf
    opf.train_data = train_data;

    // Nodes
    opf.nodes = std::vector<NodeKNN>(n_samples);
    for (int i = 0; i < n_samples; i++)
    {
        opf.nodes[i].value = read_bin<float>(ifs);
        if (!opf.anomaly)
            opf.nodes[i].label = read_bin<int>(ifs);
    }

    return opf;
}

/*****************************************/

// template <class T>
// auto unserialize(std::string& contents)
// {
//     if (contents.find("OPF") != 0)
//         throw std::invalid_argument("Input is not an OPF serialization");
//     else
//     {
//         uchar type = (unsigned char) contents[3];
//         switch (type)
//         {
//             case Type::Classifier:
//                 return SupervisedOPF<T>::unserialize(contents);
//             case Type::Clustering:
//                 return UnsupervisedOPF<T>::unserialize(contents);
//             default:
//                 std::string error("Unknown OPF type: ");
//                 error += std::to_string(type);
//                 throw std::invalid_argument(error);
//         }
//     }

// }

}

#endif
