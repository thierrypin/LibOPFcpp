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

#ifndef MST_HPP
#define MST_HPP

#include <functional>
#include <algorithm>
#include <typeinfo>
#include <fstream>
#include <cstring>
#include <string>
#include <limits>
#include <memory>
#include <vector>
#include <cmath>
#include <map>
#include <omp.h>


namespace opf
{

#define INF std::numeric_limits<float>::infinity()
#define NIL -1

// Generic distance function
template <class T>
using distance_function = std::function<T (const T*, const T*, size_t)>;

using uchar = unsigned char;

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

/*****************************************/
/************** Matrix type **************/
/*****************************************/
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
    Mat(size_t rows, size_t cols);
    Mat(size_t rows, size_t cols, T val);
    Mat(std::shared_ptr<T>& data, size_t rows, size_t cols);
    Mat(T* data, size_t rows, size_t cols);

    T* row(size_t i);
    T& at(size_t i, size_t j);
    T* operator[](size_t i);
    const T* operator[](size_t i) const;
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
	// for (size_t i = 0; i < rows; i++)
    //     for (size_t j = 0; j < cols; j++)
    //         this->at(i, j) = val;
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
T* Mat<T>::row(size_t i)
{
    size_t idx = i * this->cols;
    return &this->data.get()[idx];
}

template <class T>
T& Mat<T>::at(size_t i, size_t j)
{
    size_t idx = i * this->cols + j;
    return this->data.get()[idx];
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
            out[i][j] = this->data[i][j];
}

/*****************************************/

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
	
	float cost;        // Cost to reach the node
	int true_label;    // Ground truth
	int label;         // Assigned label
	size_t index;      // Index on the list -- makes searches easier
	int pred;          // predecessor node
	Color color;       // Color on the heap. white: never visiter, gray: on the heap, black: removed from the heap
	bool is_prototype; // Whether the node is a prototype
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


template <class T=float>
class SupervisedOPF // TODO: PIMPL
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
	bool write(std::string filename);
	static bool read(std::string filename, SupervisedOPF<T> &opf);
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

// TODO AQUI ******************************
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
	{
		std::cerr << "[OPF/fit] Error: data size does not match labels size: " << train_data.rows << " x " << labels.size() << std::endl;
		exit(1);
	}
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
bool SupervisedOPF<T>::write(std::string filename)
{
	// Open file
	std::ofstream output(filename, std::ios::out | std::ios::binary);
	if (!output.is_open())
	{
		std::cerr << "Could not open file: " << filename << std::endl;
		return false;
	}

	int n_samples = this->train_data.rows;
	int n_features = this->train_data.cols;

	// Header
	output.write("OPF", 3*sizeof(char));
	output.write((char*) &n_samples, sizeof(int));
	output.write((char*) &n_features, sizeof(int));
	if(output.bad()) {std::cerr << "Error writing to file: " << filename << std::endl; return false;}
		

	// Data
	int size = this->train_data.size;
	T* data = this->train_data.row(0);
	output.write((char*) data, size * sizeof(T));
	if(output.bad()) {std::cerr << "Error writing to file: " << filename << std::endl; return false;}

	// Nodes
	for (int i = 0; i < n_samples; i++)
	{
		output.write((char*) &this->nodes[i].cost, sizeof(float));
		output.write((char*) &this->nodes[i].label, sizeof(int));

		if(output.bad()) {std::cerr << "Error writing to file: " << filename << std::endl; return false;}
	}

	// Ordered_nodes

	output.write((char*) this->ordered_nodes.data(), n_samples*sizeof(int));
	if(output.bad()) {std::cerr << "Error writing to file: " << filename << std::endl; return false;}

	return true;
}

template <class T>
bool SupervisedOPF<T>::read(std::string filename, SupervisedOPF<T> &opf)
{
	opf = SupervisedOPF<float>();

	// Open file	
	std::ifstream output(filename, std::ios::in | std::ios::binary);
	if (!output.is_open())
	{
		std::cerr << "Could not open file: " << filename << std::endl;
		return false;
	}

	// Header
	int n_samples;
	int n_features;

	char header[4];
	output.read(header, 3*sizeof(char));
	header[3] = '\0';
	if (strcmp(header, "OPF"))
	{std::cerr << "Input is not an OPF file: " << filename << std::endl; return false;}

	output.read((char*) &n_samples, sizeof(int));
	output.read((char*) &n_features, sizeof(int));
	if(output.bad()) {std::cerr << "Error reading file: " << filename << std::endl; return false;}

	// Data
	int size = n_samples * n_features;
	opf.train_data = Mat<T>(n_samples, n_features);
	T* data = opf.train_data.row(0);
	output.read((char*) data, size * sizeof(T));
	if(output.bad()) {std::cerr << "Error reading file: " << filename << std::endl; return false;}

	// Nodes
	opf.nodes = std::vector<Node>(n_samples);
	for (int i = 0; i < n_samples; i++)
	{
		output.read((char*) &opf.nodes[i].cost, sizeof(float));
		output.read((char*) &opf.nodes[i].label, sizeof(int));

		if(output.bad()) {std::cerr << "Error reading file: " << filename << std::endl; return false;}
	}

	// Ordered_nodes
	opf.ordered_nodes = std::vector<int>(n_samples);
	output.read((char*) opf.ordered_nodes.data(), n_samples*sizeof(int));
	if(output.bad()) {std::cerr << "Error reading file: " << filename << std::endl; return false;}

	return true;
}

/*****************************************/

}

#endif
