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

#include <vector>
#include <map>
#include <limits>
#include <algorithm>
#include <cmath>

#include "libopfcpp/matrix.hpp"

namespace opf
{

#define INF std::numeric_limits<float>::infinity()
#define NIL -1


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
	int index;         // Index on the list -- makes searches easier
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
		for (int i = 0; i < n; i++)
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
	distance_function distance;

	void prim_prototype(const std::vector<int> &labels);


public:
	SupervisedOPF(bool precomputed=false, distance_function distance=euclidean_distance<float>)
	{
		this->precomputed = precomputed;
		this->distance = distance;
	}
	
	void fit(const Mat<T> &train_data, const std::vector<int> &labels);
	std::vector<int> predict(const Mat<T> &test_data);
};

/**
 * - The first step in OPF's training procedure. Finds the prototype nodes using Prim's
 * Minimum Spanning Tree algorithm.
 * - Any node with an adjacent node of a different class is taken as a prototype.
 */
template <class T>
void SupervisedOPF<T>::prim_prototype(const std::vector<int> &labels)
{
	this->nodes = std::vector<Node>(this->train_data.size());
	Heap h(&this->nodes, labels); // Heap as a priority queue

	// Arbitrary first node
	h.push(0, 0);

	while(!h.empty())
	{
		// Gets the head of the heap and marks it black
		int s = h.pop();

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
					weight = this->distance(this->train_data[s], this->train_data[t]);
				
				// Assign if smaller than current value
				if (weight < this->nodes[t].cost)
				{
					this->nodes[t].pred = s;
					h.push(t, weight);
				}
			}
		}
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
	if (train_data.size() != labels.size())
	{
		cerr << "Error: data size does not match labels size: " << train_data.size() << " x " << labels.size() << endl;
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
			h.push(node.index, node.cost);
		}
		else // Other nodes start with cost = INF
		{
			node.cost = INF;
		}
		
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
		for (size_t t = 0; t < this->nodes.size(); t++)
		{
			if (s != t && this->nodes[s].cost < this->nodes[t].cost) // && this->nodes[t].color != BLACK ??
			{
				// Compute weight
				float weight;
				if (precomputed)
					weight = this->train_data[s][t];
				else
					weight = distance(this->train_data[s], this->train_data[t]);

				float cost = max(weight, this->nodes[s].cost);
				if (cost < this->nodes[t].cost)
				{
					this->nodes[t].pred = s;
					this->nodes[t].label = this->nodes[s].true_label;
					h.push(t, cost);
				}
			}
		}
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
vector<int> SupervisedOPF<T>::predict(const Mat<T> &test_data)
{
	int n_test_samples = test_data.size();
	int n_train_samples = this->nodes.size();

	// Output predictions
	vector<int> predictions(n_test_samples);

	for (int i = 0; i < n_test_samples; i++)
	{
		int j = 0;
		int idx = this->ordered_nodes[0];
		int min_idx;
		float min_cost = INF;
		float weight = 0;

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
				weight = distance(test_data[i], this->train_data[idx]);

			// The cost corresponds to the max between the distance and the reference cost
			float cost = max(weight, this->nodes[idx].cost);

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

}

#endif
