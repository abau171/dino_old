#pragma once

#include <vector>

#include "geometry.h"

#define BVH_LEAF_MASK 0x80000000
#define BVH_I1_MASK 0x7fffffff

/*
The code in this module is heavily based on the Ray Tracey blog's GPU Path
Tracing tutorial. Specifically, parts 3 and 4 discuss GPU BVH construction and
use in great detail.
http://raytracey.blogspot.com/2016/01/gpu-path-tracing-tutorial-3-take-your.html
http://raytracey.blogspot.com/2016/09/gpu-path-tracing-tutorial-4-optimised.html
*/

/*
An axis-aligned bounding box with an index used for final sorting.
*/
struct indexed_aabb_t {
	aabb_t aabb;
	int index;
};

/*
A node of a GPU-optimized Bounding Volume Heirarchy.
The most significant bit of the i1 integer determines if the node is a leaf or
inner node.
*/
struct bvh_node_t {

	aabb_t bound;

	// If inner node, i0 and i1 are indices of left and right child nodes.
	// If leaf node, i0 and i1 are start index and end index of objects in the BVH.
	int i0, i1;

};

/*
Construct a new BVH given a vector of boundaries, and load the BVH into a flat
vector which uses no direct pointers (good for GPU use).
*/
void buildBVH(std::vector<indexed_aabb_t>& bounds, std::vector<bvh_node_t>& bvh, std::vector<int>& indices);
