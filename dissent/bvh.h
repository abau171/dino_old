#pragma once

#include <vector>

#include "geometry.h"

struct indexed_aabb_t {
	aabb_t aabb;
	int index;
};

struct bvh_node_t {

	bool is_leaf;
	aabb_t bound;

	// If inner node, i0 and i1 are indices of left and right child nodes.
	// If leaf node, i0 and i1 are start index and end index of objects in the BVH.
	int i0, i1;

};

void buildBVH(std::vector<indexed_aabb_t>& bounds, std::vector<bvh_node_t>& bvh, std::vector<int>& indices);
