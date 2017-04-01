#include <iostream>
#include <vector>

#include "bvh.h"


struct bvh_construction_node_t {
	bool is_leaf;
	aabb_t bound;
	std::vector<indexed_aabb_t> bounds;
	bvh_construction_node_t* left_child;
	bvh_construction_node_t* right_child;
};

static aabb_t encapsulateAABBs(std::vector<indexed_aabb_t>& aabbs) {

	aabb_t encap = aabbs[0].aabb;

	for (int i = 0; i < aabbs.size(); i++) {
		aabb_t cur = aabbs[i].aabb;
		encap.low.x = fminf(encap.low.x, cur.low.x);
		encap.low.y = fminf(encap.low.y, cur.low.y);
		encap.low.z = fminf(encap.low.z, cur.low.z);
		encap.high.x = fmaxf(encap.high.x, cur.high.x);
		encap.high.y = fmaxf(encap.high.y, cur.high.y);
		encap.high.z = fmaxf(encap.high.z, cur.high.z);
	}

	return encap;

}

static void loadOverlapping(std::vector<indexed_aabb_t>& dest, aabb_t bound, std::vector<indexed_aabb_t>& src) {

	for (int i = 0; i < src.size(); i++) {

		if (bound.overlaps(src[i].aabb)) {
			dest.push_back(src[i]);
		}

	}

}

static void splitBound(aabb_t bound, std::vector<indexed_aabb_t>& bounds, aabb_t& left_bound, aabb_t& right_bound) {

	vec3 boundDim = bound.high - bound.low;

	// TODO: split a little better than this
	if (boundDim.x > boundDim.y && boundDim.x > boundDim.z) {
		float split_x = (bound.low.x + bound.high.x) / 2.0f;
		left_bound = bound;
		left_bound.high.x = split_x;
		right_bound = bound;
		right_bound.low.x = split_x;
	} else if (boundDim.y > boundDim.z) {
		float split_y = (bound.low.y + bound.high.y) / 2.0f;
		left_bound = bound;
		left_bound.high.y = split_y;
		right_bound = bound;
		right_bound.low.y = split_y;
	} else {
		float split_z = (bound.low.z + bound.high.z) / 2.0f;
		left_bound = bound;
		left_bound.high.z = split_z;
		right_bound = bound;
		right_bound.low.z = split_z;
	}
}

static bvh_construction_node_t* buildBVHRecursive(std::vector<indexed_aabb_t>& bounds) {

	aabb_t bound = encapsulateAABBs(bounds);

	aabb_t left_bound, right_bound;

	splitBound(bound, bounds, left_bound, right_bound);

	std::vector<indexed_aabb_t> left_bounds;
	loadOverlapping(left_bounds, left_bound, bounds);

	std::vector<indexed_aabb_t> right_bounds;
	loadOverlapping(right_bounds, right_bound, bounds);

	if (left_bounds.size() == bounds.size() || right_bounds.size() == bounds.size()) {

		bvh_construction_node_t* leaf = new bvh_construction_node_t();
		leaf->is_leaf = true;
		leaf->bound = bound;
		leaf->bounds = bounds;
		return leaf;

	} else {

		bvh_construction_node_t* inner = new bvh_construction_node_t();
		inner->is_leaf = false;
		inner->bound = bound;
		inner->left_child = buildBVHRecursive(left_bounds);
		inner->right_child = buildBVHRecursive(right_bounds);
		return inner;

	}

}

static void unpackDFS(std::vector<bvh_node_t>& bvh, std::vector<int>& indices, bvh_construction_node_t* cnode) {

	bvh_node_t node;
	int node_index = bvh.size();
	bvh.push_back(node); // reserve a spot

	node.bound = cnode->bound;

	if (cnode->is_leaf) {

		node.is_leaf = true;
		node.i0 = indices.size();

		for (int i = 0; i < cnode->bounds.size(); i++) {
			indexed_aabb_t bound = cnode->bounds[i];
			indices.push_back(bound.index);
		}

		node.i1 = indices.size();

	} else {

		node.is_leaf = false;

		node.i0 = bvh.size();
		unpackDFS(bvh, indices, cnode->left_child);

		node.i1 = bvh.size();
		unpackDFS(bvh, indices, cnode->right_child);

	}

	bvh[node_index] = node;

}

void printBVH(std::vector<bvh_node_t>& bvh) {
	for (int i = 0; i < bvh.size(); i++) {
		bvh_node_t node = bvh[i];
		if (node.is_leaf) {
			std::cout << node.i1 - node.i0 << std::endl;
		}
	}
}

void buildBVH(std::vector<indexed_aabb_t>& bounds, std::vector<bvh_node_t>& bvh, std::vector<int>& indices) {

	bvh_construction_node_t* root = buildBVHRecursive(bounds);
	unpackDFS(bvh, indices, root);
	printBVH(bvh);

}
