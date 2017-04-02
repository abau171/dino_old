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

static int countOverlapping(aabb_t& bound, std::vector<indexed_aabb_t>& src) {

	int count = 0;

	for (int i = 0; i < src.size(); i++) {

		if (bound.overlaps(src[i].aabb)) {
			count++;
		}

	}

	return count;

}

static void loadOverlapping(std::vector<indexed_aabb_t>& dest, aabb_t& bound, std::vector<indexed_aabb_t>& src) {

	for (int i = 0; i < src.size(); i++) {

		if (bound.overlaps(src[i].aabb)) {
			dest.push_back(src[i]);
		}

	}

}

static float calculateSAH(std::vector<indexed_aabb_t>& bounds, aabb_t& left_bound, aabb_t& right_bound) {

	return left_bound.surface_area() * countOverlapping(left_bound, bounds) + right_bound.surface_area() * countOverlapping(right_bound, bounds);
}

#define SAH_SEGMENTS 100

static void splitBound(aabb_t bound, std::vector<indexed_aabb_t>& bounds, aabb_t& left_bound, aabb_t& right_bound) {

	float best_sah = INFINITY;
	aabb_t best_left, best_right;

	vec3 dim = bound.high - bound.low;

	for (int i = 1; i < SAH_SEGMENTS; i++) {

		float split_x = bound.low.x + (dim.x * i) / SAH_SEGMENTS;

		aabb_t test_left = bound;
		test_left.high.x = split_x;
		aabb_t test_right = bound;
		test_right.low.x = split_x;

		float test_sah = calculateSAH(bounds, test_left, test_right);
		if (test_sah < best_sah) {
			best_sah = test_sah;
			best_left = test_left;
			best_right = test_right;
		}

	}

	for (int i = 1; i < SAH_SEGMENTS; i++) {

		float split_y = bound.low.y + (dim.y * i) / SAH_SEGMENTS;

		aabb_t test_left = bound;
		test_left.high.y = split_y;
		aabb_t test_right = bound;
		test_right.low.y = split_y;

		float test_sah = calculateSAH(bounds, test_left, test_right);
		if (test_sah < best_sah) {
			best_sah = test_sah;
			best_left = test_left;
			best_right = test_right;
		}

	}

	for (int i = 1; i < SAH_SEGMENTS; i++) {

		float split_z = bound.low.z + (dim.z * i) / SAH_SEGMENTS;

		aabb_t test_left = bound;
		test_left.high.z = split_z;
		aabb_t test_right = bound;
		test_right.low.z = split_z;

		float test_sah = calculateSAH(bounds, test_left, test_right);
		if (test_sah < best_sah) {
			best_sah = test_sah;
			best_left = test_left;
			best_right = test_right;
		}

	}

	left_bound = best_left;
	right_bound = best_right;

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

		node.i0 = indices.size();

		for (int i = 0; i < cnode->bounds.size(); i++) {
			indexed_aabb_t bound = cnode->bounds[i];
			indices.push_back(bound.index);
		}

		node.i1 = BVH_LEAF_MASK | indices.size();

	} else {

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
		if (node.i1 & BVH_LEAF_MASK) {
			std::cout << (node.i1 & BVH_I1_MASK) - node.i0 << std::endl;
		}
	}
}

void buildBVH(std::vector<indexed_aabb_t>& bounds, std::vector<bvh_node_t>& bvh, std::vector<int>& indices) {

	bvh_construction_node_t* root = buildBVHRecursive(bounds);
	unpackDFS(bvh, indices, root);

	std::cout << "Converted " << bounds.size() << " bounding boxes into BVH with " << bvh.size() << " nodes and " << indices.size() << " bounding boxes." << std::endl;
	//printBVH(bvh);

}
