#include <iostream>
#include <vector>

#include "bvh.h"

#define SAH_SEGMENTS 100

struct bvh_construction_node_t {
	bool is_leaf;
	aabb_t bound;
	std::vector<indexed_aabb_t> bounds;
	bvh_construction_node_t* left_child;
	bvh_construction_node_t* right_child;
};

static aabb_t encapsulateAABBs(std::vector<indexed_aabb_t>& aabbs) {

	aabb_t encap;
	encap.low = {INFINITY, INFINITY, INFINITY};
	encap.high = {-INFINITY, -INFINITY, -INFINITY};

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

static void countContaining(aabb_t& bound, std::vector<indexed_aabb_t>& src, int& contained, int& not_contained) {

	contained = 0;
	not_contained = 0;

	for (int i = 0; i < src.size(); i++) {

		if (src[i].aabb.centroidWithin(bound)) {
			contained++;
		} else {
			not_contained++;
		}

	}

}

static void loadContaining(aabb_t& bound, std::vector<indexed_aabb_t>& src, std::vector<indexed_aabb_t>& dest_contained, std::vector<indexed_aabb_t>& dest_not) {

	for (int i = 0; i < src.size(); i++) {

		if (src[i].aabb.centroidWithin(bound)) {
			dest_contained.push_back(src[i]);
		} else {
			dest_not.push_back(src[i]);
		}

	}

}

static bool splitBound(aabb_t bound, std::vector<indexed_aabb_t>& bounds, aabb_t& left_bound, aabb_t& right_bound) {

	float best_sah = bound.surface_area() * bounds.size();
	bool do_split = false;
	aabb_t best_left, best_right;

	vec3 dim = bound.high - bound.low;

	for (int i = 1; i < SAH_SEGMENTS; i++) {

		float split_x = bound.low.x + (dim.x * i) / SAH_SEGMENTS;

		aabb_t test_left = bound;
		test_left.high.x = split_x;
		aabb_t test_right = bound;
		test_right.low.x = split_x;

		int left_count, right_count;
		countContaining(test_left, bounds, left_count, right_count);
		if (left_count <= 1 || right_count <= 1) continue;

		float test_sah = test_left.surface_area() * left_count + test_right.surface_area() * right_count;
		if (test_sah < best_sah) {
			best_sah = test_sah;
			best_left = test_left;
			best_right = test_right;
			do_split = true;
		}

	}

	for (int i = 1; i < SAH_SEGMENTS; i++) {

		float split_y = bound.low.y + (dim.y * i) / SAH_SEGMENTS;

		aabb_t test_left = bound;
		test_left.high.y = split_y;
		aabb_t test_right = bound;
		test_right.low.y = split_y;

		int left_count, right_count;
		countContaining(test_left, bounds, left_count, right_count);
		if (left_count <= 1 || right_count <= 1) continue;

		float test_sah = test_left.surface_area() * left_count + test_right.surface_area() * right_count;
		if (test_sah < best_sah) {
			best_sah = test_sah;
			best_left = test_left;
			best_right = test_right;
			do_split = true;
		}

	}

	for (int i = 1; i < SAH_SEGMENTS; i++) {

		float split_z = bound.low.z + (dim.z * i) / SAH_SEGMENTS;

		aabb_t test_left = bound;
		test_left.high.z = split_z;
		aabb_t test_right = bound;
		test_right.low.z = split_z;

		int left_count, right_count;
		countContaining(test_left, bounds, left_count, right_count);
		if (left_count <= 1 || right_count <= 1) continue;

		float test_sah = test_left.surface_area() * left_count + test_right.surface_area() * right_count;
		if (test_sah < best_sah) {
			best_sah = test_sah;
			best_left = test_left;
			best_right = test_right;
			do_split = true;
		}

	}

	if (do_split) {
		left_bound = best_left;
		right_bound = best_right;
		return true;
	} else {
		return false;
	}

}

static bvh_construction_node_t* buildBVHRecursive(std::vector<indexed_aabb_t>& bounds) {

	aabb_t bound = encapsulateAABBs(bounds);

	aabb_t left_bound, right_bound;
	bool do_split = splitBound(bound, bounds, left_bound, right_bound);

	if (do_split) {

		std::vector<indexed_aabb_t> left_bounds, right_bounds;
		loadContaining(left_bound, bounds, left_bounds, right_bounds);

		bvh_construction_node_t* inner = new bvh_construction_node_t;
		inner->is_leaf = false;
		inner->bound = bound;
		inner->left_child = buildBVHRecursive(left_bounds);
		inner->right_child = buildBVHRecursive(right_bounds);
		return inner;

	} else {

		bvh_construction_node_t* leaf = new bvh_construction_node_t;
		leaf->is_leaf = true;
		leaf->bound = bound;
		leaf->bounds = bounds;
		return leaf;

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

void buildBVH(std::vector<indexed_aabb_t>& bounds, std::vector<bvh_node_t>& bvh, std::vector<int>& indices) {

	bvh_construction_node_t* root = buildBVHRecursive(bounds);
	unpackDFS(bvh, indices, root);

	std::cout << "Converted " << bounds.size() << " bounding boxes into BVH with " << bvh.size() << " nodes and " << indices.size() << " bounding boxes." << std::endl;

}
