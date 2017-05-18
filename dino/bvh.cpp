#include <iostream>
#include <vector>

#include "bvh.h"

#define SAH_SEGMENTS 10

/*
A structure containing intermediate node construction information.
*/
struct bvh_construction_node_t {
	bool is_leaf;
	aabb_t bound;
	std::vector<indexed_aabb_t> bounds;
	bvh_construction_node_t* left_child;
	bvh_construction_node_t* right_child;
};

/*
Encapsulate all input axis-aligned bounding boxes in a single, large
axis-aligned bounding box.
*/
static aabb_t encapsulateAABBs(std::vector<indexed_aabb_t>& aabbs) {

	aabb_t encap;
	encap.low = {INFINITY, INFINITY, INFINITY};
	encap.high = {-INFINITY, -INFINITY, -INFINITY};

	// grow the encapsulating box to hold all smaller boxes
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

/*
Count the number of axis-aligned bounding boxes whose centroids fall within a
provided bound.
*/
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

/*
Sort all bounding boxes into two vectors based on whether their centroids fall
within a provided bound.
*/
static void loadContaining(aabb_t& bound, std::vector<indexed_aabb_t>& src, std::vector<indexed_aabb_t>& dest_contained, std::vector<indexed_aabb_t>& dest_not) {

	for (int i = 0; i < src.size(); i++) {

		if (src[i].aabb.centroidWithin(bound)) {
			dest_contained.push_back(src[i]);
		} else {
			dest_not.push_back(src[i]);
		}

	}

}

/*
Split a group of bounding boxes into two groups along an axis-aligned plane,
using the Surface Area Heuristic to */
static bool splitBound(aabb_t bound, std::vector<indexed_aabb_t>& bounds, aabb_t& left_bound, aabb_t& right_bound) {

	// initialize the best heuristic value to the bounds with no split applied
	float best_sah = bound.surface_area() * bounds.size();
	bool do_split = false;
	aabb_t best_left, best_right;

	vec3 dim = bound.high - bound.low;

	// test a number of splits along the x axis
	for (int i = 1; i < SAH_SEGMENTS; i++) {

		float split_x = bound.low.x + (dim.x * i) / SAH_SEGMENTS;

		aabb_t test_left = bound;
		test_left.high.x = split_x;
		aabb_t test_right = bound;
		test_right.low.x = split_x;

		int left_count, right_count;
		countContaining(test_left, bounds, left_count, right_count);
		if (left_count <= 1 || right_count <= 1) continue;

		// calculate the heuristic value of the split, and see if it is the best so far
		float test_sah = test_left.surface_area() * left_count + test_right.surface_area() * right_count;
		if (test_sah < best_sah) {
			best_sah = test_sah;
			best_left = test_left;
			best_right = test_right;
			do_split = true;
		}

	}

	// test a number of splits along the y axis
	for (int i = 1; i < SAH_SEGMENTS; i++) {

		float split_y = bound.low.y + (dim.y * i) / SAH_SEGMENTS;

		aabb_t test_left = bound;
		test_left.high.y = split_y;
		aabb_t test_right = bound;
		test_right.low.y = split_y;

		int left_count, right_count;
		countContaining(test_left, bounds, left_count, right_count);
		if (left_count <= 1 || right_count <= 1) continue;

		// calculate the heuristic value of the split, and see if it is the best so far
		float test_sah = test_left.surface_area() * left_count + test_right.surface_area() * right_count;
		if (test_sah < best_sah) {
			best_sah = test_sah;
			best_left = test_left;
			best_right = test_right;
			do_split = true;
		}

	}

	// test a number of splits along the z axis
	for (int i = 1; i < SAH_SEGMENTS; i++) {

		float split_z = bound.low.z + (dim.z * i) / SAH_SEGMENTS;

		aabb_t test_left = bound;
		test_left.high.z = split_z;
		aabb_t test_right = bound;
		test_right.low.z = split_z;

		int left_count, right_count;
		countContaining(test_left, bounds, left_count, right_count);
		if (left_count <= 1 || right_count <= 1) continue;

		// calculate the heuristic value of the split, and see if it is the best so far
		float test_sah = test_left.surface_area() * left_count + test_right.surface_area() * right_count;
		if (test_sah < best_sah) {
			best_sah = test_sah;
			best_left = test_left;
			best_right = test_right;
			do_split = true;
		}

	}

	// finalize the split if it is desired
	if (do_split) {
		left_bound = best_left;
		right_bound = best_right;
		return true;
	} else {
		return false;
	}

}

/*
Recursively build the BVH by splitting the bounding boxes where it will probably
make the traversal more efficient.
*/
static bvh_construction_node_t* buildBVHRecursive(std::vector<indexed_aabb_t>& bounds) {

	aabb_t bound = encapsulateAABBs(bounds);

	// find if a split is desired
	aabb_t left_bound, right_bound;
	bool do_split = splitBound(bound, bounds, left_bound, right_bound);

	if (do_split) {

		// build an inner node, and recurse

		std::vector<indexed_aabb_t> left_bounds, right_bounds;
		loadContaining(left_bound, bounds, left_bounds, right_bounds);

		bvh_construction_node_t* inner = new bvh_construction_node_t;
		inner->is_leaf = false;
		inner->bound = bound;
		inner->left_child = buildBVHRecursive(left_bounds);
		inner->right_child = buildBVHRecursive(right_bounds);
		return inner;

	} else {

		// build a leaf node and stop recursing

		bvh_construction_node_t* leaf = new bvh_construction_node_t;
		leaf->is_leaf = true;
		leaf->bound = bound;
		leaf->bounds = bounds;
		return leaf;

	}

}

/*
Perform a depth-first traversal of the BVH, loading each visited node into the
flat BVH vector which will be uploaded to the GPU.
*/
static void unpackDFS(std::vector<bvh_node_t>& bvh, std::vector<int>& indices, bvh_construction_node_t* cnode) {

	bvh_node_t node;
	int node_index = (int) bvh.size();

	// reserve a spot for the node in case it is an inner node
	bvh.push_back(node);

	node.bound = cnode->bound;

	if (cnode->is_leaf) {

		// load the leaf node data into the vector

		node.i0 = (int) indices.size();

		for (int i = 0; i < cnode->bounds.size(); i++) {
			indexed_aabb_t bound = cnode->bounds[i];
			indices.push_back(bound.index);
		}

		node.i1 = BVH_LEAF_MASK | (int) indices.size();

	} else {

		// load the left and right subtrees into the vector, recording their indices

		node.i0 = (int) bvh.size();
		unpackDFS(bvh, indices, cnode->left_child);

		node.i1 = (int) bvh.size();
		unpackDFS(bvh, indices, cnode->right_child);

	}

	// finally, save the node in its reserved spot
	bvh[node_index] = node;

}

/*
Construct a new BVH given a vector of boundaries, and load the BVH into a flat
vector which uses no direct pointers (good for GPU use).
*/
void buildBVH(std::vector<indexed_aabb_t>& bounds, std::vector<bvh_node_t>& bvh, std::vector<int>& indices) {

	bvh_construction_node_t* root = buildBVHRecursive(bounds);
	unpackDFS(bvh, indices, root);

	std::cout << "Converted " << bounds.size() << " bounding boxes into BVH with " << bvh.size() << " nodes and " << indices.size() << " bounding boxes." << std::endl;

}
