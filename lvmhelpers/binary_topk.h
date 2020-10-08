// k-best assignments for independent binary variables
// (optimized version of zeroth order viterbi)
// author: vlad niculae <vlad@vene.ro>
// license: mit

#pragma once
#include <vector>
#include <algorithm>
#include <cassert>
#include <bitset>


#define CODE_MAX_SIZE 512

typedef std::bitset<CODE_MAX_SIZE> configuration;
typedef float floating;

struct scored_cfg
{
    floating score;
    configuration cfg;

    /*
     * set the ith variable on, adding v to score
     */
    scored_cfg update(unsigned i, floating v)
    {
        scored_cfg out {this->score + v, this->cfg};
        out.cfg[i] = 1;
        return out;
    }

    std::vector<int> cfg_vector(int dim)
    {
        std::vector<int> out;
        for (int i = 0; i < dim; ++i)
            out.push_back((int) cfg.test(i));
        return out;
    }

    void populate_out(floating* out, int dim)
    {
        for (int i = 0; i < dim; ++i)
            out[i] = (float) cfg.test(i);
    }
};


/*
 * Merge the two k-best lists, depending if the next state is 0 or 1.
 * floatinghe k-best for 0 is (a_begin, a_end)
 * floatinghe k-best  for 1 is [(b.score + val, b.cfg + [1]) for b in (a_begin, a_end)]
 *
 * Implementation is standard list merge, stopping once we produced k items.
 * We also avoid building the b vector.
 */
std::vector<scored_cfg>::iterator
merge_branch(std::vector<scored_cfg>::iterator a_begin,
             std::vector<scored_cfg>::iterator a_end,
             unsigned i,
             floating val,
             std::vector<scored_cfg>::iterator out_begin,
             int k)
{

    auto b_begin = a_begin;
    auto b_end = a_end;
    int inserted = 0;

    while((inserted < k) & (a_begin != a_end) & (b_begin != b_end)) {
        auto b_begin_item = b_begin->update(i, val);
        if (b_begin_item.score > a_begin->score) {
            *out_begin = b_begin_item;
            ++b_begin;
        } else {
            *out_begin = *a_begin;
            ++a_begin;
        }
        ++out_begin;
        ++inserted;
    }

    while((inserted < k) & (a_begin != a_end)) {
        *out_begin = *a_begin;
        ++a_begin;
        ++out_begin;
        ++inserted;
    }

    while((inserted < k) & (b_begin != b_end)) {
        *out_begin = b_begin->update(i, val);
        ++b_begin;
        ++out_begin;
        ++inserted;
    }

    return out_begin;
}

std::vector<scored_cfg> topk(const std::vector<floating>& x, int k)
{
    assert(k > 1);
    // partial configuration starting with 0
    scored_cfg c0 = {0, 0};

    // partial configuration starting with 1
    scored_cfg c1 = c0.update(0, x[0]);

    std::vector<scored_cfg> curr(k), next(k);
    if (x[0] >= 0) {
        curr[0] = c1;
        curr[1] = c0;
    } else {
        curr[0] = c0;
        curr[1] = c1;
    }

    auto curr_begin = curr.begin();
    auto curr_end = curr_begin + 2;
    auto next_begin = next.begin();
    auto next_end = next_begin;

    for (unsigned i = 1; i < x.size(); ++i) {
        next_end = merge_branch(curr_begin, curr_end, i, x[i], next_begin, k);
        std::swap(curr_begin, next_begin);
        std::swap(curr_end, next_end);
    }
    return std::vector<scored_cfg> (curr_begin, curr_end);
}

std::vector<scored_cfg> topk(floating* x, int size, int k)
{
    std::vector<floating> xvec(x, x + size);
    return topk(xvec, k);
}

