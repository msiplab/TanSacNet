// est_df.h
//
// Prototype of a layer-pipelined (dataflow) angle estimator for the SA-LSUN
// band kernel: every layer of the estimator is its own hardware task, and
// groups of NB blocks stream through them, so that the throughput is that
// of the slowest layer instead of the sum of all layers. One instance
// serves the five estimators of a network (3 residual blocks of width 1):
// the tasks are sized for the largest one and take the sizes of estimator
// k at run time; the parameters of all five are loaded into the tasks.
//
// Copyright (c) 2026, Shogo MURAMATSU
//
// All rights reserved.
//
// Contact address: Shogo MURAMATSU,
//    Faculty of Engineering, Niigata University,
//    8050 2-no-cho Ikarashi, Nishi-ku,
//    Niigata, 950-2181, JAPAN
//
// http://msiplab.eng.niigata-u.ac.jp/
//
#ifndef EST_DF_H
#define EST_DF_H

#include <ap_fixed.h>
#include <hls_stream.h>

namespace estdf {

constexpr int NB = 5;           // blocks per group (one lane each)
constexpr int P = 24;           // FC inputs per lane and cycle
constexpr int NEST = 5;         // estimators
constexpr int NRES = 3;         // residual blocks
constexpr int NFK[NEST] = {135, 72, 72, 72, 72};   // features (= hidden units, width 1)
constexpr int NAK[NEST] = {49, 28, 28, 28, 28};    // predicted angles
constexpr int NFMAX = 135, NAMAX = 49;
constexpr int words(int n) { return (n + P - 1) / P; }
constexpr int FIFO = 136;       // rows between tasks (the rows of a group)

// FC inputs [18 11]: values are rounded to nearest and saturated on
// assignment, as the MATLAB design rounds them (toX)
typedef ap_fixed<18, 7, AP_RND, AP_SAT> sig_t;
typedef ap_fixed<18, 2> wgt_t;  // weights    [18 16]
typedef ap_fixed<48, 21> acc_t; // sums       [48 27]
typedef ap_fixed<48, 26> sq_t;  // sums of squares [48 22]

// GELU by a table of GELU(x) at x = -8 + i/64, i = 0..1024, in [24 19],
// interpolated linearly with a 16-bit fraction (gelu_table.h, written by
// export_est_df.m, which also models it); 0 below -8, x above 8
constexpr int GELU_N = 1025;
typedef ap_fixed<24, 5> gtab_t;

struct row_t {                  // one feature of the NB blocks of a group
    sig_t v[NB];
};
struct arow_t {                 // one FC output of the NB blocks of a group
    acc_t v[NB];
};
struct wword_t {                // P consecutive weights of a row
    wgt_t w[P];
};
struct lnpar_t {                // LayerNorm scale and shift of a feature
    float gamma, beta;
};

// Words of a square layer (W1 or W2) and of the output layer of estimator k,
// and their offsets in the memory of one task (estimators in order)
constexpr int sqWords(int k) { return NFK[k] * words(NFK[k]); }
constexpr int outWords(int k) { return NAK[k] * words(NFK[k]); }
constexpr int sqOffset(int k)
{
    int s = 0;
    for (int j = 0; j < k; j++) s += sqWords(j);
    return s;
}
constexpr int outOffset(int k)
{
    int s = 0;
    for (int j = 0; j < k; j++) s += outWords(j);
    return s;
}
constexpr int vecOffset(int k)
{
    int s = 0;
    for (int j = 0; j < k; j++) s += NFK[j];
    return s;
}
constexpr int aOffset(int k)
{
    int s = 0;
    for (int j = 0; j < k; j++) s += NAK[j];
    return s;
}
constexpr int SQWORDS = sqOffset(NEST), OUTWORDS = outOffset(NEST);
constexpr int NVEC = vecOffset(NEST), NAALL = aOffset(NEST);

// Parameter memory, in order: W1 of residual block 0 for estimators 0..4,
// W2 of block 0, W1 and W2 of blocks 1 and 2, then Wo of estimators 0..4
// (rows of words(nIn) words, zero-padded); biases B1, B2 per block, then
// Bo (same order); LayerNorm parameters per block (estimators in order)
constexpr int NWORDS = NRES * 2 * SQWORDS + OUTWORDS;
constexpr int NBIAS = NRES * 2 * NVEC + NAALL;
constexpr int NLN = NRES * NVEC;

void est_engine(hls::stream<row_t> &features, hls::stream<arow_t> &angles, const wword_t *wmem,
                const acc_t *bmem, const lnpar_t *lnmem, int k, int ngroups, bool load);

} // namespace estdf

void est_df_top(const estdf::wword_t *wmem, const estdf::acc_t *bmem, const estdf::lnpar_t *lnmem,
                hls::stream<estdf::row_t> &features, hls::stream<estdf::arow_t> &angles, int k,
                int ngroups);

#endif
