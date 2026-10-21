// est_df.h
//
// Prototype of a layer-pipelined (dataflow) angle estimator for the SA-LSUN
// band kernel: every layer of the estimator is its own hardware task, and
// groups of NB blocks stream through them, so that the throughput is that
// of the slowest layer instead of the sum of all layers.
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

constexpr int NB = 15;          // blocks per group (one lane each)
constexpr int P = 8;            // FC inputs per lane and cycle
constexpr int NF = 135;         // features (15 channels x 3 x 3 neighbors)
constexpr int NH = 135;         // hidden units (width 1)
constexpr int NA = 49;          // predicted angles
constexpr int NRES = 3;         // residual blocks
constexpr int pad(int n) { return P * ((n + P - 1) / P); }
constexpr int NFP = pad(NF);    // weight rows padded to whole words of P
constexpr int NHP = pad(NH);
constexpr int FIFO = 160;       // rows between tasks (at least one group of rows)

typedef ap_fixed<18, 7> sig_t;  // FC inputs  [18 11]
typedef ap_fixed<18, 2> wgt_t;  // weights    [18 16]
typedef ap_fixed<48, 21> acc_t; // sums       [48 27]

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
    sig_t gamma, beta;
};

// Words of the weight memory, in order: for each residual block W1 (NH rows
// of NFP/P words), W2 (NF rows of NHP/P words); then Wo (NA rows of NFP/P
// words). Biases in order B1, B2 per block, then Bo; LayerNorm parameters
// per block.
constexpr int NW1 = NH * (NFP / P), NW2 = NF * (NHP / P), NWO = NA * (NFP / P);
constexpr int NWORDS = NRES * (NW1 + NW2) + NWO;
constexpr int NBIAS = NRES * (NH + NF) + NA;
constexpr int NLN = NRES * NF;

} // namespace estdf

#endif
