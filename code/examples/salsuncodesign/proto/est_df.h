// est_df.h
//
// Prototype of a layer-pipelined (dataflow) angle estimator for the SA-LSUN
// band kernel: every layer of the estimator is its own hardware task, and
// groups of NB blocks stream through them, so that the throughput is that
// of the slowest layer instead of the sum of all layers. One instance
// serves the five estimators of a network (NRES residual blocks, NHK hidden
// units):
// the tasks are sized for the largest one and take the sizes of estimator
// k at run time; the parameters of all five are loaded into the tasks.
//
// An estimator normalizes its state once, at its input (LayerNorm over the
// features of a block, the state read times 2^LN_SCALE_BITS), then runs
// NRES residual blocks of an FC layer, tanh, an FC layer and the skip
// connection, and maps the outputs z of its last FC layer to the angles
// asin(tanh(z)) in (-pi/2, pi/2) (train_salsun2d_variants with
// StandardizationScope='layer', ResidualLayerNorm=false, Activation='tanh',
// OutputMapping='asintanh'). The mapping of the outputs to angles by an
// inverse trigonometric function is an idea of Shogo Muramatsu.
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
#ifndef EST_P
#define EST_P 24
#endif
#ifndef EST_OPAR_R
#define EST_OPAR_R 1
#endif
#ifndef EST_OPAR_O
#define EST_OPAR_O 1
#endif
#ifndef EST_NRES
#define EST_NRES 3
#endif
#ifndef EST_NH0                 // hidden units of estimator 0 (width 1: 135, 0.5: 68)
#define EST_NH0 135
#endif
#ifndef EST_NH1                 // hidden units of estimators 1 to 4 (width 1: 72, 0.5: 36)
#define EST_NH1 72
#endif
constexpr int PH = 24;          // weights per word of the parameter memory (host)
constexpr int P = EST_P;        // FC inputs per lane and cycle (a divisor of PH)
constexpr int SPLIT = PH / P;   // on-chip words of a host word
constexpr int OPAR_R = EST_OPAR_R;   // FC outputs per cycle, residual blocks
constexpr int OPAR_O = EST_OPAR_O;   // FC outputs per cycle, output layer
constexpr int NEST = 5;         // estimators
constexpr int NRES = EST_NRES;  // residual blocks (1 to 3)
constexpr int NFK[NEST] = {135, 72, 72, 72, 72};   // features
constexpr int NHK[NEST] = {EST_NH0, EST_NH1, EST_NH1, EST_NH1, EST_NH1};   // hidden units
constexpr int NAK[NEST] = {49, 28, 28, 28, 28};    // predicted angles
constexpr int NFMAX = 135, NAMAX = 49;
// The FC layers of an estimator: the first (features to hidden units) and
// the second (hidden units to features) of a residual block, and the
// output layer (features to angles); their inputs and outputs for estimator k
constexpr int FC1 = 0, FC2 = 1, FCO = 2;
constexpr int nInK(int kind, int k) { return kind == FC2 ? NHK[k] : NFK[k]; }
constexpr int nOutK(int kind, int k) { return kind == FC1 ? NHK[k] : kind == FC2 ? NFK[k] : NAK[k]; }
constexpr int OPARK[3] = {OPAR_R, OPAR_R, OPAR_O};
constexpr int hwords(int n) { return (n + PH - 1) / PH; }  // host words of a row
constexpr int words(int n) { return SPLIT * hwords(n); }   // on-chip words of a row
static_assert(SPLIT * P == PH, "an on-chip word is a whole part of a host word");
static_assert(NRES >= 1 && NRES <= 3, "1 to 3 residual blocks");
// a pass of an FC layer over its outputs must cover the reading of P
// inputs, and its passes the sending of the outputs (one per cycle)
constexpr bool fcFits(int kind, int k)
{
    return (nOutK(kind, k) + OPARK[kind] - 1) / OPARK[kind] >= P && nInK(kind, k) >= P &&
           words(nInK(kind, k)) * ((nOutK(kind, k) + OPARK[kind] - 1) / OPARK[kind]) >= nOutK(kind, k);
}
constexpr bool fcFitAll(int kind = 0, int k = 0)
{
    return kind == 3 ? true : k == NEST ? fcFitAll(kind + 1, 0) : fcFits(kind, k) && fcFitAll(kind, k + 1);
}
static_assert(fcFitAll(), "too many FC outputs per cycle, or too few inputs, for an estimator");
constexpr int FIFO = 136;       // rows between tasks (the rows of a group)

// FC inputs [18 11]: values are rounded to nearest and saturated on
// assignment, as the MATLAB design rounds them (toX)
typedef ap_fixed<18, 7, AP_RND, AP_SAT> sig_t;
typedef ap_fixed<18, 2> wgt_t;  // weights    [18 16]
typedef ap_fixed<48, 21> acc_t; // sums       [48 27]
typedef ap_fixed<48, 26> sq_t;  // sums of squares [48 22]

// LayerNorm (of the state, at the input) in fixed point: gamma and beta
// [18 15]; the mean [24 16] and the reciprocal standard deviation [26 16]
// (unsigned) per block from the exact sums in single precision; (x - mean)
// * rs rounded to [24 16]; then times gamma plus beta, rounded to the
// signal format. The state is read times 2^LN_SCALE_BITS (its values are
// small for the signal format), with the epsilon of the variance (1e-8)
// scaled alike, which leaves the normalized state as it is.
constexpr int LN_SCALE_BITS = 6;
constexpr float LN_EPS = 1e-8f * (1 << (2 * LN_SCALE_BITS));
typedef ap_fixed<18, 3, AP_RND, AP_SAT> lnp_t;
typedef ap_fixed<24, 8, AP_RND, AP_SAT> lnm_t;
typedef ap_ufixed<26, 10, AP_RND, AP_SAT> lnr_t;
typedef ap_fixed<24, 8, AP_RND, AP_SAT> lnt_t;

// tanh and the output mapping gd(x) = asin(tanh(x)) by tables at x = -8 +
// i/64, i = 0..1024, in [24 22] and [24 21], interpolated linearly with a
// 16-bit fraction (act_table.h, written by pack_est_df.m; export_est_df.m
// models them); the values at -8 and 8 beyond
constexpr int ACT_N = 1025;
typedef ap_fixed<24, 2> ttab_t;
typedef ap_fixed<24, 3> gdtab_t;

struct row_t {                  // one feature of the NB blocks of a group
    sig_t v[NB];
};
struct arow_t {                 // one FC output of the NB blocks of a group
    acc_t v[NB];
};
struct wword_t {                // P consecutive weights of a row (on chip)
    wgt_t w[P];
};
struct lnpar_t {                // LayerNorm scale and shift of a feature
    float gamma, beta;
};

// Parameter memories as the host writes them: a weight word as 24 raw
// 18-bit values in the low bits of 512 (value p at bits 18p..18p+17), a
// bias as the raw 48-bit accumulator value in 64 bits, the LayerNorm
// parameters as (gamma, beta) floats
typedef ap_uint<512> wraw_t;
typedef ap_int<64> braw_t;

// Host words of an FC layer of estimator k and their offset in the
// parameter memory of that layer (estimators in order); the offsets of its
// biases
constexpr int layerWords(int kind, int k) { return nOutK(kind, k) * hwords(nInK(kind, k)); }
constexpr int layerOffset(int kind, int k)
{
    int s = 0;
    for (int j = 0; j < k; j++) s += layerWords(kind, j);
    return s;
}
constexpr int biasOffset(int kind, int k)
{
    int s = 0;
    for (int j = 0; j < k; j++) s += nOutK(kind, j);
    return s;
}
constexpr int vecOffset(int k) { return biasOffset(FC2, k); }
constexpr int W1WORDS = layerOffset(FC1, NEST), W2WORDS = layerOffset(FC2, NEST);
constexpr int OUTWORDS = layerOffset(FCO, NEST);
constexpr int NHALL = biasOffset(FC1, NEST), NVEC = biasOffset(FC2, NEST), NAALL = biasOffset(FCO, NEST);

// On chip, an FC layer with OPAR outputs per cycle keeps its weights and
// biases in OPAR banks, output o in bank o % OPAR at row o / OPAR. Rows of
// a bank for estimator k, their words (on chip), and the offsets of
// estimator k in a bank
constexpr int bankRows(int kind, int opar, int k) { return (nOutK(kind, k) + opar - 1) / opar; }
constexpr int bankWords(int kind, int opar, int k) { return bankRows(kind, opar, k) * words(nInK(kind, k)); }
constexpr int bankRowOffset(int kind, int opar, int k)
{
    int s = 0;
    for (int j = 0; j < k; j++) s += bankRows(kind, opar, j);
    return s;
}
constexpr int bankWordOffset(int kind, int opar, int k)
{
    int s = 0;
    for (int j = 0; j < k; j++) s += bankWords(kind, opar, j);
    return s;
}

// Parameter memory, in order: W1 of residual block 0 for estimators 0..4,
// W2 of block 0, W1 and W2 of blocks 1 and 2, then Wo of estimators 0..4
// (rows of hwords(nIn) words, zero-padded; W1 has
// NHK rows of NFK weights, W2 NFK rows of NHK); biases B1, B2 per block, then
// Bo (same order); LayerNorm parameters of the input (estimators in order)
constexpr int NWORDS = NRES * (W1WORDS + W2WORDS) + OUTWORDS;
constexpr int NBIAS = NRES * (NHALL + NVEC) + NAALL;
constexpr int NLN = NVEC;

void est_engine(hls::stream<row_t> &features, hls::stream<arow_t> &angles, const wraw_t *wmem,
                const braw_t *bmem, const float *lnmem, int k, int ngroups, bool load);

} // namespace estdf

void est_df_top(const estdf::wraw_t *wmem, const estdf::braw_t *bmem, const float *lnmem,
                hls::stream<estdf::row_t> &features, hls::stream<estdf::arow_t> &angles, int k,
                int ngroups);

#endif
