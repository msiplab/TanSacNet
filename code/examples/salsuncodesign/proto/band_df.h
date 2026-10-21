// band_df.h
//
// Band kernel of SA-LSUN 2-D inference with the dataflow estimator
// (est_df.cpp): one band of BAND block rows with HALO block rows of
// context above and below (HALO = 0 for a band of the whole frame height),
// analysis, coefficient mask and synthesis, as salsun2d_hls_band.m
// computes them. The estimator of each analysis stage runs as a layer
// pipeline over groups of NB blocks of a column (est_df.cpp), between a
// task that gathers the standardized neighbor features and a task that
// rotates the coefficients of the column. The angles go to memory (qmem)
// for the synthesis.
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
#ifndef BAND_DF_H
#define BAND_DF_H

#include "est_df.h"

#ifndef SZY
#define SZY 40
#endif
#ifndef SZX
#define SZX 40
#endif
#ifndef BAND
#define BAND 10
#endif
#ifndef HALO
#define HALO 0
#endif

namespace bdf {

using namespace estdf;

constexpr int MY = 4, MX = 4, NDEC = 16, PS = 8, PA = 8, NSTAGES = 4;
constexpr int NROWS = SZY / MY, NCOLS = SZX / MX;      // block rows and columns of a frame
constexpr int SUBROWS = BAND + 2 * HALO;              // block rows of a band with halo
constexpr int NG = SUBROWS / NB;                      // groups of NB rows in a column
constexpr int NBANDS = (NROWS + BAND - 1) / BAND;
constexpr int DEPTH = NCOLS * NG;                     // blocks per bank
static_assert(SUBROWS % NB == 0, "the band rows with halo must be a multiple of NB");
static_assert(HALO > 0 || BAND == NROWS, "a band without halo must be the whole frame height");

// Angles per block: stage 0 (W and U, 28 each), stages 1..3 (28), stage 4 (7)
constexpr int NROT = PA * (PA - 1) / 2;               // 28
constexpr int NROT_LAST = PA - 1;                     // 7
constexpr int NQ = 2 * NROT + 3 * NROT + NROT_LAST;   // 147
constexpr int qOffset(int s) { return s == 0 ? 0 : 2 * NROT + (s - 1) * NROT; }
constexpr int NQMEM = SUBROWS * NCOLS * NQ;           // angle memory words of a band

// Small parameters (prm): Cvh (16 x 16, column-major), Mask, V0 MusW, MusU,
// V0t MusW, MusU, stage Mus and SynMus (4 x 8 each)
constexpr int P_CVH = 0, P_MASK = 256, P_V0W = 272, P_V0U = 280, P_V0TW = 288, P_V0TU = 296;
constexpr int P_MUS = 304, P_SYNMUS = 336, NPRM = 368;

// Statistics: mu, sigma (stats_in) and the sums (stats_out), NEST x NDEC each
constexpr int NSTATS = NEST * NDEC;

struct blk_t {                    // the coefficients of a block
    float v[NDEC];
};
typedef ap_fixed<18, 1> q_t;      // angle / pi
typedef ap_fixed<20, 2> stab_t;   // sin(pi i/1024), i = 0..2048 (sin_table.h)
constexpr int SIN_N = 2049;
typedef ap_uint<32> qword_t;      // one angle per memory word

} // namespace bdf

extern "C" void salsun2d_band_df(const float *src, const float *prm, const estdf::wword_t *wmem,
                                 const estdf::acc_t *bmem, const estdf::lnpar_t *lnmem,
                                 const float *stats_in, float *dst, float *stats_out,
                                 bdf::qword_t *qmem, int first_band, int num_bands);

#endif
