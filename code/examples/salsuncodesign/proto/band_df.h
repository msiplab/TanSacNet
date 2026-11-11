// band_df.h
//
// Band kernel of SA-LSUN 2-D inference with the dataflow estimator
// (est_df.cpp): one band of BAND block rows with HALO block rows of
// context above and below (HALO = 0 for a band of the whole frame height),
// analysis, coefficient mask and synthesis, as salsun2d_hls_band.m
// computes them. The estimator of each analysis stage runs as a layer
// pipeline over groups of NB blocks of a column (est_df.cpp), between a
// task that gathers the standardized neighbor features and a task that
// rotates the coefficients of the column. The angles, rounded to QBITS
// bits, are kept on chip (qmem, URAM) for the synthesis, as offsets of
// QOFS bits from a center per angle (prm), saturated (QOFS = QBITS: the
// angles themselves).
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
#include <hls_burst_maxi.h>

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
#ifndef QBITS
#define QBITS 10                  // bits of an angle (theta/pi)
#endif
#ifndef QOFS
#define QOFS QBITS                // bits of a stored angle (offset from its center)
#endif

namespace bdf {

using namespace estdf;

constexpr int MY = 4, MX = 4, NDEC = 16, PS = 8, PA = 8, NSTAGES = 4;
constexpr int NROWS = SZY / MY, NCOLS = SZX / MX;      // block rows and columns of a frame
constexpr int SUBROWS = BAND + 2 * HALO;              // block rows of a band with halo
constexpr int NG = SUBROWS / NB;                      // groups of NB rows in a column
constexpr int NBANDS = (NROWS + BAND - 1) / BAND;
constexpr int NBLK = NCOLS * SUBROWS;                // blocks of a band (column by column)
static_assert(SUBROWS % NB == 0, "the band rows with halo must be a multiple of NB");
static_assert(HALO > 0 || BAND == NROWS, "a band without halo must be the whole frame height");

// Angles per block: stage 0 (W and U, 28 each), stages 1..3 (28), stage 4 (7)
constexpr int NROT = PA * (PA - 1) / 2;               // 28
constexpr int NROT_LAST = PA - 1;                     // 7
// Angle memory (on chip): per group, the angles k of its NB blocks (the
// offset of NQB bits from the center of angle k, saturated, of the angle
// of NQA bits), NQR angles to a word of NQB NB NQR bits (350 for 10 bits:
// five URAM wide, 280 for 8: four),
// from place NQPAD of the first word on, so that the last word is full
// (angle k in word (k + NQPAD) / NQR at bits NQB (NB i + b),
// i = (k + NQPAD) % NQR, block b), NQ5 words for the 28 angles; per group
// 2 NQ5 words for stage 0 (W, then U), NQ5 for stages 1 to 3 and one for
// the 7 angles of the last stage, the groups of a stage in a region of
// their own
constexpr int NQA = QBITS;                            // bits of an angle
constexpr int NQB = QOFS;                             // bits of a stored angle
constexpr int NQR = 7;                                // angles per word
constexpr int NQ5 = (NROT + NQR - 1) / NQR;           // words of 28 angles (4)
constexpr int NQPAD = NQ5 * NQR - NROT;               // unused places of the first word (0)
constexpr int NQLAST = (NROT_LAST + NQPAD + NQR - 1) / NQR;   // words of the last stage (1)
constexpr int NQG = (NSTAGES + 1) * NQ5 + NQLAST;     // words per group (21)
constexpr int NQMEM = NCOLS * NG * NQG;               // angle memory words of a band
constexpr int qgWords(int s) { return s == 0 ? 2 * NQ5 : (s == NSTAGES ? NQLAST : NQ5); }
constexpr int qgAddr(int s, int grp) { return s == 0 ? grp * 2 * NQ5 : NCOLS * NG * (s + 1) * NQ5 + grp * qgWords(s); }
static_assert(NQA >= 2 && NQA <= 18, "an angle has 2 to 18 bits (q_t)");
static_assert(NQB >= 2 && NQB <= NQA, "a stored angle has 2 to QBITS bits");

// Small parameters (prm): Cvh (16 x 16, column-major), Mask, V0 MusW, MusU,
// V0t MusW, MusU, stage Mus and SynMus (4 x 8 each), the centers of the
// angles (integers in units of 2^(1-QBITS): stage 0 W and U, stages 1 to 3,
// the 7 of stage 4; 147, zeros for QOFS = QBITS)
constexpr int P_CVH = 0, P_MASK = 256, P_V0W = 272, P_V0U = 280, P_V0TW = 288, P_V0TU = 296;
constexpr int P_MUS = 304, P_SYNMUS = 336, P_CTR = 368;
constexpr int NPRM = P_CTR + (NSTAGES + 1) * NROT + NROT_LAST;   // 515

// Statistics: mu, sigma (stats_in) and the sums (stats_out), NEST x NDEC each
constexpr int NSTATS = NEST * NDEC;

struct blk_t {                    // the coefficients of a block
    float v[NDEC];
};
typedef ap_fixed<18, 1> q_t;      // angle / pi
typedef ap_fixed<20, 2> stab_t;   // sin(pi i/1024), i = 0..2048 (sin_table.h)
constexpr int SIN_N = 2049;
typedef ap_uint<NQB * NB * NQR> qword_t;   // NQR angles of the NB blocks of a group
typedef ap_int<NQA> qctr_t;                 // the center of an angle

} // namespace bdf

extern "C" void salsun2d_band_df(hls::burst_maxi<float> src, const float *prm, const estdf::wraw_t *wmem,
                                 const estdf::braw_t *bmem, const float *lnmem,
                                 const float *stats_in, float *dst, float *stats_out,
                                 int first_band, int num_bands);

#endif
