// band_df.cpp: see band_df.h
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
#include "band_df.h"

#include "sin_table.h"

namespace bdf {

// Stage configuration (salsun2d_hls_layout): shifts [v h] and targets
// (1: the difference half is shifted, 2: the sum half)
constexpr int SHIFT[NSTAGES][2] = {{0, 1}, {0, -1}, {1, 0}, {-1, 0}};
constexpr int TARGET[NSTAGES] = {1, 2, 1, 2};
constexpr int SYNSHIFT[NSTAGES][2] = {{0, -1}, {0, 1}, {-1, 0}, {1, 0}};
constexpr int SYNTARGET[NSTAGES] = {1, 2, 1, 2};
constexpr int CH0[NEST] = {1, 8, 8, 8, 8};    // first input channel of each estimator (0-based)
constexpr int NCH[NEST] = {15, 8, 8, 8, 8};
constexpr int NZ0 = 7;                         // leading zero angles of W (stage 0)

// Pairs of the Givens rotations: (0,1), (0,2), ..., (0,7), (1,2), ..., (6,7)
const int GTOP[NROT] = {0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 5, 5, 6};
const int GBTM[NROT] = {1, 2, 3, 4, 5, 6, 7, 2, 3, 4, 5, 6, 7, 3, 4, 5, 6, 7, 4, 5, 6, 7, 5, 6, 7, 6, 7, 7};

int wrap(int i, int n)
{
    return i < 0 ? i + n : (i >= n ? i - n : i);
}

// sin(pi q) and cos(pi q) from the table (sin_table.h) interpolated linearly
float sinpi(q_t q)
{
    const ap_uint<18> u = q.range(17, 0);      // (q mod 2) * 2^17
    const ap_uint<11> i = u >> 7;
    const ap_uint<7> fr = u & 127;
    const stab_t t0 = SIN_TABLE[i], t1 = SIN_TABLE[i + 1];
    const ap_fixed<32, 2> s = t0 + (t1 - t0) * fr / 128;
    return (float)s;
}
float cospi(q_t q)
{
    q_t h = 0.5;
    q_t qq = q + h;                            // wraps into [-1, 1)
    return sinpi(qq);
}

// Angle (radians, accumulator format) into theta/pi [18 17]: rounded to
// [27 21], times 1/pi in [18 18], rounded and wrapped (salsun2d_round_angle)
q_t toAngle(acc_t v)
{
    const ap_fixed<27, 6, AP_RND, AP_SAT> a = v;
    const ap_ufixed<18, 0> c = 0.31830988618379067;   // 1/pi, rounded to 18 bits
    const ap_fixed<18, 1, AP_RND, AP_WRAP> q = a * c;
    return q;
}

//---------------------------------------------------------------------------
// Block DCT of the band: src (column-major frame) -> Y. Rows of the band
// with halo wrap around the frame (circular boundary).
void blockDct(const float *src, const float *prm, int r0, blk_t Y[NB][DEPTH])
{
    float cvh[NDEC][NDEC];
#pragma HLS ARRAY_PARTITION variable = cvh type = complete dim = 2
    for (int i = 0; i < NDEC * NDEC; i++) {
#pragma HLS PIPELINE II = 1
        cvh[i % NDEC][i / NDEC] = prm[P_CVH + i];        // Cvh(k,n) at k + 16 n
    }
    float px[MX][SUBROWS * MY];
#pragma HLS ARRAY_PARTITION variable = px type = complete dim = 1
columns:
    for (int c = 0; c < NCOLS; c++) {
    load:
        for (int ih = 0; ih < MX; ih++) {
            for (int i = 0; i < SUBROWS * MY; i++) {
#pragma HLS PIPELINE II = 1
                const int brow = wrap(r0 - HALO + i / MY, NROWS);
                px[ih][i] = src[(c * MX + ih) * SZY + brow * MY + i % MY];
            }
        }
    blocks:
        for (int r = 0; r < SUBROWS; r++) {
            blk_t y;
            for (int k = 0; k < NDEC; k++) {
#pragma HLS PIPELINE II = 1
                float acc = 0.0f;
                for (int ih = 0; ih < MX; ih++) {
                    for (int iv = 0; iv < MY; iv++) {
                        acc += cvh[k][iv + ih * MY] * px[ih][r * MY + iv];
                    }
                }
                y.v[k] = acc;
            }
            Y[r % NB][c * NG + r / NB] = y;
        }
    }
}

// Inverse block DCT of the band rows (without halo) -> dst
void blockIdct(const blk_t Y[NB][DEPTH], const float *prm, int r0, float *dst)
{
    float cvh[NDEC][NDEC];
#pragma HLS ARRAY_PARTITION variable = cvh type = complete dim = 1
    for (int i = 0; i < NDEC * NDEC; i++) {
#pragma HLS PIPELINE II = 1
        cvh[i % NDEC][i / NDEC] = prm[P_CVH + i];
    }
    float px[MX][BAND * MY];
#pragma HLS ARRAY_PARTITION variable = px type = complete dim = 1
columns:
    for (int c = 0; c < NCOLS; c++) {
    blocks:
        for (int rr = 0; rr < BAND; rr++) {
            const int r = HALO + rr;
            const blk_t y = Y[r % NB][c * NG + r / NB];
            for (int n = 0; n < NDEC; n++) {
#pragma HLS PIPELINE II = 1
                float acc = 0.0f;
                for (int k = 0; k < NDEC; k++) {
                    acc += cvh[k][n] * y.v[k];
                }
                px[n / MY][rr * MY + n % MY] = acc;
            }
        }
    store:
        for (int ih = 0; ih < MX; ih++) {
            for (int i = 0; i < BAND * MY; i++) {
#pragma HLS PIPELINE II = 1
                dst[(c * MX + ih) * SZY + r0 * MY + i] = px[ih][i];
            }
        }
    }
}

//---------------------------------------------------------------------------
// Atom extension: the difference (target 1) or sum (target 2) half of each
// block taken from the block shifted by [sv sh] (circular in the band rows
// and the frame columns)
void atomExtension(const blk_t Yin[NB][DEPTH], blk_t Yout[NB][DEPTH], int sv, int sh, int target)
{
columns:
    for (int c = 0; c < NCOLS; c++) {
        const int cs = wrap(c - sh, NCOLS);
    rows:
        for (int r = 0; r < SUBROWS; r++) {
#pragma HLS PIPELINE II = 1
            const int rs = wrap(r - sv, SUBROWS);
            const blk_t y = Yin[r % NB][c * NG + r / NB];
            const blk_t z = Yin[rs % NB][cs * NG + rs / NB];
            blk_t o;
            for (int j = 0; j < PS; j++) {
                float ys, yd;
                if (target == 1) {
                    ys = y.v[j] + y.v[PS + j];
                    yd = z.v[j] - z.v[PS + j];
                } else {
                    ys = z.v[j] + z.v[PS + j];
                    yd = y.v[j] - y.v[PS + j];
                }
                o.v[j] = 0.5f * (ys + yd);
                o.v[PS + j] = 0.5f * (ys - yd);
            }
            Yout[r % NB][c * NG + r / NB] = o;
        }
    }
}

// Coefficient mask (in place)
void mask(blk_t Y[NB][DEPTH], const float *prm)
{
    float m[NDEC];
#pragma HLS ARRAY_PARTITION variable = m type = complete
    for (int k = 0; k < NDEC; k++) m[k] = prm[P_MASK + k];
    for (int i = 0; i < DEPTH; i++) {
        for (int b = 0; b < NB; b++) {
#pragma HLS PIPELINE II = 1
            blk_t y = Y[b][i];
            for (int k = 0; k < NDEC; k++) y.v[k] = y.v[k] * m[k];
            Y[b][i] = y;
        }
    }
}

//---------------------------------------------------------------------------
// Analysis stage s (0: initial rotation, 1..4: intermediate)

struct group_blk_t {              // the NB blocks of a group
    blk_t b[NB];
};

// Gathers the standardized features of every group of the band for the
// estimator of stage s (one row of NB lanes per cycle), forwards the
// blocks themselves to the rotation, and adds the channel sums of the
// rows sumFirst..sumLast for the statistics of the next frame
void gather(const blk_t Y[NB][DEPTH], const float *stats_in, int s, int sumFirst, int sumLast,
            hls::stream<row_t> &features, hls::stream<group_blk_t> &blocks, float sums[2][NDEC])
{
    const int ch0 = CH0[s], nCh = NCH[s];
    float mu[NDEC], invSigma[NDEC];
#pragma HLS ARRAY_PARTITION variable = mu type = complete
#pragma HLS ARRAY_PARTITION variable = invSigma type = complete
    for (int j = 0; j < NDEC; j++) {
#pragma HLS PIPELINE II = 1
        mu[j] = stats_in[s * NDEC + j];
        invSigma[j] = 1.0f / stats_in[NSTATS + s * NDEC + j];
    }
    float s1[NB][NDEC], s2[NB][NDEC];
#pragma HLS ARRAY_PARTITION variable = s1 type = complete dim = 0
#pragma HLS ARRAY_PARTITION variable = s2 type = complete dim = 0
    for (int b = 0; b < NB; b++)
        for (int j = 0; j < NDEC; j++) {
            s1[b][j] = 0.0f;
            s2[b][j] = 0.0f;
        }
    blk_t nb[NB];
#pragma HLS ARRAY_PARTITION variable = nb type = complete
columns:
    for (int c = 0; c < NCOLS; c++) {
    groups:
        for (int g = 0; g < NG; g++) {
            group_blk_t own;
        own_blocks:
            for (int b = 0; b < NB; b++) {
#pragma HLS UNROLL
                own.b[b] = Y[b][c * NG + g];
            }
            blocks.write(own);
        sums_loop:
            for (int j = 0; j < NDEC; j++) {
#pragma HLS PIPELINE II = 1
                for (int b = 0; b < NB; b++) {
                    const int r = g * NB + b;
                    if (r >= sumFirst && r <= sumLast && j >= ch0) {
                        const float v = own.b[b].v[j];
                        s1[b][j] += v;
                        s2[b][j] += v * v;
                    }
                }
            }
        neighbors:
            for (int n = 0; n < 9; n++) {               // (iv, ih) with ih fastest
                const int vshift = 1 - n / 3, hshift = 1 - n % 3;
                const int cs = wrap(c - hshift, NCOLS);
                for (int kb = 0; kb < NB; kb++) {         // bank kb holds the row of lane b
#pragma HLS UNROLL
                    const int b = (kb + vshift + NB) % NB;
                    const int rs = wrap(g * NB + b - vshift, SUBROWS);
                    nb[b] = Y[kb][cs * NG + rs / NB];
                }
            channels:
                for (int j = 0; j < nCh; j++) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT max = 15
                    row_t f;
                    for (int b = 0; b < NB; b++) {
                        f.v[b] = (nb[b].v[ch0 + j] - mu[ch0 + j]) * invSigma[ch0 + j];
                    }
                    features.write(f);
                }
            }
        }
    }
    for (int j = 0; j < NDEC; j++) {
        float a1 = 0.0f, a2 = 0.0f;
        for (int b = 0; b < NB; b++) {
            a1 += s1[b][j];
            a2 += s2[b][j];
        }
        sums[0][j] = a1;
        sums[1][j] = a2;
    }
}

// Rotations of the blocks of a column by the angles of stage s: stage 0
// rotates the symmetric (W) and antisymmetric (U) halves, the others the
// antisymmetric half (in the last stage only the first PA-1 rotations; the
// outputs but channel PS are zero as the mask would make them). Then the
// sign flips (mus). Writes the angles to qmem for the synthesis.
void rotate(hls::stream<arow_t> &angles, hls::stream<group_blk_t> &blocks, const float *prm, int s,
            blk_t Y[NB][DEPTH], qword_t *qmem)
{
    const int nRot = (s == NSTAGES) ? NROT_LAST : NROT;
    const int nA = NAK[s];
    float musW[PS], musU[PA];
#pragma HLS ARRAY_PARTITION variable = musW type = complete
#pragma HLS ARRAY_PARTITION variable = musU type = complete
    for (int j = 0; j < PA; j++) {
#pragma HLS PIPELINE II = 1
        musW[j] = prm[P_V0W + j];
        musU[j] = (s == 0) ? prm[P_V0U + j] : prm[P_MUS + (s - 1) * PA + j];
    }
    q_t qw[NG][NB][NROT], qu[NG][NB][NROT];
#pragma HLS ARRAY_PARTITION variable = qw type = complete dim = 2
#pragma HLS ARRAY_PARTITION variable = qu type = complete dim = 2
    float vs[NG][NB][PS], va[NG][NB][PA];
#pragma HLS ARRAY_PARTITION variable = vs type = complete dim = 2
#pragma HLS ARRAY_PARTITION variable = vs type = complete dim = 3
#pragma HLS ARRAY_PARTITION variable = va type = complete dim = 2
#pragma HLS ARRAY_PARTITION variable = va type = complete dim = 3
    group_blk_t own[NG];
columns:
    for (int c = 0; c < NCOLS; c++) {
    read:
        for (int g = 0; g < NG; g++) {
            own[g] = blocks.read();
            for (int b = 0; b < NB; b++) {
#pragma HLS UNROLL
                for (int j = 0; j < PS; j++) {
                    vs[g][b][j] = own[g].b[b].v[j];
                    va[g][b][j] = own[g].b[b].v[PS + j];
                }
            }
        read_angles:
            for (int a = 0; a < nA; a++) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT max = 49
                const arow_t z = angles.read();
                for (int b = 0; b < NB; b++) {
                    const q_t q = toAngle(z.v[b]);
                    if (s == 0) {
                        if (a < NROT - NZ0) {
                            qw[g][b][NZ0 + a] = q;
                        } else {
                            qu[g][b][a - (NROT - NZ0)] = q;
                        }
                    } else if (a < NROT) {
                        qu[g][b][a] = q;
                    }
                }
            }
            if (s == 0) {
                for (int b = 0; b < NB; b++)
                    for (int a = 0; a < NZ0; a++) qw[g][b][a] = 0;
            }
        }
    rotations:
        for (int k = 0; k < nRot; k++) {
            for (int g = 0; g < NG; g++) {
#pragma HLS PIPELINE II = 1
#pragma HLS DEPENDENCE variable = vs type = inter direction = RAW dependent = true distance = NG
#pragma HLS DEPENDENCE variable = va type = inter direction = RAW dependent = true distance = NG
                const int t = GTOP[k], bt = GBTM[k];
                for (int b = 0; b < NB; b++) {
                    const float cu = cospi(qu[g][b][k]), su = sinpi(qu[g][b][k]);
                    const float at = va[g][b][t], ab = va[g][b][bt];
                    va[g][b][t] = cu * at - su * ab;
                    va[g][b][bt] = su * at + cu * ab;
                    if (s == 0) {
                        const float cw = cospi(qw[g][b][k]), sw = sinpi(qw[g][b][k]);
                        const float wt = vs[g][b][t], wb = vs[g][b][bt];
                        vs[g][b][t] = cw * wt - sw * wb;
                        vs[g][b][bt] = sw * wt + cw * wb;
                    }
                }
            }
        }
    write:
        for (int g = 0; g < NG; g++) {
#pragma HLS PIPELINE II = 1
            for (int b = 0; b < NB; b++) {
                blk_t o = own[g].b[b];
                for (int j = 0; j < PA; j++) {
                    if (s == 0) o.v[j] = musW[j] * vs[g][b][j];
                    if (s == NSTAGES && j > 0) {
                        o.v[PS + j] = 0.0f;
                    } else {
                        o.v[PS + j] = musU[j] * va[g][b][j];
                    }
                }
                Y[b][c * NG + g] = o;
            }
        }
    store_angles:
        for (int g = 0; g < NG; g++) {
            for (int b = 0; b < NB; b++) {
                const int blk = c * SUBROWS + g * NB + b;
                for (int k = 0; k < NROT; k++) {
#pragma HLS PIPELINE II = 1
                    if (s == 0) {
                        qmem[blk * 2 * NROT + k] = qw[g][b][k].range(17, 0);
                        qmem[blk * 2 * NROT + NROT + k] = qu[g][b][k].range(17, 0);
                    } else if (k < nRot) {
                        const int base = SUBROWS * NCOLS * (2 * NROT + (s - 1) * NROT);
                        qmem[base + blk * nRot + k] = qu[g][b][k].range(17, 0);
                    }
                }
            }
        }
    }
}

// One analysis stage: features -> estimator -> rotations, as a dataflow
void analysisStage(const blk_t Yin[NB][DEPTH], blk_t Yout[NB][DEPTH], const float *prm,
                   const wword_t *wmem, const acc_t *bmem, const lnpar_t *lnmem,
                   const float *stats_in, qword_t *qmem, int s, int sumFirst, int sumLast,
                   bool load, float sums[2][NDEC])
{
#pragma HLS DATAFLOW
    hls::stream<row_t, FIFO> features;
    hls::stream<arow_t, FIFO> angles;
    hls::stream<group_blk_t, NG + 24> blocks;   // the groups the estimator holds, and a column
    gather(Yin, stats_in, s, sumFirst, sumLast, features, blocks, sums);
    est_engine(features, angles, wmem, bmem, lnmem, s, NCOLS * NG, load);
    rotate(angles, blocks, prm, s, Yout, qmem);
}

//---------------------------------------------------------------------------
// Synthesis: the transposed rotations of stage s (s = 0: the final
// rotation, W and U), after the sign flips, from the angles in qmem
void synthesisRotation(const blk_t Yin[NB][DEPTH], blk_t Yout[NB][DEPTH], const float *prm, int s,
                       const qword_t *qmem)
{
    const int nRot = (s == NSTAGES) ? NROT_LAST : NROT;
    float musW[PS], musU[PA];
#pragma HLS ARRAY_PARTITION variable = musW type = complete
#pragma HLS ARRAY_PARTITION variable = musU type = complete
    for (int j = 0; j < PA; j++) {
#pragma HLS PIPELINE II = 1
        musW[j] = prm[P_V0TW + j];
        musU[j] = (s == 0) ? prm[P_V0TU + j] : prm[P_SYNMUS + (s - 1) * PA + j];
    }
    q_t qw[NG][NB][NROT], qu[NG][NB][NROT];
#pragma HLS ARRAY_PARTITION variable = qw type = complete dim = 2
#pragma HLS ARRAY_PARTITION variable = qu type = complete dim = 2
    float vs[NG][NB][PS], va[NG][NB][PA];
#pragma HLS ARRAY_PARTITION variable = vs type = complete dim = 2
#pragma HLS ARRAY_PARTITION variable = vs type = complete dim = 3
#pragma HLS ARRAY_PARTITION variable = va type = complete dim = 2
#pragma HLS ARRAY_PARTITION variable = va type = complete dim = 3
    blk_t own[NG][NB];
#pragma HLS ARRAY_PARTITION variable = own type = complete dim = 2
columns:
    for (int c = 0; c < NCOLS; c++) {
    read:
        for (int g = 0; g < NG; g++) {
            for (int b = 0; b < NB; b++) {
#pragma HLS UNROLL
                own[g][b] = Yin[b][c * NG + g];
                for (int j = 0; j < PS; j++) {
                    vs[g][b][j] = musW[j] * own[g][b].v[j];
                    va[g][b][j] = musU[j] * own[g][b].v[PS + j];
                }
            }
            for (int b = 0; b < NB; b++) {
                const int blk = c * SUBROWS + g * NB + b;
                for (int k = 0; k < NROT; k++) {
#pragma HLS PIPELINE II = 1
                    if (s == 0) {
                        qw[g][b][k].range(17, 0) = qmem[blk * 2 * NROT + k];
                        qu[g][b][k].range(17, 0) = qmem[blk * 2 * NROT + NROT + k];
                    } else if (k < nRot) {
                        const int base = SUBROWS * NCOLS * (2 * NROT + (s - 1) * NROT);
                        qu[g][b][k].range(17, 0) = qmem[base + blk * nRot + k];
                    }
                }
            }
        }
    rotations:
        for (int kk = 0; kk < nRot; kk++) {
            const int k = nRot - 1 - kk;               // transposes in reverse order
            for (int g = 0; g < NG; g++) {
#pragma HLS PIPELINE II = 1
#pragma HLS DEPENDENCE variable = vs type = inter direction = RAW dependent = true distance = NG
#pragma HLS DEPENDENCE variable = va type = inter direction = RAW dependent = true distance = NG
                const int t = GTOP[k], bt = GBTM[k];
                for (int b = 0; b < NB; b++) {
                    const float cu = cospi(qu[g][b][k]), su = sinpi(qu[g][b][k]);
                    const float at = va[g][b][t], ab = va[g][b][bt];
                    va[g][b][t] = cu * at + su * ab;
                    va[g][b][bt] = cu * ab - su * at;
                    if (s == 0) {
                        const float cw = cospi(qw[g][b][k]), sw = sinpi(qw[g][b][k]);
                        const float wt = vs[g][b][t], wb = vs[g][b][bt];
                        vs[g][b][t] = cw * wt + sw * wb;
                        vs[g][b][bt] = cw * wb - sw * wt;
                    }
                }
            }
        }
    write:
        for (int g = 0; g < NG; g++) {
#pragma HLS PIPELINE II = 1
            for (int b = 0; b < NB; b++) {
                blk_t o = own[g][b];
                for (int j = 0; j < PA; j++) {
                    if (s == 0) o.v[j] = vs[g][b][j];
                    o.v[PS + j] = va[g][b][j];
                }
                Yout[b][c * NG + g] = o;
            }
        }
    }
}

} // namespace bdf

//---------------------------------------------------------------------------
extern "C" void salsun2d_band_df(const float *src, const float *prm, const estdf::wword_t *wmem,
                                 const estdf::acc_t *bmem, const estdf::lnpar_t *lnmem,
                                 const float *stats_in, float *dst, float *stats_out,
                                 bdf::qword_t *qmem, int first_band, int num_bands)
{
#pragma HLS INTERFACE m_axi port = src offset = slave bundle = gmem0 depth = SZY * SZX
#pragma HLS INTERFACE m_axi port = dst offset = slave bundle = gmem0 depth = SZY * SZX
#pragma HLS INTERFACE m_axi port = prm offset = slave bundle = gmem1 depth = bdf::NPRM
#pragma HLS INTERFACE m_axi port = stats_in offset = slave bundle = gmem1 depth = 2 * bdf::NSTATS
#pragma HLS INTERFACE m_axi port = stats_out offset = slave bundle = gmem1 depth = 2 * bdf::NSTATS
#pragma HLS INTERFACE m_axi port = wmem offset = slave bundle = gmem2 depth = estdf::NWORDS
#pragma HLS INTERFACE m_axi port = bmem offset = slave bundle = gmem3 depth = estdf::NBIAS
#pragma HLS INTERFACE m_axi port = lnmem offset = slave bundle = gmem3 depth = estdf::NLN
#pragma HLS INTERFACE m_axi port = qmem offset = slave bundle = gmem4 depth = bdf::NQMEM
    using namespace bdf;
    static blk_t A[NB][DEPTH], B[NB][DEPTH];
#pragma HLS ARRAY_PARTITION variable = A type = complete dim = 1
#pragma HLS ARRAY_PARTITION variable = B type = complete dim = 1
#pragma HLS AGGREGATE variable = A
#pragma HLS AGGREGATE variable = B
#pragma HLS BIND_STORAGE variable = A type = ram_2p impl = uram
#pragma HLS BIND_STORAGE variable = B type = ram_2p impl = uram
#pragma HLS ALLOCATION function instances = analysisStage limit = 1
    float acc[2][NEST][NDEC];
    for (int i = 0; i < 2 * NSTATS; i++) acc[i / NSTATS][(i % NSTATS) / NDEC][i % NDEC] = 0.0f;
    bool load = true;                          // the estimator parameters, once per call
    int prevStart = -BAND;
bands:
    for (int k = 0; k < NBANDS; k++) {
        int r0 = k * BAND;
        if (r0 + BAND > NROWS) r0 = NROWS - BAND;          // move the last band up
        const int repeated = (prevStart + BAND > r0) ? (prevStart + BAND - r0) : 0;
        prevStart = r0;
        if (k < first_band || k >= first_band + num_bands) continue;
        const int sumFirst = HALO + repeated, sumLast = HALO + BAND - 1;

        blockDct(src, prm, r0, A);
        float sums[2][NDEC];
        for (int s = 0; s <= NSTAGES; s++) {
            if (s > 0) atomExtension(B, A, SHIFT[s - 1][0], SHIFT[s - 1][1], TARGET[s - 1]);
            analysisStage(A, B, prm, wmem, bmem, lnmem, stats_in, qmem, s, sumFirst, sumLast, load,
                          sums);
            load = false;
            for (int j = 0; j < NDEC; j++) {
                acc[0][s][j] += sums[0][j];
                acc[1][s][j] += sums[1][j];
            }
        }
        mask(B, prm);
        for (int s = NSTAGES; s >= 1; s--) {
            synthesisRotation(B, A, prm, s, qmem);
            atomExtension(A, B, SYNSHIFT[s - 1][0], SYNSHIFT[s - 1][1], SYNTARGET[s - 1]);
        }
        synthesisRotation(B, A, prm, 0, qmem);
        blockIdct(A, prm, r0, dst);
    }
    for (int i = 0; i < 2 * NSTATS; i++) stats_out[i] = acc[i / NSTATS][(i % NSTATS) / NDEC][i % NDEC];
}
