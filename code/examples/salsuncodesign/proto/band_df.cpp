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

// The sign parameters (mus) are +-1 and the mask 0 or 1: they are applied
// as sign flips and zeros, which equal the products
float flip(bool neg, float v)
{
    return neg ? -v : v;
}

// Givens rotation of channels t and bt of the PA values of a block,
// [c -s; s c] or its transpose; all channels are read and written, so
// that every channel memory has one read and one write per cycle
void givens(float v[PA], int t, int bt, float c, float s, bool transposed)
{
    float x[PA];
    for (int j = 0; j < PA; j++) x[j] = v[j];
    const float vt = x[t], vb = x[bt];
    const float nt = transposed ? c * vt + s * vb : c * vt - s * vb;
    const float nb = transposed ? c * vb - s * vt : s * vt + c * vb;
    for (int j = 0; j < PA; j++) v[j] = (j == t) ? nt : ((j == bt) ? nb : x[j]);
}

// 28 angles to and from a memory word (constant bit positions)
qword_t pack(const q_t q[NROT])
{
    qword_t w = 0;
    for (int k = 0; k < NROT; k++) w.range(18 * k + 17, 18 * k) = q[k].range(17, 0);
    return w;
}
void unpack(qword_t w, q_t q[NROT])
{
    for (int k = 0; k < NROT; k++) q[k].range(17, 0) = w.range(18 * k + 17, 18 * k);
}

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
void blockDct(const float *src, const float prm[NPRM], int r0, blk_t Y[NBLK])
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
            Y[c * SUBROWS + r] = y;
        }
    }
}

// Inverse block DCT of the band rows (without halo) -> dst
void blockIdct(const blk_t Y[NBLK], const float prm[NPRM], int r0, float *dst)
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
            const blk_t y = Y[c * SUBROWS + r];
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
void atomExtension(const blk_t Yin[NBLK], blk_t Yout[NBLK], int sv, int sh, int target)
{
columns:
    for (int c = 0; c < NCOLS; c++) {
        const int cs = wrap(c - sh, NCOLS);
    rows:
        for (int r = 0; r < SUBROWS; r++) {
#pragma HLS PIPELINE II = 1
            const int rs = wrap(r - sv, SUBROWS);
            const blk_t y = Yin[c * SUBROWS + r];
            const blk_t z = Yin[cs * SUBROWS + rs];
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
            Yout[c * SUBROWS + r] = o;
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
void gather(const blk_t Y[NBLK], const float stats_in[2 * NSTATS], int s, int sumFirst, int sumLast,
            hls::stream<row_t> &features, hls::stream<blk_t> &blocks, float sums[2][NDEC])
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
                own.b[b] = Y[c * SUBROWS + g * NB + b];
                blocks.write(own.b[b]);
            }
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
                for (int b = 0; b < NB; b++) {            // read over three cycles (two ports)
                    const int rs = wrap(g * NB + b - vshift, SUBROWS);
                    nb[b] = Y[cs * SUBROWS + rs];
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
// sign flips (mus), and in the last stage the coefficient mask. Writes the
// angles to qmem for the synthesis.
void rotate(hls::stream<arow_t> &angles, hls::stream<blk_t> &blocks, const float prm[NPRM], int s,
            blk_t Y[NBLK], qword_t *qmem)
{
    const int nRot = (s == NSTAGES) ? NROT_LAST : NROT;
    const int nA = NAK[s];
    bool negW[PS], negU[PA], drop[NDEC];
#pragma HLS ARRAY_PARTITION variable = negW type = complete
#pragma HLS ARRAY_PARTITION variable = negU type = complete
#pragma HLS ARRAY_PARTITION variable = drop type = complete
    for (int j = 0; j < PA; j++) {
#pragma HLS PIPELINE II = 1
        negW[j] = prm[P_V0W + j] < 0;
        negU[j] = ((s == 0) ? prm[P_V0U + j] : prm[P_MUS + (s - 1) * PA + j]) < 0;
    }
    for (int j = 0; j < NDEC; j++) drop[j] = (s == NSTAGES) && prm[P_MASK + j] == 0.0f;
    q_t qw[NG][NB][NROT], qu[NG][NB][NROT];   // the angles of each block
#pragma HLS ARRAY_PARTITION variable = qw type = complete dim = 3
#pragma HLS ARRAY_PARTITION variable = qu type = complete dim = 3
    float vs[NG][NB][PS], va[NG][NB][PA];
#pragma HLS ARRAY_PARTITION variable = vs type = complete dim = 3
#pragma HLS ARRAY_PARTITION variable = va type = complete dim = 3
    // The blocks and angles are read one block per cycle (the estimator
    // takes longer per group), the symmetric half kept in vs in every stage
columns:
    for (int c = 0; c < NCOLS; c++) {
    read:
        for (int g = 0; g < NG; g++) {
            arow_t z;
            for (int b = 0; b < NB; b++) {
#pragma HLS PIPELINE II = 1
                const blk_t y = blocks.read();
                for (int j = 0; j < PS; j++) {
                    vs[g][b][j] = y.v[j];
                    va[g][b][j] = y.v[PS + j];
                }
                for (int k = 0; k < NZ0; k++) qw[g][b][k] = 0;   // the leading zero angles of W
            }
        read_angles:
            for (int a = 0; a < nA; a++) {
#pragma HLS LOOP_TRIPCOUNT max = 49
                for (int b = 0; b < NB; b++) {
#pragma HLS PIPELINE II = 1
                    if (b == 0) z = angles.read();
                    const q_t q = toAngle(z.v[b]);
                    if (s == 0 && a < NROT - NZ0) {
                        qw[g][b][NZ0 + a] = q;
                    } else {
                        const int k = (s == 0) ? a - (NROT - NZ0) : a;
                        if (k < NROT) qu[g][b][k] = q;
                    }
                }
            }
        }
    rotations:
        for (int k = 0; k < nRot; k++) {
            for (int g = 0; g < NG; g++) {
                const int t = GTOP[k], bt = GBTM[k];
                for (int b = 0; b < NB; b++) {       // one block per cycle
#pragma HLS PIPELINE II = 1
#pragma HLS DEPENDENCE variable = vs type = inter direction = RAW dependent = true distance = NG * NB
#pragma HLS DEPENDENCE variable = va type = inter direction = RAW dependent = true distance = NG * NB
                    const q_t qk = qu[g][b][k];
                    givens(va[g][b], t, bt, cospi(qk), sinpi(qk), false);
                    if (s == 0) {
                        const q_t qwk = qw[g][b][k];
                        givens(vs[g][b], t, bt, cospi(qwk), sinpi(qwk), false);
                    }
                }
            }
        }
    write:
        for (int g = 0; g < NG; g++) {
            for (int b = 0; b < NB; b++) {
#pragma HLS PIPELINE II = 1
                blk_t o;
                for (int j = 0; j < PA; j++) {
                    o.v[j] = (s == 0) ? flip(negW[j], vs[g][b][j]) : vs[g][b][j];
                    if (s == NSTAGES && j > 0) {
                        o.v[PS + j] = 0.0f;
                    } else {
                        o.v[PS + j] = flip(negU[j], va[g][b][j]);
                    }
                }
                for (int j = 0; j < NDEC; j++) {
                    if (drop[j]) o.v[j] = 0.0f;
                }
                Y[c * SUBROWS + g * NB + b] = o;
            }
        }
    store_angles:
        for (int g = 0; g < NG; g++) {
            for (int b = 0; b < NB; b++) {
#pragma HLS PIPELINE II = 2
                const int blk = c * SUBROWS + g * NB + b;
                if (s == 0) {
                    qmem[blk * 2] = pack(qw[g][b]);
                    qmem[blk * 2 + 1] = pack(qu[g][b]);
                } else {
                    qmem[SUBROWS * NCOLS * qwOffset(s) + blk] = pack(qu[g][b]);
                }
            }
        }
    }
}

// One analysis stage: features -> estimator -> rotations, as a dataflow
void analysisStage(const blk_t Yin[NBLK], blk_t Yout[NBLK], const float prm[NPRM],
                   const wraw_t *wmem, const braw_t *bmem, const float *lnmem,
                   const float stats_in[2 * NSTATS], qword_t *qmem, int s, int sumFirst, int sumLast,
                   bool load, float sums[2][NDEC])
{
#pragma HLS DATAFLOW
    hls::stream<row_t, FIFO> features;
    hls::stream<arow_t, FIFO> angles;
    hls::stream<blk_t, (NG + 24) * NB> blocks;  // the groups the estimator holds, and a column
    gather(Yin, stats_in, s, sumFirst, sumLast, features, blocks, sums);
    est_engine(features, angles, wmem, bmem, lnmem, s, NCOLS * NG, load);
    rotate(angles, blocks, prm, s, Yout, qmem);
}

//---------------------------------------------------------------------------
// Synthesis: the transposed rotations of stage s (s = 0: the final
// rotation, W and U), after the sign flips, from the angles in qmem
void synthesisRotation(const blk_t Yin[NBLK], blk_t Yout[NBLK], const float prm[NPRM], int s,
                       const qword_t *qmem)
{
    const int nRot = (s == NSTAGES) ? NROT_LAST : NROT;
    bool negW[PS], negU[PA];
#pragma HLS ARRAY_PARTITION variable = negW type = complete
#pragma HLS ARRAY_PARTITION variable = negU type = complete
    for (int j = 0; j < PA; j++) {
#pragma HLS PIPELINE II = 1
        negW[j] = prm[P_V0TW + j] < 0;
        negU[j] = ((s == 0) ? prm[P_V0TU + j] : prm[P_SYNMUS + (s - 1) * PA + j]) < 0;
    }
    q_t qw[NG][NB][NROT], qu[NG][NB][NROT];   // the angles of each block
#pragma HLS ARRAY_PARTITION variable = qw type = complete dim = 3
#pragma HLS ARRAY_PARTITION variable = qu type = complete dim = 3
    float vs[NG][NB][PS], va[NG][NB][PA];
#pragma HLS ARRAY_PARTITION variable = vs type = complete dim = 3
#pragma HLS ARRAY_PARTITION variable = va type = complete dim = 3
columns:
    for (int c = 0; c < NCOLS; c++) {
    read:
        for (int g = 0; g < NG; g++) {
            for (int b = 0; b < NB; b++) {
#pragma HLS PIPELINE II = 2
                const blk_t y = Yin[c * SUBROWS + g * NB + b];
                for (int j = 0; j < PS; j++) {
                    vs[g][b][j] = (s == 0) ? flip(negW[j], y.v[j]) : y.v[j];
                    va[g][b][j] = flip(negU[j], y.v[PS + j]);
                }
                const int blk = c * SUBROWS + g * NB + b;
                if (s == 0) {
                    unpack(qmem[blk * 2], qw[g][b]);
                    unpack(qmem[blk * 2 + 1], qu[g][b]);
                } else {
                    unpack(qmem[SUBROWS * NCOLS * qwOffset(s) + blk], qu[g][b]);
                }
            }
        }
    rotations:
        for (int kk = 0; kk < nRot; kk++) {
            const int k = nRot - 1 - kk;               // transposes in reverse order
            for (int g = 0; g < NG; g++) {
                const int t = GTOP[k], bt = GBTM[k];
                for (int b = 0; b < NB; b++) {       // one block per cycle
#pragma HLS PIPELINE II = 1
#pragma HLS DEPENDENCE variable = vs type = inter direction = RAW dependent = true distance = NG * NB
#pragma HLS DEPENDENCE variable = va type = inter direction = RAW dependent = true distance = NG * NB
                    const q_t qk = qu[g][b][k];
                    givens(va[g][b], t, bt, cospi(qk), sinpi(qk), true);
                    if (s == 0) {
                        const q_t qwk = qw[g][b][k];
                        givens(vs[g][b], t, bt, cospi(qwk), sinpi(qwk), true);
                    }
                }
            }
        }
    write:
        for (int g = 0; g < NG; g++) {
            for (int b = 0; b < NB; b++) {
#pragma HLS PIPELINE II = 1
                blk_t o;
                for (int j = 0; j < PA; j++) {
                    o.v[j] = vs[g][b][j];
                    o.v[PS + j] = va[g][b][j];
                }
                Yout[c * SUBROWS + g * NB + b] = o;
            }
        }
    }
}

} // namespace bdf

//---------------------------------------------------------------------------
extern "C" void salsun2d_band_df(const float *src, const float *prm, const estdf::wraw_t *wmem,
                                 const estdf::braw_t *bmem, const float *lnmem,
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
#pragma HLS INTERFACE m_axi port = lnmem offset = slave bundle = gmem3 depth = 2 * estdf::NLN
#pragma HLS INTERFACE m_axi port = qmem offset = slave bundle = gmem4 depth = bdf::NQMEM
    using namespace bdf;
    // The coefficients of the band, one block (16 floats) per word, column
    // by column, in two buffers of two ports (URAM). (Five banks by row with
    // block-wide multiplexers on every access congested the routing.)
    static blk_t A[NBLK], B[NBLK];
#pragma HLS AGGREGATE variable = A
#pragma HLS AGGREGATE variable = B
#pragma HLS BIND_STORAGE variable = A type = ram_t2p impl = uram
#pragma HLS BIND_STORAGE variable = B type = ram_t2p impl = uram
#pragma HLS ALLOCATION function instances = analysisStage limit = 1
    // the small parameters and the statistics, read once (the tasks of a
    // dataflow region cannot share their memory interface)
    float prmL[NPRM], stL[2 * NSTATS];
    for (int i = 0; i < NPRM; i++) {
#pragma HLS PIPELINE II = 1
        prmL[i] = prm[i];
    }
    for (int i = 0; i < 2 * NSTATS; i++) {
#pragma HLS PIPELINE II = 1
        stL[i] = stats_in[i];
    }
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

        blockDct(src, prmL, r0, A);
        float sums[2][NDEC];
        for (int s = 0; s <= NSTAGES; s++) {
            if (s > 0) atomExtension(B, A, SHIFT[s - 1][0], SHIFT[s - 1][1], TARGET[s - 1]);
            analysisStage(A, B, prmL, wmem, bmem, lnmem, stL, qmem, s, sumFirst, sumLast, load,
                          sums);
            load = false;
            for (int j = 0; j < NDEC; j++) {
                acc[0][s][j] += sums[0][j];
                acc[1][s][j] += sums[1][j];
            }
        }
        for (int s = NSTAGES; s >= 1; s--) {
            synthesisRotation(B, A, prmL, s, qmem);
            atomExtension(A, B, SYNSHIFT[s - 1][0], SYNSHIFT[s - 1][1], SYNTARGET[s - 1]);
        }
        synthesisRotation(B, A, prmL, 0, qmem);
        blockIdct(A, prmL, r0, dst);
    }
    for (int i = 0; i < 2 * NSTATS; i++) stats_out[i] = acc[i / NSTATS][(i % NSTATS) / NDEC][i % NDEC];
}
