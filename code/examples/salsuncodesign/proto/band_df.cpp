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
static_assert(NB <= 8, "gather reads the NB blocks of the next neighbor within a neighbor (8 channels or more)");
constexpr int NZ0 = 7;                         // leading zero angles of W (stage 0)
static_assert((NZ0 + NQPAD) / NQR == 1, "rotate zeros the first word of W and starts within the second");

// Pairs of the Givens rotations: (0,1), (0,2), ..., (0,7), (1,2), ..., (6,7)
const int GTOP[NROT] = {0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 5, 5, 6};
const int GBTM[NROT] = {1, 2, 3, 4, 5, 6, 7, 2, 3, 4, 5, 6, 7, 3, 4, 5, 6, 7, 4, 5, 6, 7, 5, 6, 7, 6, 7, 7};

// The sign parameters (mus) are +-1 and the mask 0 or 1: they are applied
// as sign flips and zeros, which equal the products
float flip(bool neg, float v)
{
    return neg ? -v : v;
}

// Givens rotation of channels t and bt of a half (h: the antisymmetric
// one) of a block, [c -s; s c] or its transpose. Every channel is read
// once, and written (if en) only where it changes, so that every channel
// memory has one read and one write per cycle
void givens(float v[NDEC], bool h, int t, int bt, float c, float s, bool transposed, bool en)
{
    const int it = (h ? PS : 0) + t, ib = (h ? PS : 0) + bt;
    float x[NDEC];
    for (int j = 0; j < NDEC; j++) x[j] = v[j];
    const float vt = x[it], vb = x[ib];
    const float nt = transposed ? c * vt + s * vb : c * vt - s * vb;
    const float nb = transposed ? c * vb - s * vt : s * vt + c * vb;
    for (int j = 0; j < NDEC; j++) {
        if (en && (j == it || j == ib)) v[j] = (j == it) ? nt : nb;
    }
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

// Angle (radians, accumulator format) into theta/pi [NQA NQA-1]: rounded
// to [27 21], times 1/pi in [18 18], rounded and wrapped
// (salsun2d_round_angle with NQA bits)
ap_fixed<NQA, 1> toAngle(acc_t v)
{
    const ap_fixed<27, 6, AP_RND, AP_SAT> a = v;
    // 1/pi rounded to 18 bits; below 1/2, so 17 bits hold it and the
    // product fits one DSP (as two DSPs, wide products were wrong on the card)
    const ap_ufixed<17, -1> c = 0.31830988618379067;
    const ap_fixed<NQA, 1, AP_RND, AP_WRAP> q = a * c;
    return q;
}

// The stored angle: the offset of an angle from its center c (units of
// 2^(1-NQA), wrapped), saturated to NQB bits
ap_uint<NQB> encodeAngle(ap_fixed<NQA, 1> q, qctr_t c)
{
    ap_int<NQA> qi;
    qi.range(NQA - 1, 0) = q.range(NQA - 1, 0);
    ap_int<NQA> d = qi - c;                    // wrapped
    if (NQB < NQA) {
        const int lim = 1 << (NQB - 1);
        if (d < -lim) d = -lim;
        if (d > lim - 1) d = lim - 1;
    }
    return d.range(NQB - 1, 0);
}

// The centers of the angles of stage s from prm: W (stage 0) and U
void loadCenters(const float prm[NPRM], int s, qctr_t ctrW[NROT], qctr_t ctrU[NROT])
{
    const int baseU = (s == 0) ? P_CTR + NROT : P_CTR + (s + 1) * NROT;
    const int nRot = (s == NSTAGES) ? NROT_LAST : NROT;
    for (int k = 0; k < NROT; k++) {
#pragma HLS PIPELINE II = 1
        ctrW[k] = (s == 0 && k >= NZ0) ? qctr_t((int)prm[P_CTR + k]) : qctr_t(0);   // zero angles: stored as 0
        ctrU[k] = (k < nRot) ? qctr_t((int)prm[baseU + k]) : qctr_t(0);
    }
}

//---------------------------------------------------------------------------
// Requests the rows of the band with halo for the MX pixel columns of block
// column c, in runs up to the bottom of the frame (then on from the top)
void requestColumn(hls::burst_maxi<float> &src, int c, int row0)
{
    for (int ih = 0; ih < MX; ih++) {
        const int base = (c * MX + ih) * SZY;
        int row = row0;
    runs:
        for (int i0 = 0; i0 < SUBROWS * MY;) {
#pragma HLS LOOP_TRIPCOUNT max = 2
            const int n = (SZY - row < SUBROWS * MY - i0) ? SZY - row : SUBROWS * MY - i0;
            src.read_request(base + row, n);
            i0 += n;
            row = 0;
        }
    }
}

// Block DCT of the band: src (column-major frame) -> Y. Rows of the band
// with halo wrap around the frame (circular boundary). The runs of the
// next column are requested before the pixels of a column are read, so
// that the latency of the bursts overlaps (one run at a time cost about
// 200 cycles each), and the pixels of a column are read while the blocks
// of the previous one are transformed (two buffers)
static_assert(MX * MY == NDEC, "a column of pixels takes as many cycles as its coefficients");
void blockDct(hls::burst_maxi<float> src, const float prm[NPRM], int r0, blk_t Y[NBLK])
{
    float cvh[NDEC][NDEC];
#pragma HLS ARRAY_PARTITION variable = cvh type = complete dim = 2
    for (int i = 0; i < NDEC * NDEC; i++) {
#pragma HLS PIPELINE II = 1
        cvh[i % NDEC][i / NDEC] = prm[P_CVH + i];        // Cvh(k,n) at k + 16 n
    }
    // two buffers, a bank per pixel of a block (iv + ih MY), a word per block row
    float px[2][NDEC][SUBROWS];
#pragma HLS ARRAY_PARTITION variable = px type = complete dim = 1
#pragma HLS ARRAY_PARTITION variable = px type = complete dim = 2
    const int row0 = wrap(r0 - HALO, NROWS) * MY;
    requestColumn(src, 0, row0);
columns:
    for (int c = 0; c <= NCOLS; c++) {
        if (c + 1 < NCOLS) requestColumn(src, c + 1, row0);
        const int wb = c & 1;
        int ih = 0, i = 0;                     // the pixel read: column ih, row i of the band
        int r = 0, k = 0;                      // the coefficient: k of block row r (column c - 1)
        float yv[NDEC];
#pragma HLS ARRAY_PARTITION variable = yv type = complete
    step:
        for (int t = 0; t < SUBROWS * NDEC; t++) {
#pragma HLS PIPELINE II = 1
#pragma HLS DEPENDENCE variable = px inter false
#pragma HLS DEPENDENCE variable = px intra false
            if (c < NCOLS) px[wb][i % MY + ih * MY][i / MY] = src.read();
            if (c > 0) {
                float acc = 0.0f;
                for (int j = 0; j < NDEC; j++) acc += cvh[k][j] * px[1 - wb][j][r];
                yv[k] = acc;
                if (k == NDEC - 1) {
                    blk_t y;
                    for (int j = 0; j < NDEC; j++) y.v[j] = yv[j];
                    Y[(c - 1) * SUBROWS + r] = y;
                }
            }
            if (i == SUBROWS * MY - 1) { i = 0; ih++; } else { i++; }
            if (k == NDEC - 1) { k = 0; r++; } else { k++; }
        }
    }
}

// Inverse block DCT of the band rows (without halo) -> dst, as two tasks:
// the pixels of a column are transformed while those of the previous one
// are stored. The transform sends the pixels in the order of the frame
// (pixel column ih, then the rows), reading a block once per pixel column
void idctTransform(const blk_t Y[NBLK], const float prm[NPRM], hls::stream<float> &pixels)
{
    float cvh[NDEC][NDEC];
#pragma HLS ARRAY_PARTITION variable = cvh type = complete dim = 1
    for (int i = 0; i < NDEC * NDEC; i++) {
#pragma HLS PIPELINE II = 1
        cvh[i % NDEC][i / NDEC] = prm[P_CVH + i];
    }
    int c = 0, ih = 0, rr = 0, iv = 0;
pixels:
    for (int i = 0; i < NCOLS * MX * BAND * MY; i++) {
#pragma HLS PIPELINE II = 1
        const blk_t y = Y[c * SUBROWS + HALO + rr];
        const int n = ih * MY + iv;
        float acc = 0.0f;
        for (int k = 0; k < NDEC; k++) {
            acc += cvh[k][n] * y.v[k];
        }
        pixels.write(acc);
        if (iv == MY - 1) {
            iv = 0;
            if (rr == BAND - 1) {
                rr = 0;
                if (ih == MX - 1) {
                    ih = 0;
                    c++;
                } else {
                    ih++;
                }
            } else {
                rr++;
            }
        } else {
            iv++;
        }
    }
}
void idctStore(hls::stream<float> &pixels, int r0, float *dst)
{
columns:
    for (int c = 0; c < NCOLS; c++) {
    store:
        for (int ih = 0; ih < MX; ih++) {
            for (int i = 0; i < BAND * MY; i++) {
#pragma HLS PIPELINE II = 1
                dst[(c * MX + ih) * SZY + r0 * MY + i] = pixels.read();
            }
        }
    }
}
void blockIdct(const blk_t Y[NBLK], const float prm[NPRM], int r0, float *dst)
{
#pragma HLS DATAFLOW
    hls::stream<float, 2 * MX * BAND * MY> pixels;     // two columns
    idctTransform(Y, prm, pixels);
    idctStore(pixels, r0, dst);
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

// Gathers the state of every group of the band for the estimator of
// stage s (one row of NB lanes per cycle): the coefficients of the
// neighbors times 2^LN_SCALE_BITS, rounded to the signal format (the
// estimator normalizes them, est_df.h); forwards the blocks themselves to
// the rotation
sig_t toState(float v)
{
    const ap_fixed<18, 7 - LN_SCALE_BITS, AP_RND, AP_SAT> t = v;
    sig_t f;
    f.range(17, 0) = t.range(17, 0);
    return f;
}
void gather(const blk_t Y[NBLK], int s, hls::stream<row_t> &features, hls::stream<blk_t> &blocks)
{
    const int ch0 = CH0[s], nCh = NCH[s];
    blk_t nb[NB], nx[NB];   // the blocks of a neighbor, and of the next
#pragma HLS ARRAY_PARTITION variable = nb type = complete
#pragma HLS ARRAY_PARTITION variable = nx type = complete
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
            // The neighbors (iv, ih) with ih fastest, one feature row per
            // cycle over all of them: the blocks of the next neighbor are
            // read one per cycle in the first NB cycles of a neighbor
            for (int b = 0; b < NB; b++) {
                nb[b] = Y[wrap(c - 1, NCOLS) * SUBROWS + wrap(g * NB + b - 1, SUBROWS)];
            }
        neighbors:
            // (iv, ih) of the next neighbor, from (0, 1)
            for (int t = 0, n = 0, j = 0, iv = 0, ih = 1; t < 9 * nCh; t++) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT max = 135
                if (n < 8 && j < NB) {
                    nx[j] = Y[wrap(c - 1 + ih, NCOLS) * SUBROWS + wrap(g * NB + j - 1 + iv, SUBROWS)];
                }
                row_t f;
                for (int b = 0; b < NB; b++) {
                    f.v[b] = toState(nb[b].v[ch0 + j]);
                }
                features.write(f);
                if (j == nCh - 1) {
                    for (int b = 0; b < NB; b++) nb[b] = nx[b];
                    j = 0;
                    n++;
                    if (ih == 2) {
                        ih = 0;
                        iv++;
                    } else {
                        ih++;
                    }
                } else {
                    j++;
                }
            }
        }
    }
}

// Angle k of block b of a group in its word of qmem (band_df.h), with
// its center c
q_t angleOf(const qword_t &w, int k, int b, qctr_t c)
{
    const int bit = NQB * (NB * ((k + NQPAD) % NQR) + b);
    ap_int<NQB> r;
    r.range(NQB - 1, 0) = w.range(bit + NQB - 1, bit);
    const ap_int<NQA> v = c + r;               // wrapped
    ap_fixed<NQA, 1> a;
    a.range(NQA - 1, 0) = v.range(NQA - 1, 0);
    return a;
}

// The rotations of the blocks of a tile, NLANE columns of the band at a
// time, one block of every column per cycle: of the antisymmetric half
// (U), after those of the symmetric half (W) in stage 0; in the last stage
// only the first PA-1 rotations. The angles are those of the groups as in
// qmem (per column the words of W, then of U), ctr their centers (W, U).
// The same block comes again SUBROWS cycles later (the latency of a
// rotation is below it for the bands of a frame)
void rotateTile(float v[NLANE][SUBROWS][NDEC], const qword_t qa[NLANE][2 * NQ5 * NG],
                const qctr_t ctr[2][NROT], int s, int nc, bool transposed)
{
#pragma HLS INLINE
    const int nRot = (s == NSTAGES) ? NROT_LAST : NROT;
    const int n = ((s == 0) ? 2 : 1) * nRot * SUBROWS;
    int kk = 0, g = 0, b = 0;
    bool h = (s != 0);
rotations:
    for (int i = 0; i < n; i++) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT max = 2 * NROT * SUBROWS
#pragma HLS DEPENDENCE variable = v type = inter direction = RAW dependent = true distance = SUBROWS
        const int k = transposed ? nRot - 1 - kk : kk;   // transposes in reverse order
        const int t = GTOP[k], bt = GBTM[k];
        const qctr_t ck = ctr[h][k];
        const int wk = ((h ? NQ5 : 0) + (k + NQPAD) / NQR) * NG + g;
        for (int l = 0; l < NLANE; l++) {
#pragma HLS UNROLL
            const q_t qk = angleOf(qa[l][wk], k, b, ck);
            givens(v[l][g * NB + b], h, t, bt, cospi(qk), sinpi(qk), transposed, l < nc);
        }
        if (b == NB - 1) {
            b = 0;
            if (g == NG - 1) {
                g = 0;
                if (kk == nRot - 1) {
                    kk = 0;
                    h = true;
                } else {
                    kk++;
                }
            } else {
                g++;
            }
        } else {
            b++;
        }
    }
}

// Rotations of the blocks of the band by the angles of stage s (rotateTile,
// NLANE columns at a time). Then the sign flips (mus), and in the last
// stage the coefficient mask (the outputs but channel PS are zero as the
// mask would make them). Writes the angles to qmem for the synthesis.
void rotate(hls::stream<arow_t> &angles, hls::stream<blk_t> &blocks, const float prm[NPRM], int s,
            blk_t Y[NBLK], qword_t qmem[NQMEM])
{
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
    qctr_t ctr[2][NROT];
    loadCenters(prm, s, ctr[0], ctr[1]);
    qword_t qa[NLANE][2 * NQ5 * NG];      // the angles of the groups of a column, as in qmem
#pragma HLS ARRAY_PARTITION variable = qa type = complete dim = 1
    float v[NLANE][SUBROWS][NDEC];
#pragma HLS ARRAY_PARTITION variable = v type = complete dim = 1
#pragma HLS ARRAY_PARTITION variable = v type = complete dim = 3
    // The blocks are read one per cycle, the angles one row of the NB
    // blocks per cycle, put together NQR rows to a word
tiles:
    for (int c0 = 0; c0 < NCOLS; c0 += NLANE) {
        const int nc = (NCOLS - c0 < NLANE) ? NCOLS - c0 : NLANE;
    read:
        for (int l = 0; l < nc; l++) {
#pragma HLS LOOP_TRIPCOUNT max = NLANE
            for (int g = 0; g < NG; g++) {
                for (int b = 0; b < NB; b++) {
#pragma HLS PIPELINE II = 1
                    const blk_t y = blocks.read();
                    for (int j = 0; j < NDEC; j++) v[l][g * NB + b][j] = y.v[j];
                }
                // The rows go in at the top of a word and down one place per
                // row, so that a word is complete after its last place (the
                // leading zero angles of W: the first word, the first places
                // of the second, as acc starts from zero)
                qa[l][g] = 0;
                ap_uint<NQB * NB * NQR> acc = 0;
                bool toW = (s == 0);
                int i = toW ? (NZ0 + NQPAD) % NQR : NQPAD, w = toW ? (NZ0 + NQPAD) / NQR : 0;
            read_angles:
                for (int a = 0; a < nA; a++) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT max = 49
                    const arow_t z = angles.read();
                    const int k = w * NQR + i - NQPAD;
                    const qctr_t ck = ctr[toW ? 0 : 1][k];
                    ap_uint<NQB * NB> row;
                    for (int b = 0; b < NB; b++) row.range(NQB * b + NQB - 1, NQB * b) = encodeAngle(toAngle(z.v[b]), ck);
                    acc = (acc >> (NQB * NB)) | (ap_uint<NQB * NB * NQR>(row) << (NQB * NB * (NQR - 1)));
                    if (i == NQR - 1) {
                        qa[l][((toW ? 0 : NQ5) + w) * NG + g] = acc;
                        acc = 0;
                    }
                    if (i == NQR - 1 && toW && w == NQ5 - 1) {   // W done: U from its first angle
                        toW = false;
                        i = NQPAD;
                        w = 0;
                    } else if (i == NQR - 1) {
                        i = 0;
                        w++;
                    } else {
                        i++;
                    }
                }
            }
        }
        rotateTile(v, qa, ctr, s, nc, false);
    write:
        for (int l = 0; l < nc; l++) {
#pragma HLS LOOP_TRIPCOUNT max = NLANE
            for (int r = 0; r < SUBROWS; r++) {
#pragma HLS PIPELINE II = 1
                blk_t o;
                for (int j = 0; j < PA; j++) {
                    o.v[j] = (s == 0) ? flip(negW[j], v[l][r][j]) : v[l][r][j];
                    if (s == NSTAGES && j > 0) {
                        o.v[PS + j] = 0.0f;
                    } else {
                        o.v[PS + j] = flip(negU[j], v[l][r][PS + j]);
                    }
                }
                for (int j = 0; j < NDEC; j++) {
                    if (drop[j]) o.v[j] = 0.0f;
                }
                Y[(c0 + l) * SUBROWS + r] = o;
            }
        }
        // the words of the groups of a column, one per cycle
        const int nw = qgWords(s), w0 = (s == 0) ? 0 : NQ5;
    store_columns:
        for (int l = 0; l < nc; l++) {
#pragma HLS LOOP_TRIPCOUNT max = NLANE
            const int qbase = qgAddr(s, (c0 + l) * NG);
            int g = 0, w = 0;
        store_angles:
            for (int i = 0; i < NG * nw; i++) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT max = NG * 2 * NQ5
                qmem[qbase + i] = qa[l][(w0 + w) * NG + g];
                if (w == nw - 1) {
                    w = 0;
                    g++;
                } else {
                    w++;
                }
            }
        }
    }
}

// One analysis stage: features -> estimator -> rotations, as a dataflow
void analysisStage(const blk_t Yin[NBLK], blk_t Yout[NBLK], const float prm[NPRM],
                   const wraw_t *wmem, const braw_t *bmem, const float *lnmem,
                   qword_t qmem[NQMEM], int s, bool load)
{
#pragma HLS DATAFLOW
    hls::stream<row_t, FIFO> features;
    // a column of angles, so that the estimator goes on while the
    // rotation works on the previous column; the blocks one column more
    hls::stream<arow_t, NG * NAMAX> angles;
    hls::stream<blk_t, (2 * NG + 24) * NB> blocks;  // the groups the estimator holds, and two columns
    gather(Yin, s, features, blocks);
    est_engine(features, angles, wmem, bmem, lnmem, s, NCOLS * NG, load);
    rotate(angles, blocks, prm, s, Yout, qmem);
}

//---------------------------------------------------------------------------
// Synthesis: the transposed rotations of stage s (s = 0: the final
// rotation, W and U), after the sign flips, from the angles in qmem
// (rotateTile, NLANE columns at a time)
void synthesisRotation(const blk_t Yin[NBLK], blk_t Yout[NBLK], const float prm[NPRM], int s,
                       const qword_t qmem[NQMEM])
{
    bool negW[PS], negU[PA];
#pragma HLS ARRAY_PARTITION variable = negW type = complete
#pragma HLS ARRAY_PARTITION variable = negU type = complete
    for (int j = 0; j < PA; j++) {
#pragma HLS PIPELINE II = 1
        negW[j] = prm[P_V0TW + j] < 0;
        negU[j] = ((s == 0) ? prm[P_V0TU + j] : prm[P_SYNMUS + (s - 1) * PA + j]) < 0;
    }
    qctr_t ctr[2][NROT];
    loadCenters(prm, s, ctr[0], ctr[1]);
    qword_t qa[NLANE][2 * NQ5 * NG];      // the angles of the groups of a column, as in qmem
#pragma HLS ARRAY_PARTITION variable = qa type = complete dim = 1
    float v[NLANE][SUBROWS][NDEC];
#pragma HLS ARRAY_PARTITION variable = v type = complete dim = 1
#pragma HLS ARRAY_PARTITION variable = v type = complete dim = 3
tiles:
    for (int c0 = 0; c0 < NCOLS; c0 += NLANE) {
        const int nc = (NCOLS - c0 < NLANE) ? NCOLS - c0 : NLANE;
    read:
        for (int l = 0; l < nc; l++) {
#pragma HLS LOOP_TRIPCOUNT max = NLANE
            for (int r = 0; r < SUBROWS; r++) {
#pragma HLS PIPELINE II = 1
                const blk_t y = Yin[(c0 + l) * SUBROWS + r];
                for (int j = 0; j < PS; j++) {
                    v[l][r][j] = (s == 0) ? flip(negW[j], y.v[j]) : y.v[j];
                    v[l][r][PS + j] = flip(negU[j], y.v[PS + j]);
                }
            }
        }
        // the words of the groups of a column, one per cycle
        const int nw = qgWords(s), w0 = (s == 0) ? 0 : NQ5;
    read_columns:
        for (int l = 0; l < nc; l++) {
#pragma HLS LOOP_TRIPCOUNT max = NLANE
            const int qbase = qgAddr(s, (c0 + l) * NG);
            int g = 0, w = 0;
        read_angles:
            for (int i = 0; i < NG * nw; i++) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT max = NG * 2 * NQ5
                qa[l][(w0 + w) * NG + g] = qmem[qbase + i];
                if (w == nw - 1) {
                    w = 0;
                    g++;
                } else {
                    w++;
                }
            }
        }
        rotateTile(v, qa, ctr, s, nc, true);
    write:
        for (int l = 0; l < nc; l++) {
#pragma HLS LOOP_TRIPCOUNT max = NLANE
            for (int r = 0; r < SUBROWS; r++) {
#pragma HLS PIPELINE II = 1
                blk_t o;
                for (int j = 0; j < NDEC; j++) o.v[j] = v[l][r][j];
                Yout[(c0 + l) * SUBROWS + r] = o;
            }
        }
    }
}

} // namespace bdf

//---------------------------------------------------------------------------
extern "C" void salsun2d_band_df(hls::burst_maxi<float> src, const float *prm, const estdf::wraw_t *wmem,
                                 const estdf::braw_t *bmem, const float *lnmem,
                                 const float *stats_in, float *dst, float *stats_out,
                                 int first_band, int num_bands)
{
    // src on a bundle of its own: with dst (writes widened to 128 bits) the
    // port was 128 bits wide, the float reads one per beat in bursts of 16
    // with one burst in flight (about 13 cycles per pixel). The buffer holds
    // the runs of two columns (requested ahead, blockDct)
#pragma HLS INTERFACE m_axi port = src offset = slave bundle = gmem5 depth = SZY * SZX max_read_burst_length = 64 num_read_outstanding = 64
#pragma HLS INTERFACE m_axi port = dst offset = slave bundle = gmem0 depth = SZY * SZX
#pragma HLS INTERFACE m_axi port = prm offset = slave bundle = gmem1 depth = bdf::NPRM
#pragma HLS INTERFACE m_axi port = stats_in offset = slave bundle = gmem1 depth = 2 * bdf::NSTATS
#pragma HLS INTERFACE m_axi port = stats_out offset = slave bundle = gmem1 depth = 2 * bdf::NSTATS
#pragma HLS INTERFACE m_axi port = wmem offset = slave bundle = gmem2 depth = estdf::NWORDS
#pragma HLS INTERFACE m_axi port = bmem offset = slave bundle = gmem3 depth = estdf::NBIAS
#pragma HLS INTERFACE m_axi port = lnmem offset = slave bundle = gmem3 depth = 2 * estdf::NLN
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
    // the angles of the band for the synthesis (URAM), written by the analysis
    static qword_t qmem[NQMEM];
#pragma HLS BIND_STORAGE variable = qmem type = ram_t2p impl = uram latency = 3
    // the small parameters, read once (the tasks of a dataflow region
    // cannot share their memory interface)
    float prmL[NPRM];
    for (int i = 0; i < NPRM; i++) {
#pragma HLS PIPELINE II = 1
        prmL[i] = prm[i];
    }
    bool load = true;                          // the estimator parameters, once per call
bands:
    for (int k = 0; k < NBANDS; k++) {
        int r0 = k * BAND;
        if (r0 + BAND > NROWS) r0 = NROWS - BAND;          // move the last band up
        if (k < first_band || k >= first_band + num_bands) continue;

        blockDct(src, prmL, r0, A);
        for (int s = 0; s <= NSTAGES; s++) {
            if (s > 0) atomExtension(B, A, SHIFT[s - 1][0], SHIFT[s - 1][1], TARGET[s - 1]);
            analysisStage(A, B, prmL, wmem, bmem, lnmem, qmem, s, load);
            load = false;
        }
        for (int s = NSTAGES; s >= 1; s--) {
            synthesisRotation(B, A, prmL, s, qmem);
            atomExtension(A, B, SYNSHIFT[s - 1][0], SYNSHIFT[s - 1][1], SYNTARGET[s - 1]);
        }
        synthesisRotation(B, A, prmL, 0, qmem);
        blockIdct(A, prmL, r0, dst);
    }
    // the estimators take no statistics of the frame: stats_in is not read,
    // and stats_out is cleared (ports kept for the host programs)
    for (int i = 0; i < 2 * NSTATS; i++) {
#pragma HLS PIPELINE II = 1
        stats_out[i] = 0.0f;
    }
}
