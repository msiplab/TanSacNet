// est_df.cpp: see est_df.h
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
#include "est_df.h"

#include <hls_math.h>

#include "act_table.h"

namespace estdf {

// Fully connected layer of estimator k, output stationary over the NB
// lanes: for each group of P inputs, every cycle adds P products to OPAR
// outputs of all NB blocks. The weights and biases are in OPAR banks
// (output o in bank o % OPAR, see est_df.h), one read of each per cycle.
// The inputs of the next group of P are read during the first P cycles of
// a pass over the outputs, those of the first group of P of the next
// group of blocks during the last pass, so the groups of inputs and of
// blocks are one pipeline. The rows of a group of blocks are kept and sent
// one per cycle during the next group of blocks (and after the last). A
// pass has ceil(nOut / OPAR) cycles: at least P, and ngi passes at least
// nOut cycles. ID makes each call site its own function, with its own
// static parameter memory (kept between calls); KIND: the layer (FC1, FC2
// or FCO, which gives its sizes, est_df.h). woff and boff are the offsets
// of estimator k in the banks of weights (words) and of biases (rows).
template <int ID, int KIND, int OPAR>
void fc(hls::stream<row_t> &in, hls::stream<arow_t> &out, hls::stream<wword_t> &wload,
        hls::stream<acc_t> &bload, int nIn, int ngi, int nOut, int woff, int boff, int ngroups,
        bool load)
{
    constexpr int NOUTMAX = nOutK(KIND, 0);
    constexpr int NJ = passLen(NOUTMAX, OPAR);
    constexpr int NWQ = bankWordOffset(KIND, OPAR, NEST);
    constexpr int NBQ = bankRowOffset(KIND, OPAR, NEST);
    static wword_t W[OPAR][NWQ];        // weights of this layer, on chip (kept between calls)
    static acc_t B[OPAR][NBQ];
#pragma HLS AGGREGATE variable = W
#pragma HLS ARRAY_PARTITION variable = W type = complete dim = 1
#pragma HLS ARRAY_PARTITION variable = B type = complete dim = 1
#pragma HLS BIND_STORAGE variable = W type = ram_s2p impl = uram
#pragma HLS BIND_STORAGE variable = B type = ram_s2p
    if (load) {                         // rows in order, the words of a row in order
    load_w:
        for (int k = 0, base = 0; k < NEST; k++) {
            const int no = nOutK(KIND, k), ng = words(nInK(KIND, k));
            int q = 0, r = base, g = 0;
        load_w_k:
            for (int t = 0; t < no * ng; t++) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT max = NFMAX * words(NFMAX)
                W[q][r + g] = wload.read();
                if (g == ng - 1) {
                    g = 0;
                    if (q == OPAR - 1) {
                        q = 0;
                        r += ng;
                    } else {
                        q++;
                    }
                } else {
                    g++;
                }
            }
            base += (no + OPAR - 1) / OPAR * ng;
        }
    load_b:
        for (int k = 0, base = 0; k < NEST; k++) {
            const int no = nOutK(KIND, k);
            int q = 0, r = base;
        load_b_k:
            for (int o = 0; o < no; o++) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT max = NFMAX
                B[q][r] = bload.read();
                if (q == OPAR - 1) {
                    q = 0;
                    r++;
                } else {
                    q++;
                }
            }
            base += (no + OPAR - 1) / OPAR;
        }
    }
    arow_t Z[OPAR][NJ];                 // partial sums: output OPAR j + q in Z[q][j]
    arow_t Za[OPAR][NJ], Zb[OPAR][NJ];  // finished rows, groups of blocks in turn
#pragma HLS AGGREGATE variable = Z
#pragma HLS AGGREGATE variable = Za
#pragma HLS AGGREGATE variable = Zb
#pragma HLS ARRAY_PARTITION variable = Z type = complete dim = 1
#pragma HLS ARRAY_PARTITION variable = Za type = complete dim = 1
#pragma HLS ARRAY_PARTITION variable = Zb type = complete dim = 1
    sig_t xr[P][NB], xn[P][NB];
#pragma HLS ARRAY_PARTITION variable = xr type = complete dim = 0
#pragma HLS ARRAY_PARTITION variable = xn type = complete dim = 0
    const int npass = passLen(nOut, OPAR);  // cycles of a pass (idle beyond the outputs)
first:
    for (int p = 0; p < P; p++) {       // the first inputs of the first group of blocks
#pragma HLS PIPELINE II = 1
        row_t r;
        if (p < nIn && ngroups > 0) {
            r = in.read();
        } else {
            for (int b = 0; b < NB; b++) r.v[b] = 0;
        }
        for (int b = 0; b < NB; b++) xr[p][b] = r.v[b];
    }
    int ndrain = 0, qd = 0, jd = 0;     // rows of the previous group of blocks left to send
group:
    for (int g = 0; g < ngroups; g++) {
        const bool odd = g & 1;
        const bool more = g + 1 < ngroups;
        int gi = 0, j = 0;
        int inext = P;                      // next input of this group to read
        int wrow = woff;                    // first word of row j of the banks
        // the conditions of the reading of an input, kept as flags (a short
        // path to the stream): the last pass, an input left, j < P
        bool lastf = ngi == 1, inmore = P < nIn, jlow = true;
    mac:
        for (int t = 0; t < ngi * npass; t++) {
#pragma HLS PIPELINE II = 1
#pragma HLS DEPENDENCE variable = Z type = inter false
#pragma HLS DEPENDENCE variable = Za type = inter false
#pragma HLS DEPENDENCE variable = Zb type = inter false
#pragma HLS LOOP_TRIPCOUNT max = words(nInK(KIND, 0)) * NJ
            const bool last = lastf;
            if (jlow && (!last || more)) {      // inputs of the next group of P
                row_t r;
                if (last || inmore) {           // (P <= nIn: all of the first P are read)
                    r = in.read();
                } else {
                    for (int b = 0; b < NB; b++) r.v[b] = 0;
                }
                for (int b = 0; b < NB; b++) xn[j][b] = r.v[b];
                inext++;
                inmore = inext < nIn;
            }
            if (ndrain > 0) {                   // a finished row of the previous group of blocks
                out.write(odd ? Za[qd][jd] : Zb[qd][jd]);
                if (qd == OPAR - 1) {
                    qd = 0;
                    jd++;
                } else {
                    qd++;
                }
                ndrain--;
            }
            for (int q = 0; q < OPAR; q++) {
                const bool valid = OPAR * j + q < nOut;
                const wword_t wv = W[q][valid ? wrow + gi : 0];
                const arow_t zo = Z[q][j];
                arow_t zn;
                for (int b = 0; b < NB; b++) {
                    // the products as one chain of multiply-adds (DSP cascade,
                    // no adder tree in the fabric); exact, so the order is free
#pragma HLS EXPRESSION_BALANCE off
                    acc_t s = wv.w[0] * xr[0][b];
                    for (int p = 1; p < P; p++) {
                        s += wv.w[p] * xr[p][b];
                    }
                    zn.v[b] = s + ((gi == 0) ? B[q][valid ? boff + j : 0] : zo.v[b]);
                }
                Z[q][j] = zn;
                if (last && valid) {
                    if (odd) {
                        Zb[q][j] = zn;
                    } else {
                        Za[q][j] = zn;
                    }
                }
            }
            if (j == npass - 1) {
                for (int p = 0; p < P; p++)
                    for (int b = 0; b < NB; b++) xr[p][b] = xn[p][b];
                j = 0;
                jlow = true;
                lastf = gi + 2 == ngi;
                gi++;
                wrow = woff;
            } else {
                j++;
                jlow = j < P;
                wrow += ngi;
            }
        }
        ndrain = nOut;
        qd = 0;
        jd = 0;
    }
    const bool odd = ngroups & 1;           // the buffer of the last group of blocks
drain:
    for (; ndrain > 0; ndrain--) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT max = NOUTMAX
        out.write(odd ? Za[qd][jd] : Zb[qd][jd]);
        if (qd == OPAR - 1) {
            qd = 0;
            jd++;
        } else {
            qd++;
        }
    }
}

// LayerNorm over the nF features of each block, as three tasks so that the
// groups of blocks overlap: the exact sums in fixed point (ln_sums, which
// keeps the rows for the normalization), the mean and the reciprocal
// standard deviation of each block in single precision (ln_scale), the
// normalization in fixed point (ln_norm; est_df.h)
struct lnsum_t {
    acc_t s1[NB];
    sq_t s2[NB];
};
struct lnms_t {
    lnm_t mean[NB];
    lnr_t rs[NB];
};
void ln_sums(hls::stream<row_t> &in, hls::stream<row_t> &keep, hls::stream<lnsum_t> &sums, int nF,
             int ngroups)
{
group:
    for (int g = 0; g < ngroups; g++) {
        lnsum_t s;
        for (int b = 0; b < NB; b++) {
#pragma HLS UNROLL
            s.s1[b] = 0;
            s.s2[b] = 0;
        }
    sums:
        for (int i = 0; i < nF; i++) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT max = 135
            row_t r = in.read();
            keep.write(r);
            for (int b = 0; b < NB; b++) {
                s.s1[b] += r.v[b];
                s.s2[b] += r.v[b] * r.v[b];
            }
        }
        sums.write(s);
    }
}
void ln_scale(hls::stream<lnsum_t> &sums, hls::stream<lnms_t> &ms, int nF, int ngroups)
{
group:
    for (int g = 0; g < ngroups; g++) {
        const lnsum_t s = sums.read();
        lnms_t m;
    scale:
        for (int b = 0; b < NB; b++) {
#pragma HLS PIPELINE II = 1
            const float mf = (float)s.s1[b] / nF;
            float v = (float)s.s2[b] / nF - mf * mf;
            v = v > 0.0f ? v : 0.0f;
            m.mean[b] = mf;
            m.rs[b] = 1.0f / hls::sqrt(v + LN_EPS);
        }
        ms.write(m);
    }
}
void ln_norm(hls::stream<row_t> &keep, hls::stream<lnms_t> &ms, hls::stream<row_t> &out,
             hls::stream<lnpar_t> &lnload, int nF, int poff, int ngroups, bool load)
{
    static lnp_t gam[NVEC], bet[NVEC];  // kept between calls
    if (load) {
    load:
        for (int i = 0; i < NVEC; i++) {
#pragma HLS PIPELINE II = 1
            const lnpar_t q = lnload.read();
            gam[i] = q.gamma;
            bet[i] = q.beta;
        }
    }
group:
    for (int g = 0; g < ngroups; g++) {
        const lnms_t m = ms.read();
    normalize:
        for (int i = 0; i < nF; i++) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT max = 135
            const lnp_t gg = gam[poff + i], be = bet[poff + i];
            const row_t x = keep.read();
            row_t r;
            for (int b = 0; b < NB; b++) {
                // d * rs is wider than a DSP (27 x 18), and built by Vivado
                // from two cascaded DSPs it was wrong on the card (bits from
                // 2^17 LSBs up counted twice), though not in simulation; in
                // LUTs it was right but left the 3-unit design unroutable.
                // So take the integer and fraction parts of rs apart: each
                // product fits one DSP, and as the split is at 16 bits, not
                // the cascade's 17, they cannot be merged into that again.
                const ap_fixed<25, 9> d = x.v[b] - m.mean[b];
                const ap_uint<10> rh = m.rs[b].range(25, 16);
                ap_ufixed<16, 0> rl;
                rl.range(15, 0) = m.rs[b].range(15, 0);
                const ap_fixed<35, 19> ph = d * rh;
                const ap_fixed<41, 9> pl = d * rl;
                const ap_fixed<51, 19> pr = ph + pl;
                const lnt_t t = pr;
                r.v[b] = t * gg + be;
            }
            out.write(r);
        }
    }
}

// Value at z of a table of ACT_N points at x = -8 + i/64, interpolated
// linearly with a 16-bit fraction (the end values beyond -8 and 8)
template <typename T>
ap_fixed<48, 8> interpolate(const T *table, acc_t z)
{
    if (z >= 8) return table[ACT_N - 1];
    if (z <= -8) return table[0];
    const ap_ufixed<40, 11> u = (ap_ufixed<40, 11>)(z + 8) * 64;    // in [0, 1024)
    const ap_uint<10> i = u.range(38, 29);                           // integer part
    const ap_ufixed<16, 0> fr = (ap_ufixed<16, 0>)u;                 // fraction, truncated
    const T t0 = table[i], t1 = table[i + 1];
    const ap_fixed<48, 8> a = t0 + (ap_fixed<26, 4>)(t1 - t0) * fr;
    return a;
}

// tanh of the rows of an FC layer, NB lanes (one row per cycle), rounded
// to the signal format
void tanh_rows(hls::stream<arow_t> &in, hls::stream<row_t> &out, int nH, int ngroups)
{
group:
    for (int g = 0; g < ngroups; g++) {
    rows:
        for (int i = 0; i < nH; i++) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT max = 135
            const arow_t z = in.read();
            row_t a;
            for (int b = 0; b < NB; b++) a.v[b] = (sig_t)interpolate(TANH_TABLE, z.v[b]);
            out.write(a);
        }
    }
}

// Angles asin(tanh(z)) of the rows z of the output layer (radians,
// truncated to the accumulator format)
void angle_rows(hls::stream<arow_t> &in, hls::stream<arow_t> &out, int nA, int ngroups)
{
group:
    for (int g = 0; g < ngroups; g++) {
    rows:
        for (int i = 0; i < nA; i++) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT max = 49
            const arow_t z = in.read();
            arow_t a;
            for (int b = 0; b < NB; b++) a.v[b] = (acc_t)interpolate(GD_TABLE, z.v[b]);
            out.write(a);
        }
    }
}

// The rows of the input of a residual block, to its first FC layer and to
// its skip connection
void fork_rows(hls::stream<row_t> &in, hls::stream<row_t> &out, hls::stream<row_t> &skip, int nF,
               int ngroups)
{
group:
    for (int g = 0; g < ngroups; g++) {
    rows:
        for (int i = 0; i < nF; i++) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT max = 135
            const row_t r = in.read();
            out.write(r);
            skip.write(r);
        }
    }
}

// Residual add: the block input (skip) plus the second FC layer, rounded
// to the signal format
void residual_add(hls::stream<arow_t> &z, hls::stream<row_t> &skip, hls::stream<row_t> &out, int nF,
                  int ngroups)
{
group:
    for (int g = 0; g < ngroups; g++) {
    rows:
        for (int i = 0; i < nF; i++) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT max = 135
            const arow_t zr = z.read();
            const row_t sr = skip.read();
            row_t r;
            for (int b = 0; b < NB; b++) {
                r.v[b] = (sig_t)(sr.v[b] + zr.v[b]);
            }
            out.write(r);
        }
    }
}

// Distributes the parameters from memory to the tasks that keep them
wword_t toWord(wraw_t r, int h)          // part h of a host word
{
    wword_t w;
    for (int p = 0; p < P; p++) w.w[p].range(17, 0) = r.range(18 * (h * P + p) + 17, 18 * (h * P + p));
    return w;
}
acc_t toAcc(braw_t r)
{
    acc_t a;
    a.range(47, 0) = r.range(47, 0);
    return a;
}

void loader(const wraw_t *wmem, const braw_t *bmem, const float *lnmem,
            hls::stream<wword_t> w1s[NRES], hls::stream<wword_t> w2s[NRES], hls::stream<wword_t> &wos,
            hls::stream<acc_t> b1s[NRES], hls::stream<acc_t> b2s[NRES], hls::stream<acc_t> &bos,
            hls::stream<lnpar_t> &lns, bool load)
{
    if (!load) return;
    int kw = 0, kb = 0;
    for (int r = 0; r < NRES; r++) {
        for (int i = 0; i < W1WORDS; i++) {  // one host word, SPLIT on-chip words
#pragma HLS PIPELINE II = SPLIT
            const wraw_t raw = wmem[kw++];
            for (int h = 0; h < SPLIT; h++) w1s[r].write(toWord(raw, h));
        }
        for (int i = 0; i < W2WORDS; i++) {
#pragma HLS PIPELINE II = SPLIT
            const wraw_t raw = wmem[kw++];
            for (int h = 0; h < SPLIT; h++) w2s[r].write(toWord(raw, h));
        }
        for (int i = 0; i < NHALL; i++) {
#pragma HLS PIPELINE II = 1
            b1s[r].write(toAcc(bmem[kb++]));
        }
        for (int i = 0; i < NVEC; i++) {
#pragma HLS PIPELINE II = 1
            b2s[r].write(toAcc(bmem[kb++]));
        }
    }
    for (int i = 0; i < OUTWORDS; i++) {
#pragma HLS PIPELINE II = SPLIT
        const wraw_t raw = wmem[kw++];
        for (int h = 0; h < SPLIT; h++) wos.write(toWord(raw, h));
    }
    for (int i = 0; i < NAALL; i++) {
#pragma HLS PIPELINE II = 1
        bos.write(toAcc(bmem[kb++]));
    }
    for (int i = 0; i < NVEC; i++) {
#pragma HLS PIPELINE II = 1
        lns.write(lnpar_t{lnmem[2 * i], lnmem[2 * i + 1]});
    }
}

// Sizes and parameter offsets of estimator k (a small table, no division):
// vec is the offset of the LayerNorm parameters (features in order), w1,
// w2, wo and b1, b2, bo those of the layers FC1, FC2 and FCO in their banks; ngf and ngh are
// the groups of P inputs of the features and of the hidden units
struct est_dims_t {
    int nF, nH, nA, ngf, ngh, vec, w1, w2, wo, b1, b2, bo;
};
est_dims_t dims(int k)
{
    // constant tables of the sizes and offsets (estimators 1..4 alike)
    constexpr int VEC1 = vecOffset(1), NF1 = NFK[1], NH1 = NHK[1], NA1 = NAK[1];
    constexpr int W1[3] = {bankWordOffset(FC1, OPAR_R, 1), bankWordOffset(FC2, OPAR_R, 1),
                           bankWordOffset(FCO, OPAR_O, 1)};
    constexpr int WK[3] = {bankWords(FC1, OPAR_R, 1), bankWords(FC2, OPAR_R, 1), bankWords(FCO, OPAR_O, 1)};
    constexpr int B1[3] = {bankRowOffset(FC1, OPAR_R, 1), bankRowOffset(FC2, OPAR_R, 1),
                           bankRowOffset(FCO, OPAR_O, 1)};
    constexpr int BK[3] = {bankRows(FC1, OPAR_R, 1), bankRows(FC2, OPAR_R, 1), bankRows(FCO, OPAR_O, 1)};
    est_dims_t d;
    if (k == 0) {
        d.nF = NFK[0]; d.nH = NHK[0]; d.nA = NAK[0];
        d.ngf = words(NFK[0]); d.ngh = words(NHK[0]);
        d.vec = 0;
        d.w1 = 0; d.w2 = 0; d.wo = 0; d.b1 = 0; d.b2 = 0; d.bo = 0;
    } else {
        d.nF = NF1; d.nH = NH1; d.nA = NA1;
        d.ngf = words(NF1); d.ngh = words(NH1);
        d.vec = VEC1 + (k - 1) * NF1;
        d.w1 = W1[FC1] + (k - 1) * WK[FC1]; d.b1 = B1[FC1] + (k - 1) * BK[FC1];
        d.w2 = W1[FC2] + (k - 1) * WK[FC2]; d.b2 = B1[FC2] + (k - 1) * BK[FC2];
        d.wo = W1[FCO] + (k - 1) * WK[FCO]; d.bo = B1[FCO] + (k - 1) * BK[FCO];
    }
    return d;
}

// One residual block: FC, tanh, FC, skip
#define RESBLOCK(R, FIN, FOUT)                                                                  \
    fork_rows(FIN, x##R, skip##R, d.nF, ngroups);                                             \
    fc<2 * R, FC1, OPAR_R>(x##R, z1##R, w1s[R], b1s[R], d.nF, d.ngf, d.nH, d.w1, d.b1, ngroups, load); \
    tanh_rows(z1##R, a##R, d.nH, ngroups);                                                    \
    fc<2 * R + 1, FC2, OPAR_R>(a##R, z2##R, w2s[R], b2s[R], d.nH, d.ngh, d.nF, d.w2, d.b2, ngroups, load); \
    residual_add(z2##R, skip##R, FOUT, d.nF, ngroups);

// Estimator k on ngroups groups of NB blocks: rows of the state (times
// 2^LN_SCALE_BITS) in, angle rows out (estdf::NAK[k] per group). With
// load, the parameters of all estimators are first read from memory into
// the tasks, which keep them for later calls.
void est_engine(hls::stream<row_t> &features, hls::stream<arow_t> &angles, const wraw_t *wmem,
                const braw_t *bmem, const float *lnmem, int k, int ngroups, bool load)
{
#pragma HLS DATAFLOW
    const est_dims_t d = dims(k);
    hls::stream<wword_t, 4> w1s[NRES], w2s[NRES], wos;
    hls::stream<acc_t, 4> b1s[NRES], b2s[NRES], bos;
    hls::stream<lnpar_t, 4> lns;
    // the rows of a group wait in keep for its statistics (with those of the
    // next group arriving meanwhile)
    hls::stream<row_t, 256> keep;
    hls::stream<lnsum_t, 2> sums;
    hls::stream<lnms_t, 2> ms;
    hls::stream<row_t, FIFO> f0, x0, a0, f1;
    hls::stream<arow_t, FIFO> z10, z20, zo;
    // groups wait for their two FC layers: about three groups (135 rows) in
    // stage 0, more than 512 rows at times (the same URAM up to 4096)
    hls::stream<row_t, 1024> skip0;
#pragma HLS BIND_STORAGE variable = skip0 type = fifo impl = uram
#if EST_NRES > 1
    hls::stream<row_t, FIFO> x1, a1, f2;
    hls::stream<arow_t, FIFO> z11, z21;
    hls::stream<row_t, 1024> skip1;
#pragma HLS BIND_STORAGE variable = skip1 type = fifo impl = uram
#endif
#if EST_NRES > 2
    hls::stream<row_t, FIFO> x2, a2, f3;
    hls::stream<arow_t, FIFO> z12, z22;
    hls::stream<row_t, 1024> skip2;
#pragma HLS BIND_STORAGE variable = skip2 type = fifo impl = uram
#endif
    loader(wmem, bmem, lnmem, w1s, w2s, wos, b1s, b2s, bos, lns, load);
    ln_sums(features, keep, sums, d.nF, ngroups);
    ln_scale(sums, ms, d.nF, ngroups);
    ln_norm(keep, ms, f0, lns, d.nF, d.vec, ngroups, load);
    RESBLOCK(0, f0, f1)
#if EST_NRES == 1
#define EST_LAST f1
#elif EST_NRES == 2
    RESBLOCK(1, f1, f2)
#define EST_LAST f2
#else
    RESBLOCK(1, f1, f2)
    RESBLOCK(2, f2, f3)
#define EST_LAST f3
#endif
    fc<2 * NRES, FCO, OPAR_O>(EST_LAST, zo, wos, bos, d.nF, d.ngf, d.nA, d.wo, d.bo, ngroups, load);
    angle_rows(zo, angles, d.nA, ngroups);
}

} // namespace estdf

// Top function for synthesis and C simulation of the estimator alone:
// loads the parameters of all estimators, then runs ngroups groups of
// blocks through estimator k
void est_df_top(const estdf::wraw_t *wmem, const estdf::braw_t *bmem, const float *lnmem,
                hls::stream<estdf::row_t> &features, hls::stream<estdf::arow_t> &angles, int k,
                int ngroups)
{
#pragma HLS INTERFACE m_axi port = wmem offset = slave bundle = gmem0 depth = estdf::NWORDS
#pragma HLS INTERFACE m_axi port = bmem offset = slave bundle = gmem1 depth = estdf::NBIAS
#pragma HLS INTERFACE m_axi port = lnmem offset = slave bundle = gmem2 depth = 2 * estdf::NLN
    estdf::est_engine(features, angles, wmem, bmem, lnmem, k, ngroups, true);
}
