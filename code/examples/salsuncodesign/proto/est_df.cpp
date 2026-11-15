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

#include "gelu_table.h"

namespace estdf {

// Fully connected layer of estimator k, output stationary over the NB
// lanes: for each group of P inputs, every cycle adds P products to OPAR
// outputs (1 or 2: two reads of the weights per cycle, the two ports of the
// URAM) of all NB blocks. The inputs of the next group of P are read
// during the first P cycles of a pass over the outputs, those of the first
// group of P of the next group of blocks during the last pass, so the
// groups of inputs and of blocks are one pipeline. ID makes each call site
// its own function, with its own static parameter memory (kept between
// calls). NWMEM words of weights for all estimators (offsets woff, rows of
// words(nIn) words, in URAM), biases boff. With OPAR = 1 an output row is
// sent as soon as it is complete, during the last pass over the inputs;
// with OPAR = 2 the rows of a group of blocks are kept and sent one per
// cycle during the next group of blocks (and after the last). A pass has
// ceil(nOut / OPAR) cycles: at least P, and (OPAR = 2) ngi >= 2.
template <int ID, int NOUTMAX, int NWMEM, int NBMEM, int OPAR>
void fc(hls::stream<row_t> &in, hls::stream<arow_t> &out, hls::stream<wword_t> &wload,
        hls::stream<acc_t> &bload, int nIn, int ngi, int nOut, int woff, int boff, int ngroups,
        bool load)
{
    constexpr int NJ = (NOUTMAX + OPAR - 1) / OPAR;
    static wword_t W[NWMEM];            // weights of this layer, on chip (kept between calls)
    static acc_t B[NBMEM];
#pragma HLS AGGREGATE variable = W
#pragma HLS BIND_STORAGE variable = W type = ram_t2p impl = uram
#pragma HLS BIND_STORAGE variable = B type = ram_t2p
    if (load) {
    load_w:
        for (int i = 0; i < NWMEM; i++) {
#pragma HLS PIPELINE II = 1
            W[i] = wload.read();
        }
    load_b:
        for (int i = 0; i < NBMEM; i++) {
#pragma HLS PIPELINE II = 1
            B[i] = bload.read();
        }
    }
    arow_t Z[OPAR][NJ];                 // partial sums: output OPAR j + q in Z[q][j]
    arow_t Za[OPAR][NJ], Zb[OPAR][NJ];  // finished rows (OPAR = 2), groups of blocks in turn
#pragma HLS AGGREGATE variable = Z
#pragma HLS AGGREGATE variable = Za
#pragma HLS AGGREGATE variable = Zb
#pragma HLS ARRAY_PARTITION variable = Z type = complete dim = 1
#pragma HLS ARRAY_PARTITION variable = Za type = complete dim = 1
#pragma HLS ARRAY_PARTITION variable = Zb type = complete dim = 1
    sig_t xr[P][NB], xn[P][NB];
#pragma HLS ARRAY_PARTITION variable = xr type = complete dim = 0
#pragma HLS ARRAY_PARTITION variable = xn type = complete dim = 0
    const int npass = (nOut + OPAR - 1) / OPAR;
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
    int ndrain = 0, kd = 0;             // rows of the previous group of blocks left to send
group:
    for (int g = 0; g < ngroups; g++) {
        const bool odd = g & 1;
        const bool more = g + 1 < ngroups;
        int gi = 0, j = 0;
        int inext = P;                      // next input of this group to read
        int wrow = woff;                    // first word of the row of output OPAR j
    mac:
        for (int t = 0; t < ngi * npass; t++) {
#pragma HLS PIPELINE II = 1
#pragma HLS DEPENDENCE variable = Z type = inter false
#pragma HLS DEPENDENCE variable = Za type = inter false
#pragma HLS DEPENDENCE variable = Zb type = inter false
#pragma HLS LOOP_TRIPCOUNT max = 6 * 68
            const bool last = gi == ngi - 1;
            if (j < P && (!last || more)) {     // inputs of the next group of P
                const int i = last ? j : inext;
                row_t r;
                if (i < nIn) {
                    r = in.read();
                } else {
                    for (int b = 0; b < NB; b++) r.v[b] = 0;
                }
                for (int b = 0; b < NB; b++) xn[j][b] = r.v[b];
                inext++;
            }
            if (OPAR > 1 && ndrain > 0) {       // a finished row of the previous group of blocks
                const int qd = kd % OPAR, jd = kd / OPAR;
                out.write(odd ? Za[qd][jd] : Zb[qd][jd]);
                kd++;
                ndrain--;
            }
            for (int q = 0; q < OPAR; q++) {
                const int o = OPAR * j + q;
                const bool valid = o < nOut;
                const wword_t wv = W[valid ? wrow + q * ngi + gi : woff];
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
                    zn.v[b] = s + ((gi == 0) ? B[boff + (valid ? o : 0)] : zo.v[b]);
                }
                Z[q][j] = zn;
                if (last && valid) {
                    if (OPAR == 1) {
                        out.write(zn);
                    } else if (odd) {
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
                gi++;
                wrow = woff;
            } else {
                j++;
                wrow += OPAR * ngi;
            }
        }
        if (OPAR > 1) {
            ndrain = nOut;
            kd = 0;
        }
    }
    const bool odd = ngroups & 1;           // the buffer of the last group of blocks
drain:
    for (; ndrain > 0; ndrain--) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT max = NOUTMAX
        const int qd = kd % OPAR, jd = kd / OPAR;
        out.write(odd ? Za[qd][jd] : Zb[qd][jd]);
        kd++;
    }
}

// LayerNorm over the nF features of each block, as three tasks so that the
// groups of blocks overlap: the exact sums in fixed point (ln_sums, which
// keeps the rows for the normalization and sends them on to the residual
// add), the mean and the reciprocal standard deviation of each block in
// single precision (ln_scale), the normalization in fixed point (ln_norm;
// est_df.h)
struct lnsum_t {
    acc_t s1[NB];
    sq_t s2[NB];
};
struct lnms_t {
    lnm_t mean[NB];
    lnr_t rs[NB];
};
void ln_sums(hls::stream<row_t> &in, hls::stream<row_t> &keep, hls::stream<row_t> &skip,
             hls::stream<lnsum_t> &sums, int nF, int ngroups)
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
            skip.write(r);
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
            m.rs[b] = 1.0f / hls::sqrt(v + 1e-5f);
        }
        ms.write(m);
    }
}
template <int ID>
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

// GELU of an FC output: the table (gelu_table.h) interpolated linearly,
// rounded to the signal format
sig_t gelu1(acc_t z)
{
    if (z >= 8) return (sig_t)z;
    if (z <= -8) return 0;
    const ap_ufixed<40, 11> u = (ap_ufixed<40, 11>)(z + 8) * 64;    // in [0, 1024)
    const ap_uint<10> i = u.range(38, 29);                           // integer part
    const ap_ufixed<16, 0> fr = (ap_ufixed<16, 0>)u;                 // fraction, truncated
    const gtab_t t0 = GELU_TABLE[i], t1 = GELU_TABLE[i + 1];
    const ap_fixed<48, 8> a = t0 + (ap_fixed<26, 6>)(t1 - t0) * fr;
    return (sig_t)a;
}

// GELU of the rows, NB lanes (one row per cycle)
void gelu(hls::stream<arow_t> &in, hls::stream<row_t> &out, int nH, int ngroups)
{
group:
    for (int g = 0; g < ngroups; g++) {
    rows:
        for (int i = 0; i < nH; i++) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT max = 135
            const arow_t z = in.read();
            row_t a;
            for (int b = 0; b < NB; b++) a.v[b] = gelu1(z.v[b]);
            out.write(a);
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
wword_t toWord(wraw_t r)
{
    wword_t w;
    for (int p = 0; p < P; p++) w.w[p].range(17, 0) = r.range(18 * p + 17, 18 * p);
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
            hls::stream<lnpar_t> lns[NRES], bool load)
{
    if (!load) return;
    int kw = 0, kb = 0;
    for (int r = 0; r < NRES; r++) {
        for (int i = 0; i < SQWORDS; i++) {
#pragma HLS PIPELINE II = 1
            w1s[r].write(toWord(wmem[kw++]));
        }
        for (int i = 0; i < SQWORDS; i++) {
#pragma HLS PIPELINE II = 1
            w2s[r].write(toWord(wmem[kw++]));
        }
        for (int i = 0; i < NVEC; i++) {
#pragma HLS PIPELINE II = 1
            b1s[r].write(toAcc(bmem[kb++]));
        }
        for (int i = 0; i < NVEC; i++) {
#pragma HLS PIPELINE II = 1
            b2s[r].write(toAcc(bmem[kb++]));
        }
        for (int i = 0; i < NVEC; i++) {
#pragma HLS PIPELINE II = 1
            const int j = r * NVEC + i;
            lns[r].write(lnpar_t{lnmem[2 * j], lnmem[2 * j + 1]});
        }
    }
    for (int i = 0; i < OUTWORDS; i++) {
#pragma HLS PIPELINE II = 1
        wos.write(toWord(wmem[kw++]));
    }
    for (int i = 0; i < NAALL; i++) {
#pragma HLS PIPELINE II = 1
        bos.write(toAcc(bmem[kb++]));
    }
}

// Sizes and parameter offsets of estimator k (a small table, no division)
struct est_dims_t {
    int nF, ngi, nA, sq, out, vec, a;
};
est_dims_t dims(int k)
{
    // constant tables of the sizes and offsets (estimators 1..4 alike)
    constexpr int SQ1 = sqOffset(1), OUT1 = outOffset(1), VEC1 = vecOffset(1), A1 = aOffset(1);
    constexpr int SQW = sqWords(1), OUTW = outWords(1), NF1 = NFK[1], NA1 = NAK[1];
    est_dims_t d;
    if (k == 0) {
        d.nF = NFK[0]; d.ngi = words(NFK[0]); d.nA = NAK[0]; d.sq = 0; d.out = 0; d.vec = 0; d.a = 0;
    } else {
        d.nF = NF1; d.ngi = words(NF1); d.nA = NA1;
        d.sq = SQ1 + (k - 1) * SQW;
        d.out = OUT1 + (k - 1) * OUTW;
        d.vec = VEC1 + (k - 1) * NF1;
        d.a = A1 + (k - 1) * NA1;
    }
    return d;
}

// One residual block: LayerNorm, FC, GELU, FC, skip
#define RESBLOCK(R, FIN, FOUT)                                                                  \
    ln_sums(FIN, keep##R, skip##R, sums##R, d.nF, ngroups);                                   \
    ln_scale(sums##R, ms##R, d.nF, ngroups);                                                  \
    ln_norm<R>(keep##R, ms##R, ln##R, lns[R], d.nF, d.vec, ngroups, load);                    \
    fc<2 * R, NFMAX, SQWORDS, NVEC, 2>(ln##R, z1##R, w1s[R], b1s[R], d.nF, d.ngi, d.nF, d.sq, d.vec, ngroups, load); \
    gelu(z1##R, a##R, d.nF, ngroups);                                                         \
    fc<2 * R + 1, NFMAX, SQWORDS, NVEC, 2>(a##R, z2##R, w2s[R], b2s[R], d.nF, d.ngi, d.nF, d.sq, d.vec, ngroups, load); \
    residual_add(z2##R, skip##R, FOUT, d.nF, ngroups);

// Estimator k on ngroups groups of NB blocks: feature rows in, angle rows
// out (estdf::NAK[k] per group). With load, the parameters of all
// estimators are first read from memory into the tasks, which keep them
// for later calls.
void est_engine(hls::stream<row_t> &features, hls::stream<arow_t> &angles, const wraw_t *wmem,
                const braw_t *bmem, const float *lnmem, int k, int ngroups, bool load)
{
#pragma HLS DATAFLOW
    const est_dims_t d = dims(k);
    hls::stream<wword_t, 4> w1s[NRES], w2s[NRES], wos;
    hls::stream<acc_t, 4> b1s[NRES], b2s[NRES], bos;
    hls::stream<lnpar_t, 4> lns[NRES];
    hls::stream<row_t, FIFO> ln0, ln1, ln2, a0, a1, a2, f1, f2, f3;
    // groups wait for their two FC layers: about three groups (135 rows) in
    // stage 0, more than 512 rows at times (the same URAM up to 4096)
    hls::stream<row_t, 1024> skip0, skip1, skip2;
#pragma HLS BIND_STORAGE variable = skip0 type = fifo impl = uram
#pragma HLS BIND_STORAGE variable = skip1 type = fifo impl = uram
#pragma HLS BIND_STORAGE variable = skip2 type = fifo impl = uram
    hls::stream<arow_t, FIFO> z10, z11, z12, z20, z21, z22;
    // the rows of a group wait in keep for its statistics (with those of the
    // next group arriving meanwhile)
    hls::stream<row_t, 256> keep0, keep1, keep2;
    hls::stream<lnsum_t, 2> sums0, sums1, sums2;
    hls::stream<lnms_t, 2> ms0, ms1, ms2;
    loader(wmem, bmem, lnmem, w1s, w2s, wos, b1s, b2s, bos, lns, load);
    RESBLOCK(0, features, f1)
    RESBLOCK(1, f1, f2)
    RESBLOCK(2, f2, f3)
    fc<2 * NRES, NAMAX, OUTWORDS, NAALL, 1>(f3, angles, wos, bos, d.nF, d.ngi, d.nA, d.out, d.a, ngroups, load);
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
