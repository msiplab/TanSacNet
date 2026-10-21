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
// lanes: for each group of P inputs, every cycle adds P products to one
// output of all NB blocks. The inputs of the next group of P are read
// during the first P cycles of a group, so the groups and outputs are one
// pipeline; the outputs of a group of blocks are sent once all inputs are
// summed. ID makes each call site its own function, with its own static
// parameter memory (kept between calls). NWMEM words of weights for all estimators (offsets woff, rows of
// words(nIn) words, in URAM), biases boff. An output row is sent as soon as
// it is complete, during the pass over the last group of inputs (the
// output stream holds the rows of a group).
template <int ID, int NOUTMAX, int NWMEM, int NBMEM>
void fc(hls::stream<row_t> &in, hls::stream<arow_t> &out, hls::stream<wword_t> &wload,
        hls::stream<acc_t> &bload, int nIn, int ngi, int nOut, int woff, int boff, int ngroups,
        bool load)
{
    static wword_t W[NWMEM];            // weights of this layer, on chip (kept between calls)
    static acc_t B[NBMEM];
#pragma HLS AGGREGATE variable = W
#pragma HLS BIND_STORAGE variable = W type = ram_2p impl = uram
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
    arow_t Z[NOUTMAX];                  // partial sums, one row of NB lanes per word
#pragma HLS AGGREGATE variable = Z
    sig_t xr[P][NB], xn[P][NB];
#pragma HLS ARRAY_PARTITION variable = xr type = complete dim = 0
#pragma HLS ARRAY_PARTITION variable = xn type = complete dim = 0
group:
    for (int g = 0; g < ngroups; g++) {
    first:
        for (int p = 0; p < P; p++) {
#pragma HLS PIPELINE II = 1
            row_t r;
            if (p < nIn) {
                r = in.read();
            } else {
                for (int b = 0; b < NB; b++) r.v[b] = 0;
            }
            for (int b = 0; b < NB; b++) xr[p][b] = r.v[b];
        }
        int gi = 0, o = 0;
        int inext = P;                      // next input to read
        int wrow = woff;                    // first word of the row of output o
    mac:
        for (int t = 0; t < ngi * nOut; t++) {
#pragma HLS PIPELINE II = 1
#pragma HLS DEPENDENCE variable = Z type = inter false
#pragma HLS LOOP_TRIPCOUNT max = 6 * 135
            if (o < P && gi + 1 < ngi) {        // inputs of the next group
                row_t r;
                if (inext < nIn) {
                    r = in.read();
                } else {
                    for (int b = 0; b < NB; b++) r.v[b] = 0;
                }
                for (int b = 0; b < NB; b++) xn[o][b] = r.v[b];
                inext++;
            }
            const wword_t wv = W[wrow + gi];
            const arow_t zo = Z[o];
            arow_t zn;
            for (int b = 0; b < NB; b++) {
                acc_t acc = (gi == 0) ? B[boff + o] : zo.v[b];
                for (int p = 0; p < P; p++) {
                    acc += wv.w[p] * xr[p][b];
                }
                zn.v[b] = acc;
            }
            Z[o] = zn;
            if (gi == ngi - 1) out.write(zn);
            if (o == nOut - 1) {
                for (int p = 0; p < P; p++)
                    for (int b = 0; b < NB; b++) xr[p][b] = xn[p][b];
                o = 0;
                gi++;
                wrow = woff;
            } else {
                o++;
                wrow += ngi;
            }
        }
    }
}

// LayerNorm over the nF features of each block: the exact sums in fixed
// point, the mean and the reciprocal standard deviation of each block in
// single precision, the normalization in fixed point (est_df.h). The rows
// go on to the residual add (skip).
template <int ID>
void layer_norm(hls::stream<row_t> &in, hls::stream<row_t> &out, hls::stream<row_t> &skip,
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
    row_t buf[NFMAX];
#pragma HLS AGGREGATE variable = buf
    acc_t s1[NB];
    sq_t s2[NB];
    lnm_t mean[NB];
    lnr_t rs[NB];
#pragma HLS ARRAY_PARTITION variable = s1 type = complete
#pragma HLS ARRAY_PARTITION variable = s2 type = complete
#pragma HLS ARRAY_PARTITION variable = mean type = complete
#pragma HLS ARRAY_PARTITION variable = rs type = complete
group:
    for (int g = 0; g < ngroups; g++) {
        for (int b = 0; b < NB; b++) {
#pragma HLS UNROLL
            s1[b] = 0;
            s2[b] = 0;
        }
    sums:
        for (int i = 0; i < nF; i++) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT max = 135
            row_t r = in.read();
            buf[i] = r;
            skip.write(r);
            for (int b = 0; b < NB; b++) {
                s1[b] += r.v[b];
                s2[b] += r.v[b] * r.v[b];
            }
        }
    scale:
        for (int b = 0; b < NB; b++) {
#pragma HLS PIPELINE II = 1
            const float m = (float)s1[b] / nF;
            float v = (float)s2[b] / nF - m * m;
            v = v > 0.0f ? v : 0.0f;
            mean[b] = m;
            rs[b] = 1.0f / hls::sqrt(v + 1e-5f);
        }
    normalize:
        for (int i = 0; i < nF; i++) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT max = 135
            const lnp_t g = gam[poff + i], be = bet[poff + i];
            row_t r;
            for (int b = 0; b < NB; b++) {
                const lnt_t t = (buf[i].v[b] - mean[b]) * rs[b];
                r.v[b] = t * g + be;
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
    layer_norm<R>(FIN, ln##R, skip##R, lns[R], d.nF, d.vec, ngroups, load);                   \
    fc<2 * R, NFMAX, SQWORDS, NVEC>(ln##R, z1##R, w1s[R], b1s[R], d.nF, d.ngi, d.nF, d.sq, d.vec, ngroups, load); \
    gelu(z1##R, a##R, d.nF, ngroups);                                                         \
    fc<2 * R + 1, NFMAX, SQWORDS, NVEC>(a##R, z2##R, w2s[R], b2s[R], d.nF, d.ngi, d.nF, d.sq, d.vec, ngroups, load); \
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
    hls::stream<row_t, 512> skip0, skip1, skip2;   // groups wait for their two FC layers
#pragma HLS BIND_STORAGE variable = skip0 type = fifo impl = uram
#pragma HLS BIND_STORAGE variable = skip1 type = fifo impl = uram
#pragma HLS BIND_STORAGE variable = skip2 type = fifo impl = uram
    hls::stream<arow_t, FIFO> z10, z11, z12, z20, z21, z22;
    loader(wmem, bmem, lnmem, w1s, w2s, wos, b1s, b2s, bos, lns, load);
    RESBLOCK(0, features, f1)
    RESBLOCK(1, f1, f2)
    RESBLOCK(2, f2, f3)
    fc<2 * NRES, NAMAX, OUTWORDS, NAALL>(f3, angles, wos, bos, d.nF, d.ngi, d.nA, d.out, d.a, ngroups, load);
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
