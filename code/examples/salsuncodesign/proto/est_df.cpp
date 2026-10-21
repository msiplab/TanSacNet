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

namespace estdf {

// Fully connected layer, output stationary over the NB lanes: for each
// group of P inputs (read from the stream, P rows), every cycle adds P
// products to one output of all NB blocks. The outputs of a group are
// sent once all inputs are summed.
template <int NIN, int NINP, int NOUT>
void fc(hls::stream<row_t> &in, hls::stream<arow_t> &out, hls::stream<wword_t> &wload,
        hls::stream<acc_t> &bload, int ngroups)
{
    constexpr int NGI = NINP / P;
    wword_t W[NOUT][NGI];               // weights of this layer, on chip
    acc_t B[NOUT];
#pragma HLS AGGREGATE variable = W
    acc_t Z[NOUT][NB];
#pragma HLS ARRAY_PARTITION variable = Z type = complete dim = 2
    sig_t xr[P][NB];
#pragma HLS ARRAY_PARTITION variable = xr type = complete dim = 0
load_w:
    for (int k = 0; k < NOUT * NGI; k++) {
#pragma HLS PIPELINE II = 1
        W[k / NGI][k % NGI] = wload.read();
    }
load_b:
    for (int o = 0; o < NOUT; o++) {
#pragma HLS PIPELINE II = 1
        B[o] = bload.read();
    }
group:
    for (int g = 0; g < ngroups; g++) {
    inputs:
        for (int gi = 0; gi < NGI; gi++) {
        load:
            for (int p = 0; p < P; p++) {
#pragma HLS PIPELINE II = 1
                row_t r;
                if (gi * P + p < NIN) {
                    r = in.read();
                } else {
                    for (int b = 0; b < NB; b++) r.v[b] = 0;
                }
                for (int b = 0; b < NB; b++) xr[p][b] = r.v[b];
            }
        outputs:
            for (int o = 0; o < NOUT; o++) {
#pragma HLS PIPELINE II = 1
#pragma HLS DEPENDENCE variable = Z type = inter false
                wword_t wv = W[o][gi];
                for (int b = 0; b < NB; b++) {
                    acc_t acc = (gi == 0) ? B[o] : Z[o][b];
                    for (int p = 0; p < P; p++) {
                        acc += wv.w[p] * xr[p][b];
                    }
                    Z[o][b] = acc;
                }
            }
        }
    emit:
        for (int o = 0; o < NOUT; o++) {
#pragma HLS PIPELINE II = 1
            arow_t r;
            for (int b = 0; b < NB; b++) r.v[b] = Z[o][b];
            out.write(r);
        }
    }
}

// LayerNorm over the NF features of each block (fixed point): the rows of
// a group are buffered while their sums are taken, then sent normalized;
// the rows themselves go on to the residual add (skip)
void layer_norm(hls::stream<row_t> &in, hls::stream<row_t> &out, hls::stream<row_t> &skip,
                hls::stream<lnpar_t> &lnload, int ngroups)
{
    sig_t gamma[NF], beta[NF];
load:
    for (int i = 0; i < NF; i++) {
#pragma HLS PIPELINE II = 1
        lnpar_t q = lnload.read();
        gamma[i] = q.gamma;
        beta[i] = q.beta;
    }
    row_t buf[NF];
    acc_t s1[NB], s2[NB];
    ap_fixed<24, 8> mean[NB];
    ap_ufixed<24, 8> rs[NB];
#pragma HLS ARRAY_PARTITION variable = buf type = complete dim = 0
#pragma HLS ARRAY_PARTITION variable = s1 type = complete
#pragma HLS ARRAY_PARTITION variable = s2 type = complete
#pragma HLS ARRAY_PARTITION variable = mean type = complete
#pragma HLS ARRAY_PARTITION variable = rs type = complete
#pragma HLS AGGREGATE variable = buf
group:
    for (int g = 0; g < ngroups; g++) {
        for (int b = 0; b < NB; b++) {
#pragma HLS UNROLL
            s1[b] = 0;
            s2[b] = 0;
        }
    sums:
        for (int i = 0; i < NF; i++) {
#pragma HLS PIPELINE II = 1
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
            float m = (float)s1[b] / NF;
            float v = (float)s2[b] / NF - m * m;
            mean[b] = m;
            rs[b] = 1.0f / hls::sqrt(v + 1e-5f);
        }
    normalize:
        for (int i = 0; i < NF; i++) {
#pragma HLS PIPELINE II = 1
            row_t r;
            for (int b = 0; b < NB; b++) {
                r.v[b] = (sig_t)((buf[i].v[b] - mean[b]) * rs[b] * gamma[i] + beta[i]);
            }
            out.write(r);
        }
    }
}

// GELU (tanh form) in single precision, NB/5 units: five cycles per row
void gelu(hls::stream<arow_t> &in, hls::stream<row_t> &out, int ngroups)
{
    constexpr int NU = 3;
    constexpr int NS = NB / NU;
group:
    for (int g = 0; g < ngroups; g++) {
    rows:
        for (int i = 0; i < NH; i++) {
            arow_t z;
            row_t a;
        step:
            for (int s = 0; s < NS; s++) {
#pragma HLS PIPELINE II = 1
                if (s == 0) z = in.read();
                for (int u = 0; u < NU; u++) {
                    int b = s * NU + u;
                    float x = (float)z.v[b];
                    float t = hls::tanh(0.7978845608f * (x + 0.044715f * x * x * x));
                    a.v[b] = (sig_t)(0.5f * x * (1.0f + t));
                }
                if (s == NS - 1) out.write(a);
            }
        }
    }
}

// Residual add: the block input (skip) plus the second FC layer
void residual_add(hls::stream<arow_t> &z, hls::stream<row_t> &skip, hls::stream<row_t> &out, int ngroups)
{
group:
    for (int g = 0; g < ngroups; g++) {
    rows:
        for (int i = 0; i < NF; i++) {
#pragma HLS PIPELINE II = 1
            arow_t zr = z.read();
            row_t sr = skip.read();
            row_t r;
            for (int b = 0; b < NB; b++) {
                r.v[b] = (sig_t)(sr.v[b] + zr.v[b]);
            }
            out.write(r);
        }
    }
}

// Distributes the parameters from memory to the tasks that keep them
void loader(const wword_t *wmem, const acc_t *bmem, const lnpar_t *lnmem,
            hls::stream<wword_t> w1s[NRES], hls::stream<wword_t> w2s[NRES], hls::stream<wword_t> &wos,
            hls::stream<acc_t> b1s[NRES], hls::stream<acc_t> b2s[NRES], hls::stream<acc_t> &bos,
            hls::stream<lnpar_t> lns[NRES])
{
    int kw = 0, kb = 0;
    for (int r = 0; r < NRES; r++) {
        for (int k = 0; k < NW1; k++) {
#pragma HLS PIPELINE II = 1
            w1s[r].write(wmem[kw++]);
        }
        for (int k = 0; k < NW2; k++) {
#pragma HLS PIPELINE II = 1
            w2s[r].write(wmem[kw++]);
        }
        for (int k = 0; k < NH; k++) {
#pragma HLS PIPELINE II = 1
            b1s[r].write(bmem[kb++]);
        }
        for (int k = 0; k < NF; k++) {
#pragma HLS PIPELINE II = 1
            b2s[r].write(bmem[kb++]);
        }
        for (int k = 0; k < NF; k++) {
#pragma HLS PIPELINE II = 1
            lns[r].write(lnmem[r * NF + k]);
        }
    }
    for (int k = 0; k < NWO; k++) {
#pragma HLS PIPELINE II = 1
        wos.write(wmem[kw++]);
    }
    for (int k = 0; k < NA; k++) {
#pragma HLS PIPELINE II = 1
        bos.write(bmem[kb++]);
    }
}

// One residual block: LayerNorm, FC, GELU, FC, skip
#define RESBLOCK(R, FIN, FOUT)                                                  \
    layer_norm(FIN, ln##R, skip##R, lns[R], ngroups);                         \
    fc<NF, NFP, NH>(ln##R, z1##R, w1s[R], b1s[R], ngroups);                   \
    gelu(z1##R, a##R, ngroups);                                               \
    fc<NH, NHP, NF>(a##R, z2##R, w2s[R], b2s[R], ngroups);                    \
    residual_add(z2##R, skip##R, FOUT, ngroups);

} // namespace estdf

// Top function for synthesis (outside the namespace)
void est_df_top(const estdf::wword_t *wmem, const estdf::acc_t *bmem, const estdf::lnpar_t *lnmem,
                hls::stream<estdf::row_t> &features, hls::stream<estdf::arow_t> &angles, int ngroups)
{
#pragma HLS INTERFACE m_axi port = wmem offset = slave bundle = gmem0 depth = estdf::NWORDS
#pragma HLS INTERFACE m_axi port = bmem offset = slave bundle = gmem1 depth = estdf::NBIAS
#pragma HLS INTERFACE m_axi port = lnmem offset = slave bundle = gmem2 depth = estdf::NLN
    using namespace estdf;
#pragma HLS DATAFLOW
    hls::stream<wword_t, 4> w1s[NRES], w2s[NRES], wos;
    hls::stream<acc_t, 4> b1s[NRES], b2s[NRES], bos;
    hls::stream<lnpar_t, 4> lns[NRES];
    hls::stream<row_t, FIFO> ln0, ln1, ln2, a0, a1, a2, f1, f2, f3;
    hls::stream<row_t, 512> skip0, skip1, skip2;   // a group waits for its two FC layers
    hls::stream<arow_t, FIFO> z10, z11, z12, z20, z21, z22;
    loader(wmem, bmem, lnmem, w1s, w2s, wos, b1s, b2s, bos, lns);
    RESBLOCK(0, features, f1)
    RESBLOCK(1, f1, f2)
    RESBLOCK(2, f2, f3)
    fc<NF, NFP, NA>(f3, angles, wos, bos, ngroups);
}
