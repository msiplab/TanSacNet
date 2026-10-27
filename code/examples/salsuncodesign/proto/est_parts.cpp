// est_parts.cpp: LayerNorm, FC and GELU of residual block 0 of estimator 1
// as separate kernels, memory to memory (debugging the card-only errors)
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
#include "est_df.cpp"

namespace {
using namespace estdf;

constexpr int NF = NFK[0];

void readSig(const float *src, hls::stream<row_t> &out, int nRows)
{
    for (int i = 0; i < nRows; i++) {
        row_t r;
        for (int b = 0; b < NB; b++) {
#pragma HLS PIPELINE II = 1
            r.v[b] = src[i * NB + b];
        }
        out.write(r);
    }
}

void readAcc(const double *src, hls::stream<arow_t> &out, int nRows)
{
    for (int i = 0; i < nRows; i++) {
        arow_t r;
        for (int b = 0; b < NB; b++) {
#pragma HLS PIPELINE II = 1
            r.v[b] = src[i * NB + b];
        }
        out.write(r);
    }
}

void writeSig(hls::stream<row_t> &in, float *dst, int nRows)
{
    for (int i = 0; i < nRows; i++) {
        const row_t r = in.read();
        for (int b = 0; b < NB; b++) {
#pragma HLS PIPELINE II = 1
            dst[i * NB + b] = r.v[b].to_float();
        }
    }
}

void writeAcc(hls::stream<arow_t> &in, double *dst, int nRows)
{
    for (int i = 0; i < nRows; i++) {
        const arow_t r = in.read();
        for (int b = 0; b < NB; b++) {
#pragma HLS PIPELINE II = 1
            dst[i * NB + b] = r.v[b].to_double();
        }
    }
}

void drain(hls::stream<row_t> &in, int nRows)
{
    for (int i = 0; i < nRows; i++) {
#pragma HLS PIPELINE II = 1
        in.read();
    }
}

void loadLn(const float *lnmem, hls::stream<lnpar_t> &out)
{
    for (int i = 0; i < NVEC; i++) {
#pragma HLS PIPELINE II = 1
        out.write(lnpar_t{lnmem[2 * i], lnmem[2 * i + 1]});
    }
}

void loadFc(const wraw_t *wmem, const braw_t *bmem, hls::stream<wword_t> &ws, hls::stream<acc_t> &bs)
{
    for (int i = 0; i < SQWORDS; i++) {
#pragma HLS PIPELINE II = 1
        ws.write(toWord(wmem[i]));
    }
    for (int i = 0; i < NVEC; i++) {
#pragma HLS PIPELINE II = 1
        bs.write(toAcc(bmem[i]));
    }
}
} // namespace

// LayerNorm of block 0 (feature rows of estimator 1 in, normalized rows out)
extern "C" void part_ln(const float *src, float *dst, const float *lnmem, int ngroups)
{
#pragma HLS INTERFACE m_axi port = src offset = slave bundle = gmem0 depth = 8192
#pragma HLS INTERFACE m_axi port = dst offset = slave bundle = gmem1 depth = 8192
#pragma HLS INTERFACE m_axi port = lnmem offset = slave bundle = gmem2 depth = 2 * estdf::NLN
#pragma HLS DATAFLOW
    hls::stream<row_t, FIFO> in, out;
    hls::stream<row_t, 512> skip;
    hls::stream<lnpar_t, 4> lns;
    loadLn(lnmem, lns);
    readSig(src, in, ngroups * NF);
    layer_norm<0>(in, out, skip, lns, NF, 0, ngroups, true);
    drain(skip, ngroups * NF);
    writeSig(out, dst, ngroups * NF);
}

// First FC layer of block 0
extern "C" void part_fc(const float *src, double *dst, const estdf::wraw_t *wmem, const estdf::braw_t *bmem,
                        int ngroups)
{
#pragma HLS INTERFACE m_axi port = src offset = slave bundle = gmem0 depth = 8192
#pragma HLS INTERFACE m_axi port = dst offset = slave bundle = gmem1 depth = 8192
#pragma HLS INTERFACE m_axi port = wmem offset = slave bundle = gmem2 depth = estdf::SQWORDS
#pragma HLS INTERFACE m_axi port = bmem offset = slave bundle = gmem3 depth = estdf::NVEC
#pragma HLS DATAFLOW
    hls::stream<row_t, FIFO> in;
    hls::stream<arow_t, FIFO> out;
    hls::stream<wword_t, 4> ws;
    hls::stream<acc_t, 4> bs;
    loadFc(wmem, bmem, ws, bs);
    readSig(src, in, ngroups * NF);
    fc<0, NFMAX, SQWORDS, NVEC>(in, out, ws, bs, NF, words(NF), NF, 0, 0, ngroups, true);
    writeAcc(out, dst, ngroups * NF);
}

// GELU of the FC outputs
extern "C" void part_gelu(const double *src, float *dst, int ngroups)
{
#pragma HLS INTERFACE m_axi port = src offset = slave bundle = gmem0 depth = 8192
#pragma HLS INTERFACE m_axi port = dst offset = slave bundle = gmem1 depth = 8192
#pragma HLS DATAFLOW
    hls::stream<arow_t, FIFO> in;
    hls::stream<row_t, FIFO> out;
    readAcc(src, in, ngroups * NF);
    gelu(in, out, NF, ngroups);
    writeSig(out, dst, ngroups * NF);
}

// Intermediate values of the LayerNorm statistics of each block, as the
// HLS conversions and as f2fix give them (12 doubles per block: sums,
// sums as floats, mean and variance, rs as float, mean and rs through
// HLS, through f2fix), then gamma and beta of the NF features (HLS, f2fix)
extern "C" void part_probe(const float *src, double *dst, const float *lnmem, int ngroups)
{
#pragma HLS INTERFACE m_axi port = src offset = slave bundle = gmem0 depth = 8192
#pragma HLS INTERFACE m_axi port = dst offset = slave bundle = gmem1 depth = 8192
#pragma HLS INTERFACE m_axi port = lnmem offset = slave bundle = gmem2 depth = 2 * estdf::NLN
    int k = 0;
    for (int g = 0; g < ngroups; g++) {
        for (int b = 0; b < NB; b++) {
            acc_t s1 = 0;
            sq_t s2 = 0;
            for (int i = 0; i < NF; i++) {
#pragma HLS PIPELINE II = 1
                const sig_t x = src[(g * NF + i) * NB + b];
                s1 += x;
                s2 += x * x;
            }
            const float f1 = (float)s1, f2 = (float)s2;
            const float m = f1 / NF;
            float v = f2 / NF - m * m;
            v = v > 0.0f ? v : 0.0f;
            const float rsf = 1.0f / hls::sqrt(v + 1e-5f);
            const lnm_t mh = m;
            const lnr_t rh = rsf;
            const double o[12] = {s1.to_double(), s2.to_double(), f1, f2, m, v, rsf, mh.to_double(), rh.to_double(),
                                  f2fix<lnm_t, true>(m).to_double(), f2fix<lnr_t, false>(rsf).to_double(), 0.0};
            for (int j = 0; j < 12; j++) dst[k++] = o[j];
        }
    }
    for (int i = 0; i < NF; i++) {
        const float ga = lnmem[2 * i], be = lnmem[2 * i + 1];
        const lnp_t gh = ga, bh = be;
        const double o[4] = {gh.to_double(), bh.to_double(), f2fix<lnp_t, true>(ga).to_double(),
                             f2fix<lnp_t, true>(be).to_double()};
        for (int j = 0; j < 4; j++) dst[k++] = o[j];
    }
}
