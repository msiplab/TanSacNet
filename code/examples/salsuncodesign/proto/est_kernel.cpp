// est_kernel.cpp: the dataflow estimator alone as a kernel (debugging):
// feature rows from memory, angle rows to memory
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

namespace {
using namespace estdf;

void readRows(const float *feat, hls::stream<row_t> &out, int nRows)
{
    for (int i = 0; i < nRows; i++) {
        row_t r;
        for (int b = 0; b < NB; b++) {
#pragma HLS PIPELINE II = 1
            r.v[b] = feat[i * NB + b];
        }
        out.write(r);
    }
}

void writeRows(hls::stream<arow_t> &in, double *ang, int nRows)
{
    for (int i = 0; i < nRows; i++) {
        const arow_t r = in.read();
        for (int b = 0; b < NB; b++) {
#pragma HLS PIPELINE II = 1
            ang[i * NB + b] = r.v[b].to_double();
        }
    }
}

void run(const float *feat, double *ang, const wraw_t *wmem, const braw_t *bmem, const float *lnmem, int k,
         int ngroups)
{
#pragma HLS DATAFLOW
    hls::stream<row_t, FIFO> f;
    hls::stream<arow_t, FIFO> a;
    readRows(feat, f, ngroups * NFK[k]);
    est_engine(f, a, wmem, bmem, lnmem, k, ngroups, true);
    writeRows(a, ang, ngroups * NAK[k]);
}
} // namespace

extern "C" void est_kernel(const float *feat, double *ang, const estdf::wraw_t *wmem, const estdf::braw_t *bmem,
                           const float *lnmem, int k, int ngroups)
{
#pragma HLS INTERFACE m_axi port = feat offset = slave bundle = gmem0 depth = 8192
#pragma HLS INTERFACE m_axi port = ang offset = slave bundle = gmem1 depth = 4096
#pragma HLS INTERFACE m_axi port = wmem offset = slave bundle = gmem2 depth = estdf::NWORDS
#pragma HLS INTERFACE m_axi port = bmem offset = slave bundle = gmem3 depth = estdf::NBIAS
#pragma HLS INTERFACE m_axi port = lnmem offset = slave bundle = gmem4 depth = 2 * estdf::NLN
    run(feat, ang, wmem, bmem, lnmem, k, ngroups);
}
