// tb_est_df.cpp: C simulation of est_df_top against the MATLAB models
// (test data from export_est_df.m in ./testdata)
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

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using namespace estdf;

static std::vector<double> readBin(const std::string &name)
{
    std::vector<double> v;
    FILE *f = std::fopen(("testdata/" + name).c_str(), "rb");
    if (!f) {
        std::printf("cannot open %s\n", name.c_str());
        return v;
    }
    double x;
    while (std::fread(&x, sizeof x, 1, f) == 1) v.push_back(x);
    std::fclose(f);
    return v;
}

int main()
{
    const std::vector<double> w = readBin("wmem.bin"), b = readBin("bmem.bin"), l = readBin("ln.bin");
    if ((int)w.size() != NWORDS * P || (int)b.size() != NBIAS || (int)l.size() != 2 * NLN) {
        std::printf("unexpected sizes: %zu %zu %zu (expected %d %d %d)\n", w.size(), b.size(), l.size(),
                    NWORDS * P, NBIAS, 2 * NLN);
        return 1;
    }
    static wraw_t wmem[NWORDS];
    static braw_t bmem[NBIAS];
    static float lnmem[2 * NLN];
    for (int i = 0; i < NWORDS; i++) {
        wmem[i] = 0;
        for (int p = 0; p < P; p++) {
            const wgt_t v = w[i * P + p];
            wmem[i].range(18 * p + 17, 18 * p) = v.range(17, 0);
        }
    }
    for (int i = 0; i < NBIAS; i++) {
        const acc_t a = b[i];
        bmem[i] = 0;
        bmem[i].range(47, 0) = a.range(47, 0);
        bmem[i] = (bmem[i] << 16) >> 16;           // sign-extended
    }
    for (int i = 0; i < 2 * NLN; i++) lnmem[i] = (float)l[i];
    int fails = 0;
    for (int k = 0; k < NEST; k++) {
        const std::vector<double> feat = readBin("feat" + std::to_string(k) + ".bin");
        const std::vector<double> ref = readBin("ref" + std::to_string(k) + ".bin");
        const std::vector<double> flt = readBin("flt" + std::to_string(k) + ".bin");
        const int nF = NFK[k], nA = NAK[k];
        const int ngroups = (int)feat.size() / (nF * NB);
        hls::stream<row_t> in;
        hls::stream<arow_t> out;
        for (int g = 0; g < ngroups; g++)
            for (int i = 0; i < nF; i++) {
                row_t r;
                for (int bb = 0; bb < NB; bb++) r.v[bb] = feat[bb + NB * (i + nF * g)];
                in.write(r);
            }
        est_engine(in, out, wmem, bmem, lnmem, k, ngroups, k == 0);   // parameters loaded once
        double eRef = 0, eFlt = 0, eQ = 0;
        for (int g = 0; g < ngroups; g++)
            for (int a = 0; a < nA; a++) {
                const arow_t r = out.read();
                for (int bb = 0; bb < NB; bb++) {
                    const int j = bb + NB * (a + nA * g);
                    const double v = r.v[bb].to_double();
                    eRef = std::fmax(eRef, std::fabs(v - ref[j]));
                    eFlt = std::fmax(eFlt, std::fabs(v - flt[j]));
                    eQ = std::fmax(eQ, std::fabs(ref[j] - flt[j]));
                }
            }
        std::printf("estimator %d: %d groups, max|angle - fixed model| = %.3g rad, "
                    "max|angle - float| = %.3g (fixed model - float: %.3g)\n",
                    k + 1, ngroups, eRef, eFlt, eQ);
        if (eRef > 1e-3) fails++;
    }
    return fails;
}
