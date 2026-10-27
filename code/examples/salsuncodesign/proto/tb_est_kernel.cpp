// tb_est_kernel.cpp: C/RTL cosimulation of est_kernel (the debugging kernel run on the card)
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

extern "C" void est_kernel(const float *feat, double *ang, const estdf::wraw_t *wmem, const estdf::braw_t *bmem,
                           const float *lnmem, int k, int ngroups);

#include <algorithm>
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
        const int ngroups = (int)feat.size() / (NFK[k] * NB);
        std::vector<float> f(8192, 0.0f);              // the depths of the ports (cosim copies them whole)
        std::copy(feat.begin(), feat.end(), f.begin());
        std::vector<double> ang(4096);
        est_kernel(f.data(), ang.data(), wmem, bmem, lnmem, k, ngroups);
        double err = 0;
        int nbad = 0;
        for (size_t i = 0; i < ref.size(); i++) {
            const double d = std::fabs(ang[i] - ref[i]);
            err = std::fmax(err, d);
            nbad += d > 1e-6;
        }
        std::printf("estimator %d: %d groups, max|angle - fixed model| = %.3g, %d of %zu differ\n", k + 1, ngroups,
                    err, nbad, ref.size());
        fails += nbad > 0;
    }
    return fails;
}
