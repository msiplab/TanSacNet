// tb_band_df.cpp: C simulation of salsun2d_band_df against the MATLAB
// fixed-point model (test data from export_band_df.m in ./testdata_band)
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

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using namespace bdf;

static std::vector<double> readBin(const std::string &name)
{
    std::vector<double> v;
    FILE *f = std::fopen(("testdata_band/" + name).c_str(), "rb");
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
    const auto x = readBin("x.bin"), prm = readBin("prm.bin"), w = readBin("wmem.bin"),
               b = readBin("bmem.bin"), l = readBin("ln.bin"), st = readBin("stats.bin"),
               yref = readBin("yref.bin"), sref = readBin("sums.bin");
    if ((int)x.size() != SZY * SZX || (int)w.size() != NWORDS * P) {
        std::printf("unexpected sizes: x %zu (%d), w %zu (%d)\n", x.size(), SZY * SZX, w.size(), NWORDS * P);
        return 1;
    }
    static float src[SZY * SZX], dst[SZY * SZX], fprm[NPRM], sin_[2 * NSTATS], sout[2 * NSTATS];
    static wword_t wmem[NWORDS];
    static acc_t bmem[NBIAS];
    static lnpar_t lnmem[NLN];
    static qword_t qmem[NQMEM];
    for (int i = 0; i < SZY * SZX; i++) src[i] = (float)x[i];
    for (int i = 0; i < NPRM; i++) fprm[i] = (float)prm[i];
    for (int i = 0; i < NWORDS; i++)
        for (int p = 0; p < P; p++) wmem[i].w[p] = w[i * P + p];
    for (int i = 0; i < NBIAS; i++) bmem[i] = b[i];
    for (int i = 0; i < NLN; i++) {
        lnmem[i].gamma = (float)l[2 * i];
        lnmem[i].beta = (float)l[2 * i + 1];
    }
    for (int i = 0; i < 2 * NSTATS; i++) sin_[i] = (float)st[i];
    salsun2d_band_df(src, fprm, wmem, bmem, lnmem, sin_, dst, sout, qmem, 0, NBANDS);
    double ey = 0, ymax = 0;
    for (int i = 0; i < SZY * SZX; i++) {
        ey = std::fmax(ey, std::fabs(dst[i] - yref[i]));
        ymax = std::fmax(ymax, std::fabs(yref[i]));
    }
    double es = 0;
    for (int i = 0; i < 2 * NSTATS; i++) {
        if (sref[i] != 0) es = std::fmax(es, std::fabs(sout[i] - sref[i]) / (std::fabs(sref[i]) + 1e-3));
    }
    for (int k = 0; k < NEST; k++) {
        double e1 = 0, m1 = 0;
        for (int j = 0; j < NDEC; j++) {
            e1 = std::fmax(e1, std::fabs(sout[k * NDEC + j] - sref[k * NDEC + j]));
            m1 = std::fmax(m1, std::fabs(sref[k * NDEC + j]));
        }
        std::printf("estimator %d inputs: max|sum - model| = %.3g (max|sum| %.3g); ch %d: %.6g vs %.6g\n", k + 1, e1,
                    m1, k == 0 ? 1 : 8, sout[k * NDEC + (k == 0 ? 1 : 8)], sref[k * NDEC + (k == 0 ? 1 : 8)]);
    }
    std::printf("%dx%d, %d bands of %d rows (halo %d): max|y - fixed model| = %.3g (max|y| %.3g), "
                "max rel. error of the sums = %.3g\n",
                SZY, SZX, NBANDS, BAND, HALO, ey, ymax, es);
    return ey > 1e-4 ? 1 : 0;
}
