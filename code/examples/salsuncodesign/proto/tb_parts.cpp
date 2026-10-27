// tb_parts.cpp: C simulation of the est_parts.cpp kernels on estimator 1; writes their outputs to testdata/parts_*.bin as the references of the card test (host_parts.cpp)
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

extern "C" void part_ln(const float *src, float *dst, const float *lnmem, int ngroups);
extern "C" void part_fc(const float *src, double *dst, const estdf::wraw_t *wmem, const estdf::braw_t *bmem,
                        int ngroups);
extern "C" void part_gelu(const double *src, float *dst, int ngroups);
extern "C" void part_probe(const float *src, double *dst, const float *lnmem, int ngroups);


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

static void writeBin(const std::string &name, const std::vector<double> &v)
{
    FILE *f = std::fopen(("testdata/" + name).c_str(), "wb");
    std::fwrite(v.data(), sizeof(double), v.size(), f);
    std::fclose(f);
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
    const std::vector<double> feat = readBin("feat0.bin");
    const int ngroups = (int)feat.size() / (NFK[0] * NB), n = ngroups * NFK[0] * NB;
    std::vector<float> x(8192, 0.0f), ln(8192), gl(8192);
    std::vector<double> z(8192);
    std::copy(feat.begin(), feat.end(), x.begin());
    part_ln(x.data(), ln.data(), lnmem, ngroups);
    part_fc(ln.data(), z.data(), wmem, bmem, ngroups);
    part_gelu(z.data(), gl.data(), ngroups);
    writeBin("parts_ln.bin", std::vector<double>(ln.begin(), ln.begin() + n));
    writeBin("parts_fc.bin", std::vector<double>(z.begin(), z.begin() + n));
    writeBin("parts_gelu.bin", std::vector<double>(gl.begin(), gl.begin() + n));
    std::vector<double> pr(8192);
    part_probe(x.data(), pr.data(), lnmem, ngroups);
    writeBin("parts_probe.bin", std::vector<double>(pr.begin(), pr.begin() + ngroups * NB * 12 + NFK[0] * 4));
    std::printf("probe: s1 %g s2 %g m %g v %g rs %g mean %g/%g rs %g/%g\n", pr[0], pr[1], pr[4], pr[5], pr[6], pr[7],
                pr[9], pr[8], pr[10]);
    std::printf("%d groups: wrote parts_ln, parts_fc, parts_gelu (%d values each); first %g %g %g\n", ngroups, n,
                ln[0], z[0], gl[0]);
    return 0;
}
