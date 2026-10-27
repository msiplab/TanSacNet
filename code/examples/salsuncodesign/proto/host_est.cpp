// host_est.cpp: runs est_kernel on the card with the test data of
// export_est_df.m (./testdata) and compares the angles with the MATLAB
// fixed-point model (debugging)
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
#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static std::vector<double> readBin(const std::string &name)
{
    std::vector<double> v;
    FILE *f = std::fopen(("testdata/" + name).c_str(), "rb");
    double x;
    while (f && std::fread(&x, sizeof x, 1, f) == 1) v.push_back(x);
    if (f) std::fclose(f);
    return v;
}

int main(int argc, char **argv)
{
    const int NB = 5, P = 24, NFK[5] = {135, 72, 72, 72, 72}, NAK[5] = {49, 28, 28, 28, 28};
    auto w = readBin("wmem.bin"), b = readBin("bmem.bin"), l = readBin("ln.bin");
    const size_t nWords = w.size() / P;
    std::vector<uint8_t> words(nWords * 64, 0);
    for (size_t i = 0; i < nWords; i++) {
        uint64_t *word = reinterpret_cast<uint64_t *>(words.data() + i * 64);
        for (int p = 0; p < P; p++) {
            const uint64_t raw = static_cast<uint64_t>(std::llround(w[i * P + p] * 65536.0)) & 0x3FFFFu;
            const int bit = 18 * p, idx = bit / 64, off = bit % 64;
            word[idx] |= raw << off;
            if (off + 18 > 64) word[idx + 1] |= raw >> (64 - off);
        }
    }
    std::vector<int64_t> bias(b.size());
    for (size_t i = 0; i < b.size(); i++) bias[i] = std::llround(b[i] * 134217728.0);
    std::vector<float> ln(l.begin(), l.end());

    xrt::device dev(0);
    auto uuid = dev.load_xclbin(argv[1]);
    xrt::kernel k(dev, uuid, "est_kernel");
    xrt::bo bw(dev, words.size(), k.group_id(2)), bb(dev, bias.size() * 8, k.group_id(3)),
        bl(dev, ln.size() * 4, k.group_id(4));
    std::memcpy(bw.map<uint8_t *>(), words.data(), words.size());
    std::memcpy(bb.map<int64_t *>(), bias.data(), bias.size() * 8);
    std::memcpy(bl.map<float *>(), ln.data(), ln.size() * 4);
    bw.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bb.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bl.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    for (int e = 0; e < 5; e++) {
        auto feat = readBin("feat" + std::to_string(e) + ".bin"), ref = readBin("ref" + std::to_string(e) + ".bin");
        const int ng = (int)feat.size() / (NFK[e] * NB);
        xrt::bo bf(dev, feat.size() * 4, k.group_id(0)), ba(dev, (size_t)ng * NAK[e] * NB * 8, k.group_id(1));
        float *fp = bf.map<float *>();
        for (size_t i = 0; i < feat.size(); i++) fp[i] = (float)feat[i];
        bf.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        auto run = k(bf, ba, bw, bb, bl, e, ng);
        run.wait();
        ba.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        const double *ap = ba.map<double *>();
        double err = 0;
        int nbad = 0;
        for (size_t i = 0; i < ref.size(); i++) {
            const double d = std::fabs(ap[i] - ref[i]);
            err = std::fmax(err, d);
            nbad += d > 1e-6;
        }
        std::printf("estimator %d: %d groups, max|angle - fixed model| = %.3g, %d of %zu differ; first %.6f vs %.6f\n",
                    e + 1, ng, err, nbad, ref.size(), ap[0], ref[0]);
    }
    return 0;
}
