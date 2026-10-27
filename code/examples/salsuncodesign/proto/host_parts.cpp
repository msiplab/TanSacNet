// host_parts.cpp: runs the est_parts.cpp kernels on the card, each on the
// C-simulated input of its stage (testdata/parts_*.bin, from tb_parts.cpp),
// and compares the outputs with the C simulation (debugging)
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

static void compare(const char *name, const double *y, const std::vector<double> &ref)
{
    double err = 0;
    int nbad = 0, first = -1;
    for (size_t i = 0; i < ref.size(); i++) {
        const double d = std::fabs(y[i] - ref[i]);
        err = std::fmax(err, d);
        if (d > 1e-9 && first < 0) first = (int)i;
        nbad += d > 1e-9;
    }
    std::printf("%-5s: max|card - csim| = %.3g, %d of %zu differ", name, err, nbad, ref.size());
    if (first >= 0) std::printf("; first at %d (row %d, lane %d): %.9g vs %.9g", first, first / 5, first % 5, y[first], ref[first]);
    std::printf("\n");
}

int main(int argc, char **argv)
{
    const int P = 24, NWLN = 1674, NVEC = 423;   // words and biases of the first FC layer of block 0
    auto w = readBin("wmem.bin"), b = readBin("bmem.bin"), l = readBin("ln.bin");
    auto feat = readBin("feat0.bin"), lnRef = readBin("parts_ln.bin"), fcRef = readBin("parts_fc.bin"),
         gRef = readBin("parts_gelu.bin"), pRef = readBin("parts_probe.bin");
    const int ng = (int)feat.size() / (135 * 5), n = (int)feat.size();
    std::vector<uint8_t> words(NWLN * 64, 0);
    for (int i = 0; i < NWLN; i++) {
        uint64_t *word = reinterpret_cast<uint64_t *>(words.data() + i * 64);
        for (int p = 0; p < P; p++) {
            const uint64_t raw = static_cast<uint64_t>(std::llround(w[i * P + p] * 65536.0)) & 0x3FFFFu;
            const int bit = 18 * p, idx = bit / 64, off = bit % 64;
            word[idx] |= raw << off;
            if (off + 18 > 64) word[idx + 1] |= raw >> (64 - off);
        }
    }

    xrt::device dev(0);
    auto uuid = dev.load_xclbin(argv[1]);
    xrt::kernel kln(dev, uuid, "part_ln"), kfc(dev, uuid, "part_fc"), kg(dev, uuid, "part_gelu"),
        kp(dev, uuid, "part_probe");

    // LayerNorm on the features
    {
        xrt::bo bx(dev, n * 4, kln.group_id(0)), by(dev, n * 4, kln.group_id(1)), bl(dev, l.size() * 4, kln.group_id(2));
        float *xp = bx.map<float *>(), *lp = bl.map<float *>();
        for (int i = 0; i < n; i++) xp[i] = (float)feat[i];
        for (size_t i = 0; i < l.size(); i++) lp[i] = (float)l[i];
        bx.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        bl.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        kln(bx, by, bl, ng).wait();
        by.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        const float *yp = by.map<float *>();
        std::vector<double> y(yp, yp + n);
        compare("ln", y.data(), lnRef);
        FILE *f = std::fopen("testdata/card_ln.bin", "wb");     // for the analysis of the errors
        std::fwrite(y.data(), sizeof(double), y.size(), f);
        std::fclose(f);
    }
    // LayerNorm on the features rounded to the signal format [18 11] on the
    // host, and FC on its input plus less than half an LSB (both must give
    // the same outputs if the conversion of the kernels is right)
    {
        xrt::bo bx(dev, n * 4, kln.group_id(0)), by(dev, n * 4, kln.group_id(1)), bl(dev, l.size() * 4, kln.group_id(2));
        float *xp = bx.map<float *>(), *lp = bl.map<float *>();
        for (int i = 0; i < n; i++) xp[i] = (float)(std::floor(feat[i] * 2048 + 0.5) / 2048);
        for (size_t i = 0; i < l.size(); i++) lp[i] = (float)l[i];
        bx.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        bl.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        kln(bx, by, bl, ng).wait();
        by.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        const float *yp = by.map<float *>();
        std::vector<double> y(yp, yp + n);
        compare("ln/r", y.data(), lnRef);
    }
    {
        xrt::bo bx(dev, n * 4, kfc.group_id(0)), by(dev, n * 8, kfc.group_id(1)), bw(dev, words.size(), kfc.group_id(2)),
            bb(dev, NVEC * 8, kfc.group_id(3));
        float *xp = bx.map<float *>();
        for (int i = 0; i < n; i++) xp[i] = (float)(lnRef[i] + 0.3 / 2048);
        std::memcpy(bw.map<uint8_t *>(), words.data(), words.size());
        int64_t *bp = bb.map<int64_t *>();
        for (int i = 0; i < NVEC; i++) bp[i] = std::llround(b[i] * 134217728.0);
        bx.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        bw.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        bb.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        kfc(bx, by, bw, bb, ng).wait();
        by.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        compare("fc/p", by.map<double *>(), fcRef);
    }
    // FC on the C-simulated LayerNorm output
    {
        xrt::bo bx(dev, n * 4, kfc.group_id(0)), by(dev, n * 8, kfc.group_id(1)), bw(dev, words.size(), kfc.group_id(2)),
            bb(dev, NVEC * 8, kfc.group_id(3));
        float *xp = bx.map<float *>();
        for (int i = 0; i < n; i++) xp[i] = (float)lnRef[i];
        std::memcpy(bw.map<uint8_t *>(), words.data(), words.size());
        int64_t *bp = bb.map<int64_t *>();
        for (int i = 0; i < NVEC; i++) bp[i] = std::llround(b[i] * 134217728.0);
        bx.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        bw.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        bb.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        kfc(bx, by, bw, bb, ng).wait();
        by.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        compare("fc", by.map<double *>(), fcRef);
    }
    // GELU on the C-simulated FC output
    {
        xrt::bo bx(dev, n * 8, kg.group_id(0)), by(dev, n * 4, kg.group_id(1));
        std::memcpy(bx.map<double *>(), fcRef.data(), n * 8);
        bx.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        kg(bx, by, ng).wait();
        by.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        const float *yp = by.map<float *>();
        std::vector<double> y(yp, yp + n);
        compare("gelu", y.data(), gRef);
    }
    // LayerNorm statistics and parameters, value by value
    {
        xrt::bo bx(dev, n * 4, kp.group_id(0)), by(dev, pRef.size() * 8, kp.group_id(1)),
            bl(dev, l.size() * 4, kp.group_id(2));
        float *xp = bx.map<float *>(), *lp = bl.map<float *>();
        for (int i = 0; i < n; i++) xp[i] = (float)feat[i];
        for (size_t i = 0; i < l.size(); i++) lp[i] = (float)l[i];
        bx.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        bl.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        kp(bx, by, bl, ng).wait();
        by.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        const double *yp = by.map<double *>();
        const char *names[12] = {"s1", "s2", "(float)s1", "(float)s2", "m", "v", "rs float", "mean HLS", "rs HLS",
                                 "mean f2fix", "rs f2fix", "-"};
        const int nblk = ng * 5;
        for (int j = 0; j < 11; j++) {
            int bad = 0, first = -1;
            for (int q = 0; q < nblk; q++)
                if (yp[q * 12 + j] != pRef[q * 12 + j] && bad++ == 0) first = q;
            std::printf("probe %-10s: %3d of %d blocks differ", names[j], bad, nblk);
            if (first >= 0) std::printf("; block %d: %.9g vs %.9g", first, yp[first * 12 + j], pRef[first * 12 + j]);
            std::printf("\n");
        }
        const char *pn[4] = {"gamma HLS", "beta HLS", "gamma f2fix", "beta f2fix"};
        for (int j = 0; j < 4; j++) {
            int bad = 0, first = -1;
            for (int i = 0; i < 135; i++)
                if (yp[nblk * 12 + i * 4 + j] != pRef[nblk * 12 + i * 4 + j] && bad++ == 0) first = i;
            std::printf("probe %-10s: %3d of 135 differ", pn[j], bad);
            if (first >= 0)
                std::printf("; feature %d: %.9g vs %.9g", first, yp[nblk * 12 + first * 4 + j], pRef[nblk * 12 + first * 4 + j]);
            std::printf("\n");
        }
    }
    return 0;
}
