// host_band.cpp: runs one compute unit of salsun2d_band_df on the card
// with the Full HD test data of export_band_df.m (./testdata_fhd) over
// chosen bands, and compares each band with the MATLAB fixed-point model
// (debugging). The output is filled with NaN before each run.
//
//   host_band <xclbin> <units, e.g. 2 or 123> <first>:<count> [<first>:<count> ...]
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
#include <xrt/experimental/xrt_xclbin.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

static std::vector<double> readBin(const std::string &name)
{
    std::vector<double> v;
    const char *dir = std::getenv("TESTDIR");
    FILE *f = std::fopen(((dir ? std::string(dir) : std::string("testdata_fhd")) + "/" + name).c_str(), "rb");
    double x;
    while (f && std::fread(&x, sizeof x, 1, f) == 1) v.push_back(x);
    if (f) std::fclose(f);
    return v;
}

int main(int argc, char **argv)
{
    if (argc < 4) {
        std::printf("usage: host_band <xclbin> <unit 1..3> <first>:<count> ...\n");
        return 1;
    }
    const int SZY = 1080, SZX = 1920, BAND = 31, NROWS = SZY / 4, P = 24;
    const int NBANDS = (NROWS + BAND - 1) / BAND, NQMEM = (BAND + 14) / 5 * (SZX / 4) * 36;
    const auto x = readBin("x.bin"), prm = readBin("prm.bin"), w = readBin("wmem.bin"),
               b = readBin("bmem.bin"), l = readBin("ln.bin"), st = readBin("stats.bin"),
               yref = readBin("yref.bin");
    if ((int)x.size() != SZY * SZX) {
        std::printf("unexpected size of x: %zu\n", x.size());
        return 1;
    }
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

    xrt::device dev(0);
    xrt::xclbin xb{std::string(argv[1])};
    auto uuid = dev.load_xclbin(xb);
    // an earlier kernel keeps the angles in memory (the qmem argument)
    bool qmemPort = false;
    for (const auto &k : xb.get_kernels()) {
        if (k.get_name() == "salsun2d_band_df") qmemPort = (k.get_num_args() == 11);
    }
    // units: "2" for one, "123" for several run at once (bands split among them)
    const std::string units = argv[2];
    struct Unit {
        xrt::kernel k;
        xrt::bo bx, bp, bw, bb, bl, bs, by, bso, bq;
    };
    std::vector<Unit> us;
    // SPACER=<MB>: a buffer of that size first in each bank, so the buffers
    // land at addresses of the bank not used before
    std::vector<xrt::bo> spacers;
    for (char ch : units) {
        Unit u;
        u.k = xrt::kernel(dev, uuid, "salsun2d_band_df:{salsun2d_band_df_" + std::string(1, ch) + "}");
        if (const char *sp = std::getenv("SPACER"))
            spacers.emplace_back(dev, (size_t)std::atoi(sp) << 20, u.k.group_id(0));
        u.bx = xrt::bo(dev, x.size() * 4, u.k.group_id(0));
        u.bp = xrt::bo(dev, prm.size() * 4, u.k.group_id(1));
        u.bw = xrt::bo(dev, words.size(), u.k.group_id(2));
        u.bb = xrt::bo(dev, bias.size() * 8, u.k.group_id(3));
        u.bl = xrt::bo(dev, l.size() * 4, u.k.group_id(4));
        u.bs = xrt::bo(dev, st.size() * 4, u.k.group_id(5));
        u.by = xrt::bo(dev, x.size() * 4, u.k.group_id(6));
        u.bso = xrt::bo(dev, st.size() * 4, u.k.group_id(7));
        if (qmemPort) u.bq = xrt::bo(dev, (size_t)NQMEM * 64, u.k.group_id(8));
        float *px = u.bx.map<float *>(), *pp = u.bp.map<float *>(), *pl = u.bl.map<float *>(),
              *ps = u.bs.map<float *>();
        for (size_t i = 0; i < x.size(); i++) px[i] = (float)x[i];
        for (size_t i = 0; i < prm.size(); i++) pp[i] = (float)prm[i];
        for (size_t i = 0; i < l.size(); i++) pl[i] = (float)l[i];
        for (size_t i = 0; i < st.size(); i++) ps[i] = (float)st[i];
        std::memcpy(u.bw.map<uint8_t *>(), words.data(), words.size());
        std::memcpy(u.bb.map<int64_t *>(), bias.data(), bias.size() * 8);
        for (auto *bo : {&u.bx, &u.bp, &u.bw, &u.bb, &u.bl, &u.bs}) bo->sync(XCL_BO_SYNC_BO_TO_DEVICE);
        if (qmemPort && std::getenv("QFILL")) {   // the angle memory filled with a pattern (default: as allocated)
            std::memset(u.bq.map<uint8_t *>(), std::atoi(std::getenv("QFILL")), (size_t)NQMEM * 64);
            u.bq.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        }
        us.push_back(std::move(u));
    }

    // each argument: one run; with several units, the bands first..first+count-1
    // split into contiguous ranges as in salsun2d_band_df_mex
    for (int a = 3; a < argc; a++) {
        int first = 0, count = 0;
        std::sscanf(argv[a], "%d:%d", &first, &count);
        const int n = (int)us.size();
        std::vector<int> f(n), c(n);
        for (int j = 0, kk = first; j < n; j++) {
            c[j] = count / n + (j < count % n ? 1 : 0);
            f[j] = kk;
            kk += c[j];
        }
        std::vector<xrt::run> runs;
        std::chrono::steady_clock::time_point t0;
        // MEXLIKE: as salsun2d_band_df_mex, the output left as it is and each
        // unit started right after its frame is transferred
        const bool mexlike = std::getenv("MEXLIKE") != nullptr;
        for (int j = 0; j < n && !mexlike; j++) {
            float *py = us[j].by.map<float *>();
            std::fill(py, py + x.size(), std::numeric_limits<float>::quiet_NaN());
            us[j].by.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        }
        for (int j = 0; j < n; j++) {
            Unit &u = us[j];
            if (mexlike) {
                u.bx.sync(XCL_BO_SYNC_BO_TO_DEVICE);
                u.bs.sync(XCL_BO_SYNC_BO_TO_DEVICE);
            }
            if (j == 0) t0 = std::chrono::steady_clock::now();
            runs.push_back(qmemPort ? u.k(u.bx, u.bp, u.bw, u.bb, u.bl, u.bs, u.by, u.bso, u.bq, f[j], c[j])
                                    : u.k(u.bx, u.bp, u.bw, u.bb, u.bl, u.bs, u.by, u.bso, f[j], c[j]));
        }
        for (int j = 0; j < n; j++) {
            const auto st0 = runs[j].wait();
            std::printf("unit %c: state %d after %.3f s; ", units[j], (int)st0, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
            if (const char *qd = qmemPort ? std::getenv("QDUMP") : nullptr) {   // the angles of the last band to a file
                us[j].bq.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
                FILE *fq = std::fopen((std::string(qd) + "." + std::to_string(a) + "." + units[j]).c_str(), "wb");
                std::fwrite(us[j].bq.map<uint8_t *>(), 1, (size_t)NQMEM * 64, fq);
                std::fclose(fq);
            }
            us[j].by.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
            const float *py = us[j].by.map<float *>();
            std::printf("unit %c, bands %d..%d:", units[j], f[j], f[j] + c[j] - 1);
            for (int kk = f[j]; kk < f[j] + c[j] && kk < NBANDS; kk++) {
                int r0 = kk * BAND;
                if (r0 + BAND > NROWS) r0 = NROWS - BAND;
                double e = 0;
                long nan = 0;
                for (int cc = 0; cc < SZX; cc++) {
                    for (int r = 4 * r0; r < 4 * (r0 + BAND); r++) {
                        const size_t i = (size_t)cc * SZY + r;
                        if (std::isnan(py[i])) nan++;
                        else e = std::fmax(e, std::fabs(py[i] - yref[i]));
                    }
                }
                std::printf(" [%d] %.3g%s", kk, e, nan ? (" nan " + std::to_string(nan)).c_str() : "");
            }
            std::printf("\n");
        }
    }
    return 0;
}
