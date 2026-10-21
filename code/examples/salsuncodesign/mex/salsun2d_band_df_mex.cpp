// salsun2d_band_df_mex.cpp
//
// MEX gateway that runs salsun2d_band_df (proto/band_df.cpp: the band
// kernel with the dataflow estimator) on an Alveo U250 through XRT.
//
//   [Y, sums] = salsun2d_band_df_mex(xclbinPath, X, prm, W, B, LN, statsIn, bandRows, halo)
//   salsun2d_band_df_mex('release')
//
//   X        : one frame, szy x szx single, or one frame per compute unit,
//              szy x szx x n
//   prm      : the small parameters (368 single, proto/band_df.h)
//   W        : the estimator weights, 24 x NWORDS double (pack_est_df),
//              rounded to [18 16]
//   B        : the estimator biases, NBIAS double, rounded to [48 27]
//   LN       : the LayerNorm parameters, 2 x NLN single (gamma; beta)
//   statsIn  : NDec x NEst x 2 single (mu, sigma), or x n for n frames
//   bandRows : block rows per band of the xclbin (BAND), halo its HALO
//   Y        : the reconstructed frame(s)
//   sums     : NDec x NEst x 2 single: sums and sums of squares of the
//              estimator input channels over the blocks of each frame
//
// With one frame, the bands are split into contiguous ranges, one per
// compute unit, run at the same time; with n frames, frame j runs whole on
// unit j. The device, the units and the buffers are opened on the first
// call and reused; the parameters are transferred only when they change.
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

#include "mex.h"

#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>
#include <xrt/experimental/xrt_xclbin.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace {

const char *KERNEL_NAME = "salsun2d_band_df";
constexpr int P = 24;            // weights per word
constexpr int WORD = 64;         // bytes per weight or angle word
constexpr int NQW = 6;           // angle words per block

struct Unit {
    xrt::kernel kernel;
    xrt::bo x, prm, w, b, ln, sin, y, sout, q;
};

struct Accelerator {
    std::string xclbinPath;
    xrt::device device;
    std::vector<Unit> units;
    size_t nBytesX = 0, nBytesQ = 0, nWords = 0, nBias = 0, nLn = 0;
    std::vector<uint8_t> paramCache;   // the parameters on the devices
};

std::unique_ptr<Accelerator> accel;

void release()
{
    accel.reset();
}

Accelerator &acquire(const std::string &xclbinPath)
{
    if (!accel || accel->xclbinPath != xclbinPath) {
        accel.reset();
        auto a = std::make_unique<Accelerator>();
        a->xclbinPath = xclbinPath;
        a->device = xrt::device(0);
        xrt::xclbin xb(xclbinPath);
        auto uuid = a->device.load_xclbin(xb);
        for (const auto &k : xb.get_kernels()) {
            if (k.get_name() != KERNEL_NAME) continue;
            for (const auto &cu : k.get_cus()) {
                std::string cuName = cu.get_name();       // "kernel:instance"
                const size_t colon = cuName.rfind(':');
                if (colon != std::string::npos) cuName = cuName.substr(colon + 1);
                Unit u;
                u.kernel = xrt::kernel(a->device, uuid, std::string(KERNEL_NAME) + ":{" + cuName + "}");
                a->units.push_back(std::move(u));
            }
        }
        if (a->units.empty()) throw std::runtime_error("no compute unit of salsun2d_band_df in the xclbin");
        accel = std::move(a);
        mexAtExit(release);
    }
    return *accel;
}

// Buffers of every unit (kernel arguments: src, prm, wmem, bmem, lnmem,
// stats_in, dst, stats_out, qmem, first_band, num_bands)
void allocate(Accelerator &a, size_t nBytesX, size_t nBytesQ, size_t nWords, size_t nBias, size_t nLn)
{
    if (a.nBytesX == nBytesX && a.nBytesQ == nBytesQ && a.nWords == nWords && a.nBias == nBias &&
        a.nLn == nLn) {
        return;
    }
    for (auto &u : a.units) {
        u.x = xrt::bo(a.device, nBytesX, u.kernel.group_id(0));
        u.prm = xrt::bo(a.device, 368 * sizeof(float), u.kernel.group_id(1));
        u.w = xrt::bo(a.device, nWords * WORD, u.kernel.group_id(2));
        u.b = xrt::bo(a.device, nBias * sizeof(int64_t), u.kernel.group_id(3));
        u.ln = xrt::bo(a.device, nLn * sizeof(float), u.kernel.group_id(4));
        u.sin = xrt::bo(a.device, 160 * sizeof(float), u.kernel.group_id(5));
        u.y = xrt::bo(a.device, nBytesX, u.kernel.group_id(6));
        u.sout = xrt::bo(a.device, 160 * sizeof(float), u.kernel.group_id(7));
        u.q = xrt::bo(a.device, nBytesQ, u.kernel.group_id(8));
    }
    a.nBytesX = nBytesX;
    a.nBytesQ = nBytesQ;
    a.nWords = nWords;
    a.nBias = nBias;
    a.nLn = nLn;
    a.paramCache.clear();
}

std::string getString(const mxArray *arr, const char *what)
{
    if (!mxIsChar(arr)) mexErrMsgIdAndTxt("salsun2d_band_df_mex:invalidInput", "%s must be a character vector.", what);
    char *buf = mxArrayToUTF8String(arr);
    std::string str(buf);
    mxFree(buf);
    return str;
}

void check(const mxArray *arr, bool single, const char *what)
{
    if ((single ? !mxIsSingle(arr) : !mxIsDouble(arr)) || mxIsComplex(arr)) {
        mexErrMsgIdAndTxt("salsun2d_band_df_mex:invalidInput", "%s must be a real %s array.", what,
                          single ? "single" : "double");
    }
}

// Weight words: 24 values of [18 16] as raw 18-bit fields at bits 18p
std::vector<uint8_t> packWeights(const double *w, size_t nWords)
{
    std::vector<uint8_t> out(nWords * WORD, 0);
    for (size_t i = 0; i < nWords; i++) {
        uint64_t *word = reinterpret_cast<uint64_t *>(out.data() + i * WORD);
        for (int p = 0; p < P; p++) {
            const int64_t v = std::llround(w[i * P + p] * 65536.0);
            const uint64_t raw = static_cast<uint64_t>(v) & 0x3FFFFu;
            const int bit = 18 * p, idx = bit / 64, off = bit % 64;
            word[idx] |= raw << off;
            if (off + 18 > 64) word[idx + 1] |= raw >> (64 - off);
        }
    }
    return out;
}

// First block row of band k (as in the kernel: the last band is moved up)
int bandStart(int k, int band, int nRows)
{
    const int r0 = k * band;
    return (r0 + band > nRows) ? nRows - band : r0;
}

} // namespace

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs == 1 && getString(prhs[0], "Command") == "release") {
        release();
        return;
    }
    if (nrhs != 9 || nlhs > 2) {
        mexErrMsgIdAndTxt("salsun2d_band_df_mex:invalidUsage",
                          "Usage: [Y, sums] = salsun2d_band_df_mex(xclbinPath, X, prm, W, B, LN, statsIn, "
                          "bandRows, halo) or salsun2d_band_df_mex('release')");
    }
    const std::string xclbinPath = getString(prhs[0], "xclbinPath");
    const mxArray *x = prhs[1], *prm = prhs[2], *w = prhs[3], *b = prhs[4], *ln = prhs[5], *stats = prhs[6];
    check(x, true, "X");
    check(prm, true, "prm");
    check(w, false, "W");
    check(b, false, "B");
    check(ln, true, "LN");
    check(stats, true, "statsIn");
    if (mxGetNumberOfElements(prm) != 368 || mxGetM(w) != P) {
        mexErrMsgIdAndTxt("salsun2d_band_df_mex:invalidInput", "prm must have 368 values and W 24 rows.");
    }
    const int bandRows = static_cast<int>(mxGetScalar(prhs[7]));
    const int halo = static_cast<int>(mxGetScalar(prhs[8]));
    const size_t szy = mxGetDimensions(x)[0], szx = mxGetDimensions(x)[1];
    const size_t nFrames = (mxGetNumberOfDimensions(x) == 3) ? mxGetDimensions(x)[2] : 1;
    if (mxGetNumberOfElements(stats) != 160 * nFrames) {
        mexErrMsgIdAndTxt("salsun2d_band_df_mex:invalidInput", "statsIn must be 16 x 5 x 2 per frame.");
    }
    const size_t nBytesX = szy * szx * sizeof(float);
    const int nRows = static_cast<int>(szy / 4), nCols = static_cast<int>(szx / 4);
    const size_t nBytesQ = static_cast<size_t>(bandRows + 2 * halo) * nCols * NQW * WORD;
    const size_t nWords = mxGetN(w), nBias = mxGetNumberOfElements(b), nLn = mxGetNumberOfElements(ln);
    const int nBands = (nRows + bandRows - 1) / bandRows;

    plhs[0] = mxCreateUninitNumericArray(mxGetNumberOfDimensions(x), const_cast<mwSize *>(mxGetDimensions(x)),
                                         mxSINGLE_CLASS, mxREAL);
    mxArray *sumsOut = mxCreateNumericArray(mxGetNumberOfDimensions(stats),
                                            const_cast<mwSize *>(mxGetDimensions(stats)), mxSINGLE_CLASS, mxREAL);
    try {
        Accelerator &a = acquire(xclbinPath);
        allocate(a, nBytesX, nBytesQ, nWords, nBias, nLn);
        const size_t nUnits = a.units.size();
        if (nFrames > nUnits) {
            mexErrMsgIdAndTxt("salsun2d_band_df_mex:invalidInput", "%d frames need as many compute units (%d).",
                              (int)nFrames, (int)nUnits);
        }
        // Parameters: converted and transferred only when they change
        std::vector<uint8_t> params(368 * sizeof(float) + nWords * P * sizeof(double) +
                                    nBias * sizeof(double) + nLn * sizeof(float));
        {
            uint8_t *p = params.data();
            std::memcpy(p, mxGetSingles(prm), 368 * sizeof(float));
            p += 368 * sizeof(float);
            std::memcpy(p, mxGetDoubles(w), nWords * P * sizeof(double));
            p += nWords * P * sizeof(double);
            std::memcpy(p, mxGetDoubles(b), nBias * sizeof(double));
            p += nBias * sizeof(double);
            std::memcpy(p, mxGetSingles(ln), nLn * sizeof(float));
        }
        if (params != a.paramCache) {
            const std::vector<uint8_t> words = packWeights(mxGetDoubles(w), nWords);
            std::vector<int64_t> bias(nBias);
            for (size_t i = 0; i < nBias; i++) bias[i] = std::llround(mxGetDoubles(b)[i] * 134217728.0);  // 2^27
            for (auto &u : a.units) {
                std::memcpy(u.prm.map<float *>(), mxGetSingles(prm), 368 * sizeof(float));
                std::memcpy(u.w.map<uint8_t *>(), words.data(), words.size());
                std::memcpy(u.b.map<int64_t *>(), bias.data(), nBias * sizeof(int64_t));
                std::memcpy(u.ln.map<float *>(), mxGetSingles(ln), nLn * sizeof(float));
                u.prm.sync(XCL_BO_SYNC_BO_TO_DEVICE);
                u.w.sync(XCL_BO_SYNC_BO_TO_DEVICE);
                u.b.sync(XCL_BO_SYNC_BO_TO_DEVICE);
                u.ln.sync(XCL_BO_SYNC_BO_TO_DEVICE);
            }
            a.paramCache = std::move(params);
        }
        const float *xp = mxGetSingles(x), *sp = mxGetSingles(stats);
        float *yp = mxGetSingles(plhs[0]), *so = mxGetSingles(sumsOut);
        std::vector<xrt::run> runs;
        std::vector<int> first, count;
        if (nFrames > 1) {
            for (size_t j = 0; j < nFrames; j++) {
                Unit &u = a.units[j];
                std::memcpy(u.x.map<float *>(), xp + j * szy * szx, nBytesX);
                std::memcpy(u.sin.map<float *>(), sp + j * 160, 160 * sizeof(float));
                u.x.sync(XCL_BO_SYNC_BO_TO_DEVICE);
                u.sin.sync(XCL_BO_SYNC_BO_TO_DEVICE);
                runs.push_back(u.kernel(u.x, u.prm, u.w, u.b, u.ln, u.sin, u.y, u.sout, u.q, 0, nBands));
            }
            for (size_t j = 0; j < nFrames; j++) {
                Unit &u = a.units[j];
                runs[j].wait();
                u.y.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
                u.sout.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
                std::memcpy(yp + j * szy * szx, u.y.map<float *>(), nBytesX);
                std::memcpy(so + j * 160, u.sout.map<float *>(), 160 * sizeof(float));
            }
        } else {
            // Bands of one frame split into contiguous ranges, the first units one more
            const size_t nUsed = std::min(nUnits, static_cast<size_t>(nBands));
            int k = 0;
            for (size_t j = 0; j < nUsed; j++) {
                const int c = nBands / (int)nUsed + ((int)j < nBands % (int)nUsed ? 1 : 0);
                first.push_back(k);
                count.push_back(c);
                k += c;
                Unit &u = a.units[j];
                std::memcpy(u.x.map<float *>(), xp, nBytesX);
                std::memcpy(u.sin.map<float *>(), sp, 160 * sizeof(float));
                u.x.sync(XCL_BO_SYNC_BO_TO_DEVICE);
                u.sin.sync(XCL_BO_SYNC_BO_TO_DEVICE);
                runs.push_back(u.kernel(u.x, u.prm, u.w, u.b, u.ln, u.sin, u.y, u.sout, u.q, first[j], count[j]));
            }
            std::fill(so, so + 160, 0.0f);
            for (size_t j = 0; j < nUsed; j++) {
                Unit &u = a.units[j];
                runs[j].wait();
                u.y.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
                u.sout.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
                const float *yd = u.y.map<float *>();
                for (int kk = first[j]; kk < first[j] + count[j]; kk++) {
                    const size_t p0 = 4 * (size_t)bandStart(kk, bandRows, nRows), np = 4 * (size_t)bandRows;
                    for (size_t c = 0; c < szx; c++) std::memcpy(yp + c * szy + p0, yd + c * szy + p0, np * sizeof(float));
                }
                const float *sd = u.sout.map<float *>();
                for (int i = 0; i < 160; i++) so[i] += sd[i];
            }
        }
    } catch (const std::exception &e) {
        release();
        mexErrMsgIdAndTxt("salsun2d_band_df_mex:xrtError", "XRT error: %s", e.what());
    }
    if (nlhs > 1) {
        plhs[1] = sumsOut;
    } else {
        mxDestroyArray(sumsOut);
    }
}
