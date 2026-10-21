// salsun2d_band_mex.cpp
//
// MEX gateway that runs salsun2d_band_kernel (one frame, band by band)
// on an Alveo U250 through XRT.
//
//   [Y, statsOut] = salsun2d_band_mex(xclbinPath, X, W, statsIn, bandRows)
//   salsun2d_band_mex('release')
//
//   X        : one frame, szy x szx single (column-major)
//   W        : single vector of packed parameters (salsun2d_pack_params)
//   statsIn  : NDec x NEst x 2 single: mu and sigma per channel and estimator
//   Y        : reconstructed frame, szy x szx single
//   statsOut : NDec x NEst x 2 single: sums and sums of squares of the
//              estimator input channels over the blocks of this frame
//
//   bandRows : block rows per band of the xclbin (BAND of the Makefile);
//              needed to split the bands among several compute units
//
//   X can also hold one frame per compute unit, szy x szx x n (n up to
//   the number of units), with statsIn NDec x NEst x 2 x n: frame j runs
//   whole on unit j with its statistics, at the same time as the others,
//   and Y and statsOut have the same sizes (e.g. a kernel built for one
//   band of the whole frame height, BAND = szy/4 and HALO = 0).
//
// The frame size must match the xclbin. The device, the compute units
// and the buffers are opened on the first call and reused while the same
// xclbin path and frame size are given; the parameters are transferred
// only when they change. With several compute units (Makefile NCU) the
// bands of a frame are split into contiguous ranges, one per unit, run at
// the same time; the rows of each band are taken from the output of its
// unit and the sums of the units are added. A kernel without the band
// range arguments (older xclbins) runs the whole frame on one unit.
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

#include <algorithm>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace {

const char *KERNEL_NAME = "salsun2d_band_kernel";

struct Unit {
    xrt::kernel kernel;
    xrt::bo x, w, sin, y, sout;
};

struct Accelerator {
    std::string xclbinPath;
    xrt::device device;
    bool bandArgs = false;          // kernel takes first_band, num_bands
    std::vector<Unit> units;
    size_t nBytesX = 0, nBytesW = 0, nBytesS = 0;
    std::vector<float> wCache;      // parameters on the devices
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
            if (k.get_name() != KERNEL_NAME) {
                continue;
            }
            a->bandArgs = k.get_num_args() >= 7;
            for (const auto &cu : k.get_cus()) {
                // CU names come as "kernel:instance"
                std::string cuName = cu.get_name();
                const size_t colon = cuName.rfind(':');
                if (colon != std::string::npos) {
                    cuName = cuName.substr(colon + 1);
                }
                Unit u;
                u.kernel = xrt::kernel(a->device, uuid, std::string(KERNEL_NAME) + ":{" + cuName + "}");
                a->units.push_back(std::move(u));
            }
        }
        if (a->units.empty()) {
            throw std::runtime_error("no compute unit of salsun2d_band_kernel in the xclbin");
        }
        if (!a->bandArgs) {
            a->units.resize(1);      // older kernel: whole frame on one unit
        }
        accel = std::move(a);
        mexAtExit(release);
    }
    return *accel;
}

void allocate(Accelerator &a, size_t nBytesX, size_t nBytesW, size_t nBytesS)
{
    if (a.nBytesX == nBytesX && a.nBytesW == nBytesW && a.nBytesS == nBytesS) {
        return;
    }
    for (auto &u : a.units) {
        u.x = xrt::bo(a.device, nBytesX, u.kernel.group_id(0));
        u.w = xrt::bo(a.device, nBytesW, u.kernel.group_id(1));
        u.sin = xrt::bo(a.device, nBytesS, u.kernel.group_id(2));
        u.y = xrt::bo(a.device, nBytesX, u.kernel.group_id(3));
        u.sout = xrt::bo(a.device, nBytesS, u.kernel.group_id(4));
    }
    a.nBytesX = nBytesX;
    a.nBytesW = nBytesW;
    a.nBytesS = nBytesS;
    a.wCache.clear();
}

std::string getString(const mxArray *arr, const char *what)
{
    if (!mxIsChar(arr)) {
        mexErrMsgIdAndTxt("salsun2d_band_mex:invalidInput", "%s must be a character vector.", what);
    }
    char *buf = mxArrayToUTF8String(arr);
    std::string str(buf);
    mxFree(buf);
    return str;
}

void checkSingle(const mxArray *arr, const char *what)
{
    if (!mxIsSingle(arr) || mxIsComplex(arr)) {
        mexErrMsgIdAndTxt("salsun2d_band_mex:invalidInput", "%s must be a real single array.", what);
    }
}

// First block row of band k (as in the kernel: the last band is moved up)
int bandStart(int k, int band, int nRows)
{
    int r0 = k * band;
    return (r0 + band > nRows) ? nRows - band : r0;
}

} // namespace

// Whether W differs from the parameters on the device (and remember it)
bool wChanged(Accelerator &a, const mxArray *w)
{
    const float *wp = mxGetSingles(w);
    const size_t n = mxGetNumberOfElements(w);
    const bool changed = a.wCache.size() != n || std::memcmp(a.wCache.data(), wp, n * sizeof(float)) != 0;
    if (changed) {
        a.wCache.assign(wp, wp + n);
    }
    return changed;
}

// One whole frame per compute unit, all units at the same time
void runFrames(Accelerator &a, const mxArray *x, const mxArray *stats, mxArray *y, mxArray *statsOut,
               size_t nFrames, size_t nBytesX, size_t nBytesS, size_t nStats, bool newW)
{
    const float *xp = mxGetSingles(x);
    const float *sp = mxGetSingles(stats);
    // New parameters go to every unit, also those without a frame in this
    // call: the cache marks them as loaded on all units
    if (newW) {
        for (auto &u : a.units) {
            std::memcpy(u.w.map<float *>(), a.wCache.data(), a.wCache.size() * sizeof(float));
            u.w.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        }
    }
    std::vector<xrt::run> runs;
    for (size_t j = 0; j < nFrames; j++) {
        Unit &u = a.units[j];
        std::memcpy(u.x.map<float *>(), xp + j * (nBytesX / sizeof(float)), nBytesX);
        std::memcpy(u.sin.map<float *>(), sp + j * nStats, nBytesS);
        u.x.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        u.sin.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        runs.push_back(u.kernel(u.x, u.w, u.sin, u.y, u.sout, 0, 1 << 20));   // all bands
    }
    float *yp = mxGetSingles(y);
    float *so = mxGetSingles(statsOut);
    for (size_t j = 0; j < nFrames; j++) {
        Unit &u = a.units[j];
        runs[j].wait();
        u.y.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        u.sout.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        std::memcpy(yp + j * (nBytesX / sizeof(float)), u.y.map<float *>(), nBytesX);
        std::memcpy(so + j * nStats, u.sout.map<float *>(), nBytesS);
    }
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs == 1 && getString(prhs[0], "Command") == "release") {
        release();
        return;
    }
    if ((nrhs != 4 && nrhs != 5) || nlhs > 2) {
        mexErrMsgIdAndTxt("salsun2d_band_mex:invalidUsage",
                          "Usage: [Y, statsOut] = salsun2d_band_mex(xclbinPath, X, W, statsIn, bandRows) "
                          "or salsun2d_band_mex('release')");
    }
    const std::string xclbinPath = getString(prhs[0], "xclbinPath");
    const mxArray *x = prhs[1];
    const mxArray *w = prhs[2];
    const mxArray *stats = prhs[3];
    checkSingle(x, "X");
    checkSingle(w, "W");
    checkSingle(stats, "statsIn");
    if (mxGetNumberOfDimensions(x) > 3) {
        mexErrMsgIdAndTxt("salsun2d_band_mex:invalidInput", "X must be szy x szx or szy x szx x n.");
    }
    const int bandRows = (nrhs == 5) ? static_cast<int>(mxGetScalar(prhs[4])) : 0;
    const size_t szy = mxGetDimensions(x)[0], szx = mxGetDimensions(x)[1];
    const size_t nFrames = (mxGetNumberOfDimensions(x) == 3) ? mxGetDimensions(x)[2] : 1;
    if (mxGetNumberOfElements(stats) % nFrames != 0) {
        mexErrMsgIdAndTxt("salsun2d_band_mex:invalidInput", "statsIn must hold the statistics of every frame.");
    }
    const size_t nBytesX = szy * szx * sizeof(float);                  // one frame
    const size_t nBytesW = mxGetNumberOfElements(w) * sizeof(float);
    const size_t nStats = mxGetNumberOfElements(stats) / nFrames;      // one frame
    const size_t nBytesS = nStats * sizeof(float);

    plhs[0] = mxCreateUninitNumericArray(mxGetNumberOfDimensions(x), const_cast<mwSize *>(mxGetDimensions(x)),
                                         mxSINGLE_CLASS, mxREAL);
    mxArray *statsOut = mxCreateNumericArray(mxGetNumberOfDimensions(stats),
                                             const_cast<mwSize *>(mxGetDimensions(stats)),
                                             mxSINGLE_CLASS, mxREAL);
    if (nlhs > 1) {
        plhs[1] = statsOut;
    }

    try {
        Accelerator &a = acquire(xclbinPath);
        allocate(a, nBytesX, nBytesW, nBytesS);
        const size_t nUnits = a.units.size();
        if (nFrames > 1) {
            if (nFrames > nUnits || !a.bandArgs) {
                mexErrMsgIdAndTxt("salsun2d_band_mex:invalidInput",
                                  "%d frames need as many compute units with band ranges (the xclbin has %d).",
                                  (int)nFrames, (int)nUnits);
            }
            runFrames(a, x, stats, plhs[0], statsOut, nFrames, nBytesX, nBytesS, nStats, wChanged(a, w));
            if (nlhs <= 1) {
                mxDestroyArray(statsOut);
            }
            return;
        }
        if (nUnits > 1 && bandRows <= 0) {
            mexErrMsgIdAndTxt("salsun2d_band_mex:invalidInput",
                              "bandRows is required for an xclbin with %d compute units.", (int)nUnits);
        }
        const int nRows = static_cast<int>(szy / 4);
        const int nBands = (bandRows > 0) ? (nRows + bandRows - 1) / bandRows : 0;

        // Parameters: transfer only when they change
        const float *wp = mxGetSingles(w);
        const bool newW = wChanged(a, w);
        // Band ranges: contiguous, the first units take one more band
        std::vector<int> first(nUnits, 0), count(nUnits, 0);
        if (a.bandArgs) {
            const int nb = (nBands > 0) ? nBands : 1 << 20;   // one unit: all bands
            int k = 0;
            for (size_t j = 0; j < nUnits; j++) {
                count[j] = (nBands > 0) ? nBands / (int)nUnits + ((int)j < nBands % (int)nUnits ? 1 : 0) : nb;
                first[j] = k;
                k += count[j];
            }
        }
        std::vector<xrt::run> runs;
        for (size_t j = 0; j < nUnits; j++) {
            Unit &u = a.units[j];
            std::memcpy(u.x.map<float *>(), mxGetSingles(x), nBytesX);
            std::memcpy(u.sin.map<float *>(), mxGetSingles(stats), nBytesS);
            u.x.sync(XCL_BO_SYNC_BO_TO_DEVICE);
            u.sin.sync(XCL_BO_SYNC_BO_TO_DEVICE);
            if (newW) {
                std::memcpy(u.w.map<float *>(), wp, nBytesW);
                u.w.sync(XCL_BO_SYNC_BO_TO_DEVICE);
            }
            if (a.bandArgs) {
                runs.push_back(u.kernel(u.x, u.w, u.sin, u.y, u.sout, first[j], count[j]));
            } else {
                runs.push_back(u.kernel(u.x, u.w, u.sin, u.y, u.sout));
            }
        }
        float *yOut = mxGetSingles(plhs[0]);
        float *sOut = mxGetSingles(statsOut);
        std::fill(sOut, sOut + nStats, 0.0f);
        for (size_t j = 0; j < nUnits; j++) {
            Unit &u = a.units[j];
            runs[j].wait();
            u.y.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
            u.sout.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
            const float *yd = u.y.map<float *>();
            if (nUnits == 1) {
                std::memcpy(yOut, yd, nBytesX);
            } else {
                // Rows of the bands of this unit, column by column
                for (int k = first[j]; k < first[j] + count[j]; k++) {
                    const size_t p0 = 4 * (size_t)bandStart(k, bandRows, nRows);
                    const size_t np = 4 * (size_t)bandRows;
                    for (size_t c = 0; c < szx; c++) {
                        std::memcpy(yOut + c * szy + p0, yd + c * szy + p0, np * sizeof(float));
                    }
                }
            }
            const float *sd = u.sout.map<float *>();
            // stats_out holds sums in its first half and sums of squares in
            // the second, the same layout for every unit: add them
            for (size_t i = 0; i < nStats; i++) {
                sOut[i] += sd[i];
            }
        }
    } catch (const std::exception &e) {
        release();
        mexErrMsgIdAndTxt("salsun2d_band_mex:xrtError", "XRT error: %s", e.what());
    }
    if (nlhs <= 1) {
        mxDestroyArray(statsOut);
    }
}
