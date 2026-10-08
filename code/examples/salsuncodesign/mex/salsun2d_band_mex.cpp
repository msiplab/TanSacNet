// salsun2d_band_mex.cpp
//
// MEX gateway that runs salsun2d_band_kernel (one frame, band by band)
// on an Alveo U250 through XRT.
//
//   [Y, statsOut] = salsun2d_band_mex(xclbinPath, X, W, statsIn)
//   salsun2d_band_mex('release')
//
//   X        : one frame, szy x szx single (column-major)
//   W        : single vector of packed parameters (salsun2d_pack_params)
//   statsIn  : NDec x NEst x 2 single: mu and sigma per channel and estimator
//   Y        : reconstructed frame, szy x szx single
//   statsOut : NDec x NEst x 2 single: sums and sums of squares of the
//              estimator input channels over the blocks of this frame
//
// The frame size must match the xclbin. The device and kernel are opened
// on the first call and reused while the same xclbin path is given.
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

#include <cstring>
#include <memory>
#include <string>

namespace {

const char *KERNEL_NAME = "salsun2d_band_kernel";

struct Accelerator {
    std::string xclbinPath;
    xrt::device device;
    xrt::kernel kernel;
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
        auto uuid = a->device.load_xclbin(xclbinPath);
        a->kernel = xrt::kernel(a->device, uuid, KERNEL_NAME);
        accel = std::move(a);
        mexAtExit(release);
    }
    return *accel;
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

} // namespace

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs == 1 && getString(prhs[0], "Command") == "release") {
        release();
        return;
    }
    if (nrhs != 4 || nlhs > 2) {
        mexErrMsgIdAndTxt("salsun2d_band_mex:invalidUsage",
                          "Usage: [Y, statsOut] = salsun2d_band_mex(xclbinPath, X, W, statsIn) "
                          "or salsun2d_band_mex('release')");
    }
    const std::string xclbinPath = getString(prhs[0], "xclbinPath");
    const mxArray *x = prhs[1];
    const mxArray *w = prhs[2];
    const mxArray *stats = prhs[3];
    checkSingle(x, "X");
    checkSingle(w, "W");
    checkSingle(stats, "statsIn");
    if (mxGetNumberOfDimensions(x) != 2) {
        mexErrMsgIdAndTxt("salsun2d_band_mex:invalidInput", "X must be one frame (szy x szx).");
    }
    const size_t nBytesX = mxGetNumberOfElements(x) * sizeof(float);
    const size_t nBytesW = mxGetNumberOfElements(w) * sizeof(float);
    const size_t nBytesS = mxGetNumberOfElements(stats) * sizeof(float);

    plhs[0] = mxCreateUninitNumericArray(2, const_cast<mwSize *>(mxGetDimensions(x)), mxSINGLE_CLASS, mxREAL);
    mxArray *statsOut = mxCreateNumericArray(mxGetNumberOfDimensions(stats),
                                             const_cast<mwSize *>(mxGetDimensions(stats)),
                                             mxSINGLE_CLASS, mxREAL);
    if (nlhs > 1) {
        plhs[1] = statsOut;
    }

    try {
        Accelerator &a = acquire(xclbinPath);
        xrt::bo boX(a.device, nBytesX, a.kernel.group_id(0));
        xrt::bo boW(a.device, nBytesW, a.kernel.group_id(1));
        xrt::bo boSin(a.device, nBytesS, a.kernel.group_id(2));
        xrt::bo boY(a.device, nBytesX, a.kernel.group_id(3));
        xrt::bo boSout(a.device, nBytesS, a.kernel.group_id(4));

        std::memcpy(boX.map<float *>(), mxGetSingles(x), nBytesX);
        std::memcpy(boW.map<float *>(), mxGetSingles(w), nBytesW);
        std::memcpy(boSin.map<float *>(), mxGetSingles(stats), nBytesS);
        boX.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        boW.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        boSin.sync(XCL_BO_SYNC_BO_TO_DEVICE);

        xrt::run run = a.kernel(boX, boW, boSin, boY, boSout);
        run.wait();

        boY.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        boSout.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        std::memcpy(mxGetSingles(plhs[0]), boY.map<float *>(), nBytesX);
        std::memcpy(mxGetSingles(statsOut), boSout.map<float *>(), nBytesS);
    } catch (const std::exception &e) {
        release();
        mexErrMsgIdAndTxt("salsun2d_band_mex:xrtError", "XRT error: %s", e.what());
    }
    if (nlhs <= 1) {
        mxDestroyArray(statsOut);
    }
}
