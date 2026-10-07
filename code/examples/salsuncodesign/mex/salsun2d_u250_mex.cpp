// salsun2d_u250_mex.cpp
//
// MEX gateway that runs salsun2d_kernel on an Alveo U250 through XRT.
//
//   Y = salsun2d_u250_mex(xclbinPath, X, W)
//   [Y, T] = salsun2d_u250_mex(xclbinPath, X, W)
//   salsun2d_u250_mex('release')
//
//   X : single array of size szy x szx x N (N frames, column-major)
//   W : single vector of packed parameters (salsun2d_pack_params)
//   Y : reconstructed frames, same size as X
//   T : elapsed seconds [host-to-device, kernel, device-to-host]
//
// The frame size and the number of parameters must match those the
// xclbin was built for. Kernels with the DDR angle buffer (5 arguments)
// get a work buffer allocated here; earlier kernels (4 arguments) do not. The device and kernel are opened on the first
// call and reused while the same xclbin path is given.
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

#include <chrono>
#include <cstring>
#include <memory>
#include <string>

namespace {

const char *KERNEL_NAME = "salsun2d_kernel";
const size_t NTHETA = 168;   // rows of the angle buffer (salsun2d_hls_layout)
const size_t STRIDE = 4;     // block size

struct Accelerator {
    std::string xclbinPath;
    xrt::device device;
    xrt::kernel kernel;
    bool hasTheta = false;   // kernel takes the DDR angle buffer (5 arguments)
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
        const xrt::xclbin xclbin(xclbinPath);
        auto uuid = a->device.load_xclbin(xclbin);
        a->kernel = xrt::kernel(a->device, uuid, KERNEL_NAME);
        // Earlier kernels keep the angles on chip and have no theta argument
        for (const auto &k : xclbin.get_kernels()) {
            if (k.get_name() == KERNEL_NAME) {
                a->hasTheta = k.get_num_args() == 5;
            }
        }
        accel = std::move(a);
        mexAtExit(release);
    }
    return *accel;
}

std::string getString(const mxArray *arr, const char *what)
{
    if (!mxIsChar(arr)) {
        mexErrMsgIdAndTxt("salsun2d_u250_mex:invalidInput", "%s must be a character vector.", what);
    }
    char *buf = mxArrayToUTF8String(arr);
    std::string str(buf);
    mxFree(buf);
    return str;
}

void checkSingle(const mxArray *arr, const char *what)
{
    if (!mxIsSingle(arr) || mxIsComplex(arr)) {
        mexErrMsgIdAndTxt("salsun2d_u250_mex:invalidInput", "%s must be a real single array.", what);
    }
}

using Clock = std::chrono::steady_clock;

double seconds(Clock::time_point t0, Clock::time_point t1)
{
    return std::chrono::duration<double>(t1 - t0).count();
}

} // namespace

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs == 1 && getString(prhs[0], "Command") == "release") {
        release();
        return;
    }
    if (nrhs != 3 || nlhs > 2) {
        mexErrMsgIdAndTxt("salsun2d_u250_mex:invalidUsage",
                          "Usage: [Y, T] = salsun2d_u250_mex(xclbinPath, X, W) or salsun2d_u250_mex('release')");
    }
    const std::string xclbinPath = getString(prhs[0], "xclbinPath");
    const mxArray *x = prhs[1];
    const mxArray *w = prhs[2];
    checkSingle(x, "X");
    checkSingle(w, "W");

    const mwSize nDims = mxGetNumberOfDimensions(x);
    const mwSize *dims = mxGetDimensions(x);
    const size_t frameSize = dims[0] * dims[1];
    const int numFrames = nDims > 2 ? static_cast<int>(dims[2]) : 1;
    const size_t nBytesX = frameSize * numFrames * sizeof(float);
    const size_t nBytesW = mxGetNumberOfElements(w) * sizeof(float);
    const size_t nBytesTheta = NTHETA * (dims[0] / STRIDE) * (dims[1] / STRIDE) * sizeof(float);

    plhs[0] = mxCreateUninitNumericArray(nDims, const_cast<mwSize *>(dims), mxSINGLE_CLASS, mxREAL);
    double elapsedLocal[3] = {0.0, 0.0, 0.0};
    double *elapsed = elapsedLocal;
    if (nlhs > 1) {
        plhs[1] = mxCreateDoubleMatrix(1, 3, mxREAL);
        elapsed = mxGetDoubles(plhs[1]);
    }
    if (numFrames == 0 || frameSize == 0) {
        return;
    }

    try {
        Accelerator &a = acquire(xclbinPath);
        xrt::bo boX(a.device, nBytesX, a.kernel.group_id(0));
        xrt::bo boW(a.device, nBytesW, a.kernel.group_id(1));
        xrt::bo boY(a.device, nBytesX, a.kernel.group_id(2));
        // Work buffer for the angles of one frame; written and read by the kernel
        xrt::bo boTheta;
        if (a.hasTheta) {
            boTheta = xrt::bo(a.device, nBytesTheta, a.kernel.group_id(3));
        }

        auto t0 = Clock::now();
        std::memcpy(boX.map<float *>(), mxGetSingles(x), nBytesX);
        std::memcpy(boW.map<float *>(), mxGetSingles(w), nBytesW);
        boX.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        boW.sync(XCL_BO_SYNC_BO_TO_DEVICE);

        auto t1 = Clock::now();
        xrt::run run = a.hasTheta ? a.kernel(boX, boW, boY, boTheta, numFrames)
                                  : a.kernel(boX, boW, boY, numFrames);
        run.wait();

        auto t2 = Clock::now();
        boY.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        std::memcpy(mxGetSingles(plhs[0]), boY.map<float *>(), nBytesX);
        auto t3 = Clock::now();

        elapsed[0] = seconds(t0, t1);
        elapsed[1] = seconds(t1, t2);
        elapsed[2] = seconds(t2, t3);
    } catch (const std::exception &e) {
        release();
        mexErrMsgIdAndTxt("salsun2d_u250_mex:xrtError", "XRT error: %s", e.what());
    }
}
