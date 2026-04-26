#pragma once

// -----------------------------------------------------------------------
// OpenCvSharpExtern – cv::cuda arithmetic wrappers
// These are the C-linkage functions that the C# P/Invoke layer calls.
// Each function catches cv::Exception, stores it, and returns an
// ExceptionStatus so managed code can rethrow it as a .NET exception.
// -----------------------------------------------------------------------

#include "include_opencv.h"
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudalegacy.hpp>


// ---------- calcOpticalFlowBM --------------------------------------------------
CVAPI(ExceptionStatus) cuda_calcOpticalFlowBM(
    cv::cuda::GpuMat *prev, cv::cuda::GpuMat *curr, cv::Size block_size, cv::Size shift_size, cv::Size max_range,
    int use_previous, cv::cuda::GpuMat *velx, cv::cuda::GpuMat *vely, cv::cuda::GpuMat *buf, cv::cuda::Stream *stream)
{
    BEGIN_WRAP
    cv::cuda::Stream &streamRef = stream ? *stream : cv::cuda::Stream::Null();
    cv::cuda::calcOpticalFlowBM(*prev, *curr, block_size, shift_size, max_range, use_previous != 0,
                                *velx, *vely, *buf, streamRef);
    END_WRAP
}
