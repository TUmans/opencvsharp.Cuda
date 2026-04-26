#pragma once

// -----------------------------------------------------------------------
// OpenCvSharpExtern – cv::cuda arithmetic wrappers
// These are the C-linkage functions that the C# P/Invoke layer calls.
// Each function catches cv::Exception, stores it, and returns an
// ExceptionStatus so managed code can rethrow it as a .NET exception.
// -----------------------------------------------------------------------

#include "include_opencv.h"
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudawarping.hpp>

CVAPI(ExceptionStatus) cuda_buildWarpAffineMaps(cv::_InputArray *M, int inverse, cv::Size dsize, cv::_OutputArray *xmap, cv::_OutputArray *ymap, cv::cuda::Stream *stream)
{
    BEGIN_WRAP
    cv::cuda::Stream &streamRef = stream ? *stream : cv::cuda::Stream::Null();
    cv::cuda::buildWarpAffineMaps(*M, inverse != 0, dsize, *xmap, *ymap, streamRef);
    END_WRAP
}

CVAPI(ExceptionStatus) cuda_buildWarpPerspectiveMaps(cv::_InputArray *M, int inverse, cv::Size dsize, cv::_OutputArray *xmap, cv::_OutputArray *ymap, cv::cuda::Stream *stream)
{
    BEGIN_WRAP
    cv::cuda::Stream &streamRef = stream ? *stream : cv::cuda::Stream::Null();
    cv::cuda::buildWarpPerspectiveMaps(*M, inverse != 0, dsize, *xmap, *ymap, streamRef);
    END_WRAP
}
