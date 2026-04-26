#pragma once

// -----------------------------------------------------------------------
// OpenCvSharpExtern – cv::cuda arithmetic wrappers
// These are the C-linkage functions that the C# P/Invoke layer calls.
// Each function catches cv::Exception, stores it, and returns an
// ExceptionStatus so managed code can rethrow it as a .NET exception.
// -----------------------------------------------------------------------

#include "include_opencv.h"
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>

// ---------- Alpha Composite --------------------------------------------------
CVAPI(ExceptionStatus) cuda_alphaComp(cv::_InputArray *img1, cv::_InputArray *img2, cv::_OutputArray *dst, int alpha_op, cv::cuda::Stream *stream)
{
    BEGIN_WRAP
    cv::cuda::Stream &streamRef = stream ? *stream : cv::cuda::Stream::Null();
    cv::cuda::alphaComp(*img1, *img2, *dst, alpha_op, streamRef);
    END_WRAP
}

// ---------- BilateralFilter --------------------------------------------------
CVAPI(ExceptionStatus) cuda_bilateralFilter( cv::_InputArray *src, cv::_OutputArray *dst, int kernel_size, float sigma_color, float sigma_spatial, int borderMode, cv::cuda::Stream *stream)
{
    BEGIN_WRAP
    cv::cuda::Stream &streamRef = stream ? *stream : cv::cuda::Stream::Null();
    cv::cuda::bilateralFilter(*src, *dst, kernel_size, sigma_color, sigma_spatial, borderMode, streamRef);
    END_WRAP
}

// ---------- blendLinear --------------------------------------------------
CVAPI(ExceptionStatus) cuda_blendLinear(cv::_InputArray *img1, cv::_InputArray *img2, cv::_InputArray *weights1, cv::_InputArray *weights2, cv::_OutputArray *result, cv::cuda::Stream *stream)
{
    BEGIN_WRAP
    cv::cuda::Stream &streamRef = stream ? *stream : cv::cuda::Stream::Null();
    cv::cuda::blendLinear(*img1, *img2, *weights1, *weights2, *result, streamRef);
    END_WRAP
}

// ---------- calcHist --------------------------------------------------
CVAPI(ExceptionStatus) cuda_calcHist(cv::_InputArray *src, cv::_InputArray *mask, cv::_OutputArray *hist, cv::cuda::Stream *stream)
{
    BEGIN_WRAP
    cv::cuda::Stream &streamRef = stream ? *stream : cv::cuda::Stream::Null();
    cv::cuda::calcHist(*src, mask ? *mask : cv::noArray(), *hist, streamRef);
    END_WRAP
}
