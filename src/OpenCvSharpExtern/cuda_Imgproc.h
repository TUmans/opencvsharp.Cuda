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

// ---------- createCannyEdgeDetector --------------------------------------------------
CVAPI(ExceptionStatus) cuda_createCannyEdgeDetector(double low_thresh, double high_thresh, int apperture_size, int L2gradient,  cv::Ptr<cv::cuda::CannyEdgeDetector> **returnValue)
{
    BEGIN_WRAP
    auto ptr = cv::cuda::createCannyEdgeDetector(low_thresh, high_thresh, apperture_size, L2gradient != 0);
    *returnValue = new cv::Ptr<cv::cuda::CannyEdgeDetector>(ptr);
    END_WRAP
}

// ---------- CannyEdgeDetector_get --------------------------------------------------
CVAPI(ExceptionStatus) cuda_CannyEdgeDetector_get(cv::Ptr<cv::cuda::CannyEdgeDetector> *ptr, cv::cuda::CannyEdgeDetector **returnValue)
{
    BEGIN_WRAP
    *returnValue = ptr->get();
    END_WRAP
}

// ---------- CannyEdgeDetector_delete --------------------------------------------------
CVAPI(ExceptionStatus) cuda_CannyEdgeDetector_delete(cv::Ptr<cv::cuda::CannyEdgeDetector> *ptr)
{
    BEGIN_WRAP
    delete ptr;
    END_WRAP
}

// ---------- CannyEdgeDetector_detect --------------------------------------------------
CVAPI(ExceptionStatus) cuda_CannyEdgeDetector_detect(cv::cuda::CannyEdgeDetector *obj, cv::_InputArray *image, cv::_OutputArray *edges, cv::cuda::Stream *stream)
{
    BEGIN_WRAP
    cv::cuda::Stream &streamRef = stream ? *stream : cv::cuda::Stream::Null();
    obj->detect(*image, *edges, streamRef);
    END_WRAP
}

// ---------- CannyEdgeDetector_detect_dxdy --------------------------------------------------
CVAPI(ExceptionStatus) cuda_CannyEdgeDetector_detect_dxdy(cv::cuda::CannyEdgeDetector *obj, cv::_InputArray *dx, cv::_InputArray *dy, cv::_OutputArray *edges, cv::cuda::Stream *stream)
{
    BEGIN_WRAP
    cv::cuda::Stream &streamRef = stream ? *stream : cv::cuda::Stream::Null();
    obj->detect(*dx, *dy, *edges, streamRef);
    END_WRAP
}

// ---------- createCLAHE --------------------------------------------------
CVAPI(ExceptionStatus) cuda_createCLAHE( double clipLimit, cv::Size tileGridSize, cv::Ptr<cv::cuda::CLAHE> **returnValue)
{
    BEGIN_WRAP
    auto ptr = cv::cuda::createCLAHE(clipLimit, tileGridSize);
    *returnValue = new cv::Ptr<cv::cuda::CLAHE>(ptr);
    END_WRAP
}

// ---------- CLAHE_get --------------------------------------------------
CVAPI(ExceptionStatus) cuda_CLAHE_get(cv::Ptr<cv::cuda::CLAHE> *ptr, cv::cuda::CLAHE **returnValue)
{
    BEGIN_WRAP
    *returnValue = ptr->get();
    END_WRAP
}

// ---------- CLAHE_delete --------------------------------------------------
CVAPI(ExceptionStatus) cuda_CLAHE_delete(cv::Ptr<cv::cuda::CLAHE> *ptr)
{
    BEGIN_WRAP
    delete ptr;
    END_WRAP
}

// ---------- CLAHE_apply --------------------------------------------------
CVAPI(ExceptionStatus) cuda_CLAHE_apply(cv::cuda::CLAHE *obj, cv::_InputArray *src, cv::_OutputArray *dst, cv::cuda::Stream *stream)
{
    BEGIN_WRAP
    cv::cuda::Stream &streamRef = stream ? *stream : cv::cuda::Stream::Null();
    obj->apply(*src, *dst, streamRef);
    END_WRAP
}
