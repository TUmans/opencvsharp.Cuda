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


// ---------- cuda_createHoughSegmentDetector --------------------------------------------------
CVAPI(ExceptionStatus) cuda_createHoughSegmentDetector(float rho, float theta, int minLineLength, int maxLineGap, int maxLines, cv::Ptr<cv::cuda::HoughSegmentDetector> **returnValue)
{
    BEGIN_WRAP
    auto ptr = cv::cuda::createHoughSegmentDetector(rho, theta, minLineLength, maxLineGap, maxLines);
    *returnValue = new cv::Ptr<cv::cuda::HoughSegmentDetector>(ptr);
    END_WRAP
}

// ---------- cuda_HoughSegmentDetector_get --------------------------------------------------
CVAPI(ExceptionStatus) cuda_HoughSegmentDetector_get(cv::Ptr<cv::cuda::HoughSegmentDetector> *ptr, cv::cuda::HoughSegmentDetector **returnValue)
{
    BEGIN_WRAP
    *returnValue = ptr->get();
    END_WRAP
}

// ---------- cuda_HoughSegmentDetector_delete --------------------------------------------------
CVAPI(ExceptionStatus) cuda_HoughSegmentDetector_delete(cv::Ptr<cv::cuda::HoughSegmentDetector> *ptr)
{
    BEGIN_WRAP
    delete ptr;
    END_WRAP
}

// ---------- cuda_HoughSegmentDetector_detect --------------------------------------------------
CVAPI(ExceptionStatus) cuda_HoughSegmentDetector_detect(cv::cuda::HoughSegmentDetector *obj, cv::_InputArray *src, cv::_OutputArray *lines, cv::cuda::Stream *stream)
{
    BEGIN_WRAP
    cv::cuda::Stream &streamRef = stream ? *stream : cv::cuda::Stream::Null();
    obj->detect(*src, *lines, streamRef);
    END_WRAP
}
