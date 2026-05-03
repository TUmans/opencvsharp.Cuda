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

// ---------- cuda_createHoughLinesDetector --------------------------------------------------
CVAPI(ExceptionStatus) cuda_createHoughLinesDetector(float rho, float theta, int threshold, int doSort, int maxLines, cv::Ptr<cv::cuda::HoughLinesDetector> **returnValue)
{
    BEGIN_WRAP
    auto ptr = cv::cuda::createHoughLinesDetector(rho, theta, threshold, doSort != 0, maxLines);
    *returnValue = new cv::Ptr<cv::cuda::HoughLinesDetector>(ptr);
    END_WRAP
}

// ---------- cuda_HoughLinesDetector_get --------------------------------------------------
CVAPI(ExceptionStatus) cuda_HoughLinesDetector_get(cv::Ptr<cv::cuda::HoughLinesDetector> *ptr, cv::cuda::HoughLinesDetector **returnValue)
{
    BEGIN_WRAP
    *returnValue = ptr->get();
    END_WRAP
}

// ---------- cuda_HoughLinesDetector_delete --------------------------------------------------
CVAPI(ExceptionStatus) cuda_HoughLinesDetector_delete(cv::Ptr<cv::cuda::HoughLinesDetector> *ptr)
{
    BEGIN_WRAP
    delete ptr;
    END_WRAP
}

// ---------- cuda_HoughLinesDetector_detect --------------------------------------------------
CVAPI(ExceptionStatus) cuda_HoughLinesDetector_detect(cv::cuda::HoughLinesDetector *obj, cv::_InputArray *src, cv::_OutputArray *lines, cv::cuda::Stream *stream)
{
    BEGIN_WRAP
    cv::cuda::Stream &streamRef = stream ? *stream : cv::cuda::Stream::Null();
    obj->detect(*src, *lines, streamRef);
    END_WRAP
}
