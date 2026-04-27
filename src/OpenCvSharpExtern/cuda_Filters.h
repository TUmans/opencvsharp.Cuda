#pragma once

// -----------------------------------------------------------------------
// OpenCvSharpExtern – cv::cuda arithmetic wrappers
// These are the C-linkage functions that the C# P/Invoke layer calls.
// Each function catches cv::Exception, stores it, and returns an
// ExceptionStatus so managed code can rethrow it as a .NET exception.
// -----------------------------------------------------------------------

#include "include_opencv.h"
#include <opencv2/cudafilters.hpp>

CVAPI(ExceptionStatus) cuda_Filter_get(cv::Ptr<cv::cuda::Filter> *ptr, cv::cuda::Filter **returnValue)
{
    BEGIN_WRAP
    *returnValue= ptr->get();
    END_WRAP
}

CVAPI(ExceptionStatus) cuda_Filter_delete(cv::Ptr<cv::cuda::Filter> *ptr)
{
    BEGIN_WRAP
    delete ptr;
    END_WRAP
}

CVAPI(ExceptionStatus) cuda_Filter_apply(cv::cuda::Filter *obj, cv::_InputArray *src, cv::_OutputArray *dst, cv::cuda::Stream *stream)
{
    BEGIN_WRAP
    cv::cuda::Stream &streamRef = stream ? *stream : cv::cuda::Stream::Null();
    obj->apply(*src, *dst, streamRef);
    END_WRAP
}

CVAPI(ExceptionStatus) cuda_createBoxFilter(int srcType, int dstType, cv::Size ksize, cv::Point anchor,  int borderMode, cv::Scalar borderVal, cv::Ptr<cv::cuda::Filter> **returnValue)
{
    BEGIN_WRAP
    auto ptr = cv::cuda::createBoxFilter(srcType, dstType, ksize, anchor, borderMode, borderVal);
    *returnValue = new cv::Ptr<cv::cuda::Filter>(ptr);
    END_WRAP
}

CVAPI(ExceptionStatus) cuda_createBoxMaxFilter( int srcType, cv::Size ksize, cv::Point anchor,  int borderMode, cv::Scalar borderVal, cv::Ptr<cv::cuda::Filter> **returnValue)
{
    BEGIN_WRAP
    auto ptr = cv::cuda::createBoxMaxFilter(srcType, ksize, anchor, borderMode, borderVal);
    *returnValue = new cv::Ptr<cv::cuda::Filter>(ptr);
    END_WRAP
}

CVAPI(ExceptionStatus) cuda_createBoxMinFilter( int srcType, cv::Size ksize, cv::Point anchor,  int borderMode, cv::Scalar borderVal, cv::Ptr<cv::cuda::Filter> **returnValue)
{
    BEGIN_WRAP
    auto ptr = cv::cuda::createBoxMinFilter(srcType, ksize, anchor, borderMode, borderVal);
    *returnValue = new cv::Ptr<cv::cuda::Filter>(ptr);
    END_WRAP
}

CVAPI(ExceptionStatus) cuda_createColumnSumFilter(int srcType, int dstType, int ksize, int anchor, int borderMode, cv::Scalar borderVal, cv::Ptr<cv::cuda::Filter> **returnValue)
{
    BEGIN_WRAP
    auto ptr = cv::cuda::createColumnSumFilter(srcType, dstType, ksize, anchor, borderMode, borderVal);
    *returnValue = new cv::Ptr<cv::cuda::Filter>(ptr);
    END_WRAP
}

CVAPI(ExceptionStatus) cuda_createDerivFilter(int srcType, int dstType, int dx, int dy, int ksize, int normalize, double scale, int rowBorderMode, int columnBorderMode, cv::Ptr<cv::cuda::Filter> **returnValue)
{
    BEGIN_WRAP
    auto ptr = cv::cuda::createDerivFilter(
        srcType, dstType, dx, dy, ksize, normalize != 0, scale, rowBorderMode, columnBorderMode);
    *returnValue = new cv::Ptr<cv::cuda::Filter>(ptr);
    END_WRAP
}
