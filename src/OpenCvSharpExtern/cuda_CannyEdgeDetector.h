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

CVAPI(ExceptionStatus) cuda_CannyEdgeDetector_getAppertureSize(cv::cuda::CannyEdgeDetector *obj, int *val)
{
    BEGIN_WRAP
    *val = obj->getAppertureSize();
    END_WRAP
}

CVAPI(ExceptionStatus) cuda_CannyEdgeDetector_setAppertureSize(cv::cuda::CannyEdgeDetector *obj, int val)
{
    BEGIN_WRAP
    obj->setAppertureSize(val);
    END_WRAP
}

CVAPI(ExceptionStatus) cuda_CannyEdgeDetector_getHighThreshold(cv::cuda::CannyEdgeDetector *obj, double *val)
{
    BEGIN_WRAP
    *val = obj->getHighThreshold();
    END_WRAP
}

CVAPI(ExceptionStatus) cuda_CannyEdgeDetector_setHighThreshold(cv::cuda::CannyEdgeDetector *obj, double val)
{
    BEGIN_WRAP
    obj->setHighThreshold(val);
    END_WRAP
}

CVAPI(ExceptionStatus) cuda_CannyEdgeDetector_getL2Gradient(cv::cuda::CannyEdgeDetector *obj, int *val)
{
    BEGIN_WRAP
    *val = obj->getL2Gradient() ? 1 : 0;
    END_WRAP
}

CVAPI(ExceptionStatus) cuda_CannyEdgeDetector_setL2Gradient(cv::cuda::CannyEdgeDetector *obj, int val)
{
    BEGIN_WRAP
    obj->setL2Gradient(val != 0);
    END_WRAP
}

CVAPI(ExceptionStatus) cuda_CannyEdgeDetector_getLowThreshold(cv::cuda::CannyEdgeDetector *obj, double *val)
{
    BEGIN_WRAP
    *val = obj->getLowThreshold();
    END_WRAP
}

CVAPI(ExceptionStatus) cuda_CannyEdgeDetector_setLowThreshold(cv::cuda::CannyEdgeDetector *obj, double val)
{
    BEGIN_WRAP
    obj->setLowThreshold(val);
    END_WRAP
}
